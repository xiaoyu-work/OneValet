"""
Cloud Storage Agent - Search, browse, share, and manage cloud storage files

This agent handles cloud storage operations across Google Drive, OneDrive, and Dropbox:
- Search files by keyword
- List recent files
- Get file info
- Get download links
- Share files (requires approval)
- Check storage usage
"""
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from onevalet import valet, StandardAgent, InputField, AgentStatus, AgentResult, Message, ApprovalResult
from onevalet.constants import STORAGE_SERVICES

logger = logging.getLogger(__name__)


@valet(domain="productivity", requires_service=list(STORAGE_SERVICES))
class CloudStorageAgent(StandardAgent):
    """Search and manage files in cloud storage (Dropbox, Google Drive, OneDrive). Use when the user asks about their files or wants to share/upload."""

    action = InputField(
        prompt="What would you like to do?",
        description="Action: search, recent, info, download, share, usage",
    )
    query = InputField(
        prompt="What are you looking for?",
        description="Search query or file name",
        required=False,
    )
    provider = InputField(
        prompt="Which service?",
        description="google, onedrive, dropbox, or all",
        required=False,
    )
    target = InputField(
        prompt="Share with whom?",
        description="Email address for sharing",
        required=False,
    )

    def __init__(self, tenant_id: str = "", llm_client=None, **kwargs):
        super().__init__(
            tenant_id=tenant_id,
            llm_client=llm_client,
            **kwargs,
        )
        self._share_filename = None
        self._share_file_id = None
        self._share_provider_account = None

    def needs_approval(self) -> bool:
        action = self.collected_fields.get("action", "")
        return action == "share"

    async def extract_fields(self, user_input: str) -> Dict[str, Any]:
        """Extract cloud storage action and parameters from user input."""
        if not self.llm_client:
            return {"action": "search", "query": user_input}

        try:
            prompt = f"""Extract cloud storage action from the user's message.

User message: "{user_input}"

Determine the action and parameters:
- "find my report" -> action: "search", query: "report"
- "recent files" -> action: "recent"
- "share this with john@example.com" -> action: "share", target: "john@example.com"
- "how much space do I have" -> action: "usage"
- "download that PDF" -> action: "download", query: "PDF"
- "find report on Dropbox" -> action: "search", query: "report", provider: "dropbox"
- "what files do I have" -> action: "recent"
- "file info for budget.xlsx" -> action: "info", query: "budget.xlsx"

Return JSON with these fields (omit if not mentioned):
{{
  "action": "search|recent|info|download|share|usage",
  "query": "",
  "provider": "google|onedrive|dropbox|all",
  "target": ""
}}

Return ONLY the JSON object, no explanations.

JSON Output:"""

            result = await self.llm_client.chat_completion(
                messages=[
                    {"role": "system", "content": "You extract cloud storage actions from text and return JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format="json_object",
                enable_thinking=False,
            )

            content = result.content.strip()
            extracted = json.loads(content)

            if not extracted or "action" not in extracted:
                extracted = {"action": "search", "query": user_input}

            # Clean empty values
            result_dict = {}
            for field in ("action", "query", "provider", "target"):
                value = extracted.get(field, "")
                if isinstance(value, str):
                    value = value.strip()
                if value:
                    result_dict[field] = value

            if "action" not in result_dict:
                result_dict["action"] = "search"
                result_dict["query"] = user_input

            logger.info(f"Extracted cloud storage fields: {result_dict}")
            return result_dict

        except Exception as e:
            logger.error(f"Field extraction failed: {e}", exc_info=True)
            return {"action": "search", "query": user_input}

    # ===== State Handlers =====

    async def on_initializing(self, msg: Message) -> AgentResult:
        """Extract fields and route to appropriate state."""
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        fields = self.collected_fields
        action = fields.get("action", "")

        # Share action needs approval
        if action == "share":
            target = fields.get("target")
            query = fields.get("query")

            if not target:
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_INPUT,
                    raw_message="Who would you like to share the file with? (email address)",
                    missing_fields=["target"],
                )

            if not query:
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_INPUT,
                    raw_message="Which file would you like to share?",
                    missing_fields=["query"],
                )

            # Resolve the file first so we can show it in approval
            await self._resolve_share_file()

            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt(),
            )

        # Read-only actions go straight to running
        missing = self._get_missing_fields()
        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._get_next_prompt(),
                missing_fields=missing,
            )

        self.transition_to(AgentStatus.RUNNING)
        return await self.on_running(msg)

    async def on_waiting_for_input(self, msg: Message) -> AgentResult:
        """Continue collecting fields from user."""
        if msg:
            await self._extract_and_collect_fields(msg.get_text())

        fields = self.collected_fields
        action = fields.get("action", "")

        missing = self._get_missing_fields()
        if missing:
            return self.make_result(
                status=AgentStatus.WAITING_FOR_INPUT,
                raw_message=self._get_next_prompt(),
                missing_fields=missing,
            )

        if action == "share":
            await self._resolve_share_file()
            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt(),
            )

        self.transition_to(AgentStatus.RUNNING)
        return await self.on_running(msg)

    async def on_waiting_for_approval(self, msg: Message) -> AgentResult:
        """Handle yes/no/modify responses for share action."""
        user_input = msg.get_text() if msg else ""
        approval = await self.parse_approval_async(user_input)

        if approval == ApprovalResult.APPROVED:
            self.transition_to(AgentStatus.RUNNING)
            return await self.on_running(msg)

        elif approval == ApprovalResult.REJECTED:
            return self.make_result(
                status=AgentStatus.CANCELLED,
                raw_message="OK, cancelled the share.",
            )

        else:  # MODIFY
            await self._extract_and_collect_fields(user_input)

            missing = self._get_missing_fields()
            if missing:
                return self.make_result(
                    status=AgentStatus.WAITING_FOR_INPUT,
                    raw_message=self._get_next_prompt(),
                    missing_fields=missing,
                )

            await self._resolve_share_file()
            return self.make_result(
                status=AgentStatus.WAITING_FOR_APPROVAL,
                raw_message=self.get_approval_prompt(),
            )

    async def on_running(self, msg: Message) -> AgentResult:
        """Execute the cloud storage action."""
        from onevalet.providers.cloud_storage.resolver import CloudStorageResolver
        from onevalet.providers.cloud_storage.factory import CloudStorageProviderFactory

        fields = self.collected_fields
        action = fields.get("action", "search")
        query = fields.get("query", "")
        provider_spec = fields.get("provider", "")

        logger.info(f"Cloud storage action: {action}, query: {query}, provider: {provider_spec}")

        try:
            if action == "share":
                return await self._execute_share()

            # Resolve providers
            if provider_spec and provider_spec.lower() != "all":
                account = await CloudStorageResolver.resolve(self.tenant_id, provider_spec)
                accounts = [account] if account else []
            else:
                accounts = await CloudStorageResolver.resolve_all(self.tenant_id)

            if not accounts:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="No cloud storage accounts found. Please connect one in settings.",
                )

            if action == "search":
                return await self._execute_search(accounts, query, CloudStorageProviderFactory)
            elif action == "recent":
                return await self._execute_recent(accounts, CloudStorageProviderFactory)
            elif action == "info":
                return await self._execute_info(accounts, query, CloudStorageProviderFactory)
            elif action == "download":
                return await self._execute_download(accounts, query, CloudStorageProviderFactory)
            elif action == "usage":
                return await self._execute_usage(accounts, CloudStorageProviderFactory)
            else:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message=f"I'm not sure how to handle '{action}'. Try: search, recent, share, download, or usage.",
                )

        except Exception as e:
            logger.error(f"Cloud storage action failed: {e}", exc_info=True)
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Something went wrong accessing your cloud storage. Want to try again?",
            )

    # ===== Action Executors =====

    async def _execute_search(
        self, accounts: List[dict], query: str, factory
    ) -> AgentResult:
        """Search files across providers."""
        if not query:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="What would you like to search for?",
            )

        all_files = []
        failed_accounts = []

        for account in accounts:
            provider = factory.create_provider(account)
            if not provider:
                failed_accounts.append(self._make_failed(account, "unsupported_provider"))
                continue

            if not await provider.ensure_valid_token():
                failed_accounts.append(self._make_failed(account, "token_expired"))
                continue

            try:
                result = await provider.search_files(query=query)
                if result.get("success"):
                    files = result.get("data", [])
                    for f in files:
                        f["_provider"] = account.get("provider", "")
                        f["_provider_display"] = provider.get_provider_display_name()
                    all_files.extend(files)
                else:
                    failed_accounts.append(self._make_failed(account, "search_failed", result.get("error")))
            except Exception as e:
                logger.error(f"Search failed on {account.get('provider')}: {e}", exc_info=True)
                failed_accounts.append(self._make_failed(account, "query_failed", str(e)))

        # Sort by modified date (newest first)
        all_files.sort(key=lambda f: f.get("modified", ""), reverse=True)

        formatted = self._format_file_results(all_files, accounts, failed_accounts, f'search "{query}"')
        return self.make_result(status=AgentStatus.COMPLETED, raw_message=formatted)

    async def _execute_recent(self, accounts: List[dict], factory) -> AgentResult:
        """List recent files across providers."""
        all_files = []
        failed_accounts = []

        for account in accounts:
            provider = factory.create_provider(account)
            if not provider:
                failed_accounts.append(self._make_failed(account, "unsupported_provider"))
                continue

            if not await provider.ensure_valid_token():
                failed_accounts.append(self._make_failed(account, "token_expired"))
                continue

            try:
                result = await provider.list_recent_files()
                if result.get("success"):
                    files = result.get("data", [])
                    for f in files:
                        f["_provider"] = account.get("provider", "")
                        f["_provider_display"] = provider.get_provider_display_name()
                    all_files.extend(files)
                else:
                    failed_accounts.append(self._make_failed(account, "list_failed", result.get("error")))
            except Exception as e:
                logger.error(f"Recent files failed on {account.get('provider')}: {e}", exc_info=True)
                failed_accounts.append(self._make_failed(account, "query_failed", str(e)))

        all_files.sort(key=lambda f: f.get("modified", ""), reverse=True)

        formatted = self._format_file_results(all_files, accounts, failed_accounts, "recent files")
        return self.make_result(status=AgentStatus.COMPLETED, raw_message=formatted)

    async def _execute_info(
        self, accounts: List[dict], query: str, factory
    ) -> AgentResult:
        """Get info about a file. First search, then get details."""
        if not query:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Which file would you like info about?",
            )

        # Search for the file first
        for account in accounts:
            provider = factory.create_provider(account)
            if not provider:
                continue
            if not await provider.ensure_valid_token():
                continue

            try:
                search_result = await provider.search_files(query=query, max_results=1)
                if search_result.get("success") and search_result.get("data"):
                    file_data = search_result["data"][0]
                    file_id = file_data.get("id")
                    if file_id:
                        info_result = await provider.get_file_info(file_id)
                        if info_result.get("success"):
                            return self.make_result(
                                status=AgentStatus.COMPLETED,
                                raw_message=self._format_file_info(
                                    info_result["data"], provider.get_provider_display_name()
                                ),
                            )
            except Exception as e:
                logger.error(f"File info failed on {account.get('provider')}: {e}", exc_info=True)

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f'Could not find a file matching "{query}".',
        )

    async def _execute_download(
        self, accounts: List[dict], query: str, factory
    ) -> AgentResult:
        """Get a download link for a file."""
        if not query:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="Which file would you like to download?",
            )

        for account in accounts:
            provider = factory.create_provider(account)
            if not provider:
                continue
            if not await provider.ensure_valid_token():
                continue

            try:
                search_result = await provider.search_files(query=query, max_results=1)
                if search_result.get("success") and search_result.get("data"):
                    file_data = search_result["data"][0]
                    file_id = file_data.get("id")
                    file_name = file_data.get("name", query)
                    if file_id:
                        dl_result = await provider.get_download_link(file_id)
                        if dl_result.get("success"):
                            url = dl_result["data"].get("url", "")
                            expires = dl_result["data"].get("expires", "")
                            provider_name = provider.get_provider_display_name()
                            msg = f"[{provider_name}] {file_name}\nDownload: {url}"
                            if expires:
                                msg += f"\n(link expires: {expires})"
                            return self.make_result(
                                status=AgentStatus.COMPLETED,
                                raw_message=msg,
                            )
            except Exception as e:
                logger.error(f"Download link failed on {account.get('provider')}: {e}", exc_info=True)

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f'Could not find a file matching "{query}" to download.',
        )

    async def _execute_share(self) -> AgentResult:
        """Share a file with someone (called after approval)."""
        from onevalet.providers.cloud_storage.factory import CloudStorageProviderFactory

        fields = self.collected_fields
        target = fields.get("target", "")
        query = fields.get("query", "")

        if self._share_file_id and self._share_provider_account:
            account = self._share_provider_account
            provider = CloudStorageProviderFactory.create_provider(account)
            if not provider:
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="Sorry, I can't share files with that provider yet.",
                )

            if not await provider.ensure_valid_token():
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="I lost access to your cloud storage. Please reconnect in settings.",
                )

            try:
                result = await provider.share_file(
                    file_id=self._share_file_id,
                    email=target,
                )
                if result.get("success"):
                    share_url = result.get("data", {}).get("url", "")
                    provider_name = provider.get_provider_display_name()
                    file_name = self._share_filename or query
                    msg = f"Shared [{provider_name}] {file_name} with {target}"
                    if share_url:
                        msg += f"\nLink: {share_url}"
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=msg,
                    )
                else:
                    return self.make_result(
                        status=AgentStatus.COMPLETED,
                        raw_message=f"Couldn't share the file: {result.get('error', 'Unknown error')}",
                    )
            except Exception as e:
                logger.error(f"Share failed: {e}", exc_info=True)
                return self.make_result(
                    status=AgentStatus.COMPLETED,
                    raw_message="Something went wrong sharing the file. Want to try again?",
                )

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message=f'Could not find a file matching "{query}" to share.',
        )

    async def _execute_usage(self, accounts: List[dict], factory) -> AgentResult:
        """Show storage usage across all providers."""
        from onevalet.providers.cloud_storage.base import BaseCloudStorageProvider

        usage_parts = []
        failed_accounts = []

        for account in accounts:
            provider = factory.create_provider(account)
            if not provider:
                failed_accounts.append(self._make_failed(account, "unsupported_provider"))
                continue

            if not await provider.ensure_valid_token():
                failed_accounts.append(self._make_failed(account, "token_expired"))
                continue

            try:
                result = await provider.get_storage_usage()
                if result.get("success"):
                    data = result["data"]
                    used = BaseCloudStorageProvider.format_size(data.get("used", 0))
                    total = BaseCloudStorageProvider.format_size(data.get("total", 0))
                    percent = data.get("percent", 0)
                    provider_name = provider.get_provider_display_name()
                    usage_parts.append(f"[{provider_name}] {used} / {total} ({percent:.1f}% used)")
                else:
                    failed_accounts.append(self._make_failed(account, "usage_failed", result.get("error")))
            except Exception as e:
                logger.error(f"Usage check failed on {account.get('provider')}: {e}", exc_info=True)
                failed_accounts.append(self._make_failed(account, "query_failed", str(e)))

        if not usage_parts and not failed_accounts:
            return self.make_result(
                status=AgentStatus.COMPLETED,
                raw_message="No cloud storage accounts found. Please connect one in settings.",
            )

        response_parts = []
        if usage_parts:
            response_parts.append("Storage usage:\n")
            response_parts.extend(usage_parts)

        self._append_failed_messages(response_parts, failed_accounts)

        return self.make_result(
            status=AgentStatus.COMPLETED,
            raw_message="\n".join(response_parts),
        )

    # ===== Approval Helpers =====

    def get_approval_prompt(self) -> str:
        """Generate share confirmation prompt."""
        file_name = self._share_filename or self.collected_fields.get("query", "file")
        target = self.collected_fields.get("target", "")
        provider_name = ""
        if self._share_provider_account:
            from onevalet.providers.cloud_storage.factory import CloudStorageProviderFactory
            provider = CloudStorageProviderFactory.create_provider(self._share_provider_account)
            if provider:
                provider_name = f" on {provider.get_provider_display_name()}"

        return f"Share \"{file_name}\"{provider_name} with {target}?\n\n(yes / no / or describe changes)"

    async def parse_approval_async(self, user_input: str):
        """Parse user's approval response using LLM."""
        if not self.llm_client:
            return self.parse_approval(user_input)

        prompt = f"""The user was asked to approve sharing a file. Their response was:
"{user_input}"

Classify as one of:
- APPROVED: if they said yes, ok, sure, go ahead, confirm, etc.
- REJECTED: if they said no, cancel, nevermind, don't share, etc.
- MODIFY: if they want to change something (different file, different person, etc.)

Return ONLY one word: APPROVED, REJECTED, or MODIFY"""

        try:
            result = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                enable_thinking=False,
            )
            response = result.content.strip().upper()

            if "APPROVED" in response:
                return ApprovalResult.APPROVED
            elif "REJECTED" in response:
                return ApprovalResult.REJECTED
            else:
                return ApprovalResult.MODIFY
        except Exception as e:
            logger.error(f"Failed to parse approval: {e}")
            return ApprovalResult.MODIFY

    async def _resolve_share_file(self):
        """Find the file to share so we can show details in approval."""
        from onevalet.providers.cloud_storage.resolver import CloudStorageResolver
        from onevalet.providers.cloud_storage.factory import CloudStorageProviderFactory

        fields = self.collected_fields
        query = fields.get("query", "")
        provider_spec = fields.get("provider", "")

        if not query:
            return

        if provider_spec and provider_spec.lower() != "all":
            account = await CloudStorageResolver.resolve(self.tenant_id, provider_spec)
            accounts = [account] if account else []
        else:
            accounts = await CloudStorageResolver.resolve_all(self.tenant_id)

        for account in accounts:
            provider = CloudStorageProviderFactory.create_provider(account)
            if not provider:
                continue
            if not await provider.ensure_valid_token():
                continue

            try:
                search_result = await provider.search_files(query=query, max_results=1)
                if search_result.get("success") and search_result.get("data"):
                    file_data = search_result["data"][0]
                    self._share_file_id = file_data.get("id")
                    self._share_filename = file_data.get("name", query)
                    self._share_provider_account = account
                    return
            except Exception as e:
                logger.error(f"Could not resolve share file on {account.get('provider')}: {e}")

    # ===== Formatting =====

    def _format_file_results(
        self,
        files: List[Dict],
        searched_accounts: List[Dict],
        failed_accounts: List[Dict],
        action_label: str = "",
    ) -> str:
        """Format file search/list results."""
        if not files and not failed_accounts:
            return f"No files found for {action_label}." if action_label else "No files found."

        response_parts = []
        multi_provider = len(searched_accounts) > 1

        if not files:
            response_parts.append(f"No files found for {action_label}." if action_label else "No files found.")
        else:
            response_parts.append(f"Found {len(files)} file(s):\n")

            for i, f in enumerate(files, 1):
                name = f.get("name", "Untitled")
                size = f.get("size")
                modified = f.get("modified", "")
                provider_display = f.get("_provider_display", "")

                # Format size
                size_str = ""
                if size is not None:
                    from onevalet.providers.cloud_storage.base import BaseCloudStorageProvider
                    size_str = f" - {BaseCloudStorageProvider.format_size(size)}"

                # Format date
                date_str = ""
                if modified:
                    date_str = f" - modified {self._format_date(modified)}"

                if multi_provider and provider_display:
                    line = f"{i}. [{provider_display}] {name}{size_str}{date_str}"
                else:
                    line = f"{i}. {name}{size_str}{date_str}"

                response_parts.append(line)

        self._append_failed_messages(response_parts, failed_accounts)

        return "\n".join(response_parts)

    def _format_file_info(self, data: Dict, provider_name: str) -> str:
        """Format detailed file info."""
        from onevalet.providers.cloud_storage.base import BaseCloudStorageProvider

        name = data.get("name", "Unknown")
        file_type = data.get("type", "")
        size = data.get("size")
        modified = data.get("modified", "")
        path = data.get("path", "")
        url = data.get("url", "")
        shared = data.get("shared", False)

        parts = [f"[{provider_name}] {name}"]
        if file_type:
            parts.append(f"Type: {file_type}")
        if size is not None:
            parts.append(f"Size: {BaseCloudStorageProvider.format_size(size)}")
        if modified:
            parts.append(f"Modified: {self._format_date(modified)}")
        if path:
            parts.append(f"Path: {path}")
        if shared:
            parts.append("Shared: Yes")
        if url:
            parts.append(f"Link: {url}")

        return "\n".join(parts)

    @staticmethod
    def _format_date(date_str: str) -> str:
        """Format date string to short display format."""
        if not date_str:
            return ""
        try:
            from dateutil import parser as date_parser
            dt = date_parser.parse(date_str)
            now = datetime.now()
            if dt.year == now.year:
                return dt.strftime("%b %d").lstrip("0")
            else:
                return dt.strftime("%b %d, %Y").lstrip("0")
        except Exception:
            return date_str

    @staticmethod
    def _make_failed(account: dict, reason: str, error: str = "") -> dict:
        return {
            "provider": account.get("provider", ""),
            "email": account.get("email", ""),
            "reason": reason,
            "error": error,
        }

    @staticmethod
    def _append_failed_messages(response_parts: List[str], failed_accounts: List[Dict]):
        """Append user-friendly messages for failed accounts."""
        for failed in failed_accounts:
            provider = failed.get("provider", "")
            email = failed.get("email", "")
            reason = failed.get("reason", "unknown")
            display = email if email else provider if provider else "cloud storage"

            if reason == "token_expired":
                response_parts.append(
                    f"\nI lost access to your {display} account. Could you reconnect it in settings?"
                )
            elif reason == "unsupported_provider":
                response_parts.append(
                    f"\nSorry, I can't access {display} yet - that provider isn't supported."
                )
            else:
                response_parts.append(
                    f"\nI had trouble checking {display}. Want me to try again later?"
                )
