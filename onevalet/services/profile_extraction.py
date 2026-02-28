"""
Profile Extraction Service for OneValet

Scans user emails and extracts structured profile data using a hybrid approach:
1. Fast metadata scan (parallel)
2. Rule-based filter (keep high-priority matches)
3. LLM filter for remaining emails (evaluate titles)
4. Merge selections with diversity
5. Parallel content fetch
6. Batch LLM extraction
7. Merge & validate results

Target: ~3-4 minutes, comprehensive coverage
"""

import json
import logging
import asyncio
import base64
import re
import uuid
import httpx
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from collections import defaultdict
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

MAX_EMAILS_TO_SCAN = 5000
METADATA_BATCH_SIZE = 100
CONTENT_BATCH_SIZE = 30

LLM_FILTER_BATCH_SIZE = 100
LLM_FILTER_PARALLEL = 5

MAX_RULE_MATCHED = 500
MAX_LLM_MATCHED = 500
MAX_TOTAL_EMAILS = 800
MAX_PER_SENDER = 8

EXTRACT_BATCH_SIZE = 40
EXTRACT_PARALLEL_LIMIT = 5

# Gmail API base
GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"

# =============================================================================
# Pydantic Profile Models (inlined from koiai/models/profile.py)
# =============================================================================

VALID_LOYALTY_STATUS = {
    "gold", "platinum", "diamond", "silver", "premier", "executive",
    "elite", "preferred elite", "titanium", "ambassador", "globalist",
    "explorist", "discoverist", "1k", "premier 1k", "premier platinum",
    "premier gold", "premier silver", "million miler", "platinum pro",
}


class Identity(BaseModel):
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    birthday: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None

    @field_validator("phone", mode="before")
    @classmethod
    def validate_phone(cls, v):
        if not v:
            return None
        digits = re.sub(r"\D", "", str(v))
        if 10 <= len(digits) <= 11:
            return v
        return None

    @model_validator(mode="after")
    def build_full_name(self):
        if self.first_name and self.last_name:
            self.full_name = f"{self.first_name} {self.last_name}"
        return self


class Address(BaseModel):
    label: str
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None


class Job(BaseModel):
    employer: str
    title: Optional[str] = None
    is_current: bool = False


class Work(BaseModel):
    jobs: List[Job] = Field(default_factory=list)


class School(BaseModel):
    name: str
    degree: Optional[str] = None
    major: Optional[str] = None


class Education(BaseModel):
    schools: List[School] = Field(default_factory=list)


class LoyaltyProgram(BaseModel):
    program: str
    type: str
    number: Optional[str] = None
    status: Optional[str] = None

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v):
        if v and str(v).lower() in ("airline", "hotel"):
            return str(v).lower()
        raise ValueError("type must be 'airline' or 'hotel'")

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v):
        if not v:
            return None
        if str(v).lower().strip() in VALID_LOYALTY_STATUS:
            return v
        return None

    @field_validator("number", mode="before")
    @classmethod
    def validate_number(cls, v):
        if not v:
            return None
        v_str = str(v).strip().lower()
        if v_str in ("n/a", "null", "none", "unknown", ""):
            return None
        return str(v).strip()


class Travel(BaseModel):
    loyalty_programs: List[LoyaltyProgram] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def dedupe_programs(cls, data):
        if isinstance(data, dict) and "loyalty_programs" in data:
            programs = data.get("loyalty_programs", [])
            seen = set()
            deduped = []
            for p in programs:
                if isinstance(p, dict):
                    key = p.get("program", "").lower()
                    if key and key not in seen:
                        seen.add(key)
                        deduped.append(p)
            data["loyalty_programs"] = deduped
        return data


class Pet(BaseModel):
    name: str
    type: str


class Vehicle(BaseModel):
    make: str
    model: str
    year: Optional[int] = None
    is_current: bool = False


class Lifestyle(BaseModel):
    pets: List[Pet] = Field(default_factory=list)
    vehicles: List[Vehicle] = Field(default_factory=list)


class Person(BaseModel):
    name: str
    relationship: str
    birthday: Optional[str] = None


class Relationships(BaseModel):
    family: List[Person] = Field(default_factory=list)
    significant_other: Optional[Person] = None
    anniversary: Optional[str] = None


class UserProfile(BaseModel):
    identity: Identity = Field(default_factory=Identity)
    addresses: List[Address] = Field(default_factory=list)
    work: Work = Field(default_factory=Work)
    education: Education = Field(default_factory=Education)
    travel: Travel = Field(default_factory=Travel)
    lifestyle: Lifestyle = Field(default_factory=Lifestyle)
    relationships: Relationships = Field(default_factory=Relationships)


def _merge_profiles(existing: dict, new_data: dict) -> dict:
    """Merge new profile data into existing. Lists are deduped, scalars overwritten."""
    result = existing.copy()
    for key, value in new_data.items():
        if value is None:
            continue
        if key not in result or result[key] is None:
            result[key] = value
        elif key == "addresses" and isinstance(value, list):
            existing_addrs = result.get("addresses", [])
            if not isinstance(existing_addrs, list):
                existing_addrs = []
            seen_zips: set[str] = set()
            deduped = []
            for addr in existing_addrs:
                if isinstance(addr, dict):
                    zip_code = (addr.get("zip") or "").split("-")[0].strip()
                    if zip_code and zip_code not in seen_zips:
                        deduped.append(addr)
                        seen_zips.add(zip_code)
            for addr in value:
                if isinstance(addr, dict):
                    zip_code = (addr.get("zip") or "").split("-")[0].strip()
                    if zip_code and zip_code not in seen_zips:
                        deduped.append(addr)
                        seen_zips.add(zip_code)
            result["addresses"] = deduped
        elif isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = _merge_profiles(result[key], value)
        elif isinstance(value, list) and isinstance(result[key], list):
            existing_set = {str(item).lower() for item in result[key]}
            for item in value:
                item_str = str(item).lower()
                if item_str not in existing_set:
                    result[key].append(item)
                    existing_set.add(item_str)
        else:
            result[key] = value
    return result


# =============================================================================
# Rule-Based Filtering
# =============================================================================

IMPORTANT_DOMAINS = {
    "bankofamerica", "chase", "wellsfargo", "citi", "citibank", "usbank",
    "capitalone", "americanexpress", "amex", "discover", "pnc", "tdbank",
    "schwab", "fidelity", "vanguard", "etrade", "robinhood", "coinbase",
    "paypal", "venmo", "zelle", "wise",
    "apple", "google", "microsoft", "amazon", "meta", "facebook",
    "ebay", "etsy", "walmart", "target", "bestbuy", "costco",
    "fedex", "ups", "usps", "dhl",
    "united", "delta", "american", "southwest", "jetblue", "alaska",
    "marriott", "hilton", "hyatt", "airbnb", "booking", "expedia",
    "uber", "lyft", "doordash", "grubhub", "instacart",
    "comcast", "xfinity", "att", "verizon", "tmobile", "spectrum",
    "geico", "progressive", "statefarm", "allstate", "usaa",
    "anthem", "bluecross", "kaiser", "cigna", "aetna",
    "linkedin", "indeed", "glassdoor", "workday", "adp", "gusto",
    "netflix", "spotify", "hulu", "disney", "adobe", "dropbox",
}

SKIP_SENDER_PATTERNS = [
    r"noreply.*@.*linkedin\.com",
    r"notification.*@.*facebook",
    r"marketing@", r"newsletter@", r"promo@", r"deals@",
    r"mailer-daemon", r"postmaster@",
]

SKIP_SUBJECT_PATTERNS = [
    r"\d+% off", r"flash sale", r"limited time offer",
    r"act now", r"don't miss out", r"last chance",
    r"weekly digest", r"daily digest",
]

HIGH_VALUE_KEYWORDS = [
    r"order confirm", r"order shipped", r"delivery",
    r"receipt", r"invoice", r"payment",
    r"statement", r"balance", r"direct deposit",
    r"booking confirm", r"reservation", r"itinerary",
    r"welcome to", r"account created", r"verify",
    r"password reset", r"security alert",
    r"job offer", r"offer letter", r"interview",
    r"lease", r"rental", r"rent",
    r"insurance", r"policy", r"claim",
    r"flight", r"boarding", r"check-in",
]


def _extract_domain(email_addr: str) -> str:
    match = re.search(r"@([\w.-]+)", email_addr.lower())
    if match:
        domain = match.group(1)
        for prefix in ["mail.", "email.", "e.", "mx."]:
            if domain.startswith(prefix):
                domain = domain[len(prefix):]
        parts = domain.split(".")
        return parts[-2] if len(parts) >= 2 else parts[0]
    return ""


def _should_skip_email(email: Dict[str, Any]) -> bool:
    sender = email.get("sender", "").lower()
    subject = email.get("subject", "").lower()
    for pattern in SKIP_SENDER_PATTERNS:
        if re.search(pattern, sender):
            return True
    for pattern in SKIP_SUBJECT_PATTERNS:
        if re.search(pattern, subject):
            return True
    return False


def _rule_matches_email(email: Dict[str, Any]) -> bool:
    sender = email.get("sender", "").lower()
    subject = email.get("subject", "").lower()
    snippet = email.get("snippet", "").lower()

    domain = _extract_domain(sender)
    if domain in IMPORTANT_DOMAINS:
        return True

    combined = f"{subject} {snippet}"
    for keyword in HIGH_VALUE_KEYWORDS:
        if re.search(keyword, combined):
            return True
    return False


def _split_emails_by_rules(emails: List[Dict[str, Any]]) -> tuple:
    rule_matched = []
    needs_llm = []
    skipped = 0
    for email in emails:
        if _should_skip_email(email):
            skipped += 1
            continue
        if _rule_matches_email(email):
            rule_matched.append(email)
        else:
            needs_llm.append(email)
    logger.info(f"Rule split: {len(rule_matched)} matched, {len(needs_llm)} need LLM, {skipped} skipped")
    return rule_matched, needs_llm


def _select_diverse_emails(
    emails: List[Dict[str, Any]], max_total: int, max_per_sender: int
) -> List[Dict[str, Any]]:
    selected = []
    sender_counts: Dict[str, int] = defaultdict(int)
    for email in emails:
        if len(selected) >= max_total:
            break
        domain = _extract_domain(email.get("sender", ""))
        if sender_counts[domain] >= max_per_sender:
            continue
        selected.append(email)
        sender_counts[domain] += 1
    logger.info(f"Selected {len(selected)} emails from {len(sender_counts)} senders")
    return selected


# =============================================================================
# LLM Prompts
# =============================================================================

LLM_FILTER_PROMPT = """Review these email subjects and senders. For each email, determine if it likely contains PERSONAL INFORMATION about the recipient (user).

Personal information includes:
- Name, home address, phone number, birthday
- Employment info (job offers, company, work email)
- Education info (school, university, degree, graduation)
- Travel (flight bookings, hotel reservations, loyalty programs)
- Vehicles (insurance, registration, purchase)
- Family info (birthday cards, family events)

Rate each email:
- 1 = Likely contains personal info
- 0 = Unlikely (marketing, newsletters, promotions, general notifications)

EMAILS:
{emails}

Return JSON array with same order as input:
[{{"idx": 0, "relevant": 1}}, {{"idx": 1, "relevant": 0}}, ...]

Return ONLY the JSON array."""


EXTRACTION_PROMPT = """Extract the USER's personal profile from their emails. The user's email address is: {user_email}

CRITICAL: You are extracting information ABOUT THE USER (the email recipient), NOT about senders or companies that emailed them.

EMAILS:
{emails}

Extract into this structure:
{{
  "identity": {{
    "first_name": "user's first name",
    "last_name": "user's last name",
    "birthday": "month day",
    "email": "user's primary email",
    "phone": "user's personal phone number"
  }},
  "addresses": [
    {{"label": "home", "street": "full street address", "city": "city", "state": "state", "zip": "zip"}}
  ],
  "work": {{
    "jobs": [
      {{"employer": "Company Name", "title": "Job Title", "is_current": true/false}}
    ]
  }},
  "education": {{
    "schools": [
      {{"name": "University Name", "degree": "BS/MS/PhD/MBA", "major": "Field of Study"}}
    ]
  }},
  "travel": {{
    "loyalty_programs": [{{"program": "Program Name", "type": "airline/hotel", "number": "member number", "status": "tier level"}}]
  }},
  "lifestyle": {{
    "pets": [{{"name": "Pet Name", "type": "dog/cat/etc"}}],
    "vehicles": [{{"make": "Make", "model": "Model", "year": 2022, "is_current": true/false}}]
  }},
  "relationships": {{
    "family": [{{"name": "Name", "relationship": "mother/father/sibling", "birthday": "Month Day"}}],
    "significant_other": {{"name": "Name", "relationship": "spouse/partner", "birthday": "Month Day"}},
    "anniversary": "Month Day"
  }}
}}

=== STRICT EXTRACTION RULES ===

1. WORK - VERY STRICT:
   INCLUDE only if there's CLEAR PROOF the user is/was an EMPLOYEE:
     - Offer letters, employment contracts
     - Pay stubs, W-2, direct deposit TO user
     - 401k/benefits enrollment FROM employer's HR
     - Stock options/RSU vesting FROM employer
     - Performance reviews, PTO balance, internal HR emails

   DO NOT INCLUDE - these are NOT proof of employment:
     - Banks/financial institutions (Chase, BofA, Wells Fargo, Capital One, Citi, etc.) = CUSTOMER, not employee
     - Credit card companies = CUSTOMER
     - Stores/retailers = CUSTOMER
     - Any company just sending statements, marketing, or account notifications
     - Job applications or recruiter emails (applying != employed)

2. ADDRESSES - STRICT:
   INCLUDE: User's HOME address from:
     - Shipping addresses where user RECEIVES packages at home
     - Account profile "home address"
     - Billing address clearly marked as home/residential

   DO NOT INCLUDE:
     - Store/merchant addresses (where packages ship FROM)
     - Return addresses on shipping labels
     - Business addresses, office locations

3. VEHICLES - STRICT:
   INCLUDE only if user OWNS the vehicle:
     - Car insurance policy in user's name
     - Vehicle registration documents
     - Car purchase/financing receipts

   DO NOT INCLUDE:
     - Rental car confirmations
     - Test drive appointments
     - Dealership marketing/promotions
     - Uber/Lyft ride receipts

4. IDENTITY:
   - phone: ONLY user's personal phone (from account settings, shipping contact)
   - NEVER include: customer service numbers, business numbers, sender's phone

5. LOYALTY PROGRAMS:
   - type: MUST be "airline" or "hotel" ONLY
   - status: MUST be a tier level (Gold, Platinum, Diamond, Silver, Premier, Executive)
   - DO NOT include generic status like "Active", "Member", "Enrolled"

6. GENERAL:
   - When in doubt, OMIT the information
   - OMIT empty fields entirely - no null values, no empty arrays
   - If uncertain whether info is about user vs. a company, OMIT it

Return ONLY valid JSON."""


# =============================================================================
# Profile Extraction Service
# =============================================================================

class ProfileExtractionService:
    """
    Manages profile extraction jobs. Each job runs in-memory (no DB table).
    """

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def _update_job(self, job_id: str, updates: Dict[str, Any]):
        if job_id in self._jobs:
            self._jobs[job_id].update(updates)

    # -----------------------------------------------------------------
    # LLM helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_llm_json(text: str) -> Any:
        """Strip markdown fences and parse JSON."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())

    async def _call_llm(self, llm_client, prompt: str, max_tokens: int = 4096) -> str:
        """Call the OneValet LLM client and return raw text."""
        response = await llm_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            config={"temperature": 0.1, "max_tokens": max_tokens},
        )
        return response.content or ""

    # -----------------------------------------------------------------
    # LLM Title Filtering
    # -----------------------------------------------------------------

    async def _llm_filter_emails(
        self, llm_client, emails: List[Dict], job_id: str,
    ) -> List[Dict]:
        if not emails:
            return []

        total = len(emails)
        relevant: List[Dict] = []
        batches = [emails[i:i + LLM_FILTER_BATCH_SIZE] for i in range(0, total, LLM_FILTER_BATCH_SIZE)]
        logger.info(f"LLM filtering {total} emails in {len(batches)} batches...")

        async def process_batch(batch: List[Dict]) -> List[Dict]:
            lines = []
            for i, email in enumerate(batch):
                sender = email.get("sender", "")[:60]
                subject = email.get("subject", "")[:100]
                lines.append(f"{i}. From: {sender} | Subject: {subject}")

            prompt = LLM_FILTER_PROMPT.format(emails="\n".join(lines))
            try:
                text = await self._call_llm(llm_client, prompt)
                results = self._parse_llm_json(text)
                matched = []
                if isinstance(results, list):
                    for r in results:
                        if isinstance(r, dict) and r.get("relevant") == 1:
                            idx = r.get("idx", -1)
                            if 0 <= idx < len(batch):
                                matched.append(batch[idx])
                return matched
            except Exception as e:
                logger.error(f"LLM filter batch failed: {e}")
                return []

        for i in range(0, len(batches), LLM_FILTER_PARALLEL):
            parallel = batches[i:i + LLM_FILTER_PARALLEL]
            batch_results = await asyncio.gather(*[process_batch(b) for b in parallel])
            for matches in batch_results:
                relevant.extend(matches)

            processed = min((i + LLM_FILTER_PARALLEL) * LLM_FILTER_BATCH_SIZE, total)
            logger.info(f"LLM filtered {processed}/{total}, found {len(relevant)} relevant")
            self._update_job(job_id, {
                "progress": {
                    "phase": "llm_filtering",
                    "filtered": processed,
                    "total": total,
                    "relevant_found": len(relevant),
                },
            })

        logger.info(f"LLM filter complete: {len(relevant)} relevant from {total}")
        return relevant

    # -----------------------------------------------------------------
    # LLM Extraction
    # -----------------------------------------------------------------

    async def _extract_profiles_batch(
        self, llm_client, email_contents: List[Dict], job_id: str, user_email: str,
    ) -> List[Dict]:
        total = len(email_contents)
        batches = [email_contents[i:i + EXTRACT_BATCH_SIZE] for i in range(0, total, EXTRACT_BATCH_SIZE)]
        logger.info(f"Extracting from {total} emails in {len(batches)} batches...")

        all_profiles: List[Dict] = []

        async def process_batch(batch: List[Dict]) -> Dict:
            texts = []
            for i, email in enumerate(batch):
                texts.append(
                    f"\n=== Email {i + 1} ===\n"
                    f"From: {email.get('sender', '')}\n"
                    f"To: {email.get('to', '')}\n"
                    f"Subject: {email.get('subject', '')}\n"
                    f"Date: {email.get('date', '')}\n\n"
                    f"{email.get('body', '')}\n"
                )
            prompt = EXTRACTION_PROMPT.format(user_email=user_email, emails="\n".join(texts))
            try:
                text = await self._call_llm(llm_client, prompt, max_tokens=8192)
                return self._parse_llm_json(text)
            except Exception as e:
                logger.error(f"LLM extract batch failed: {e}")
                return {}

        for i in range(0, len(batches), EXTRACT_PARALLEL_LIMIT):
            parallel = batches[i:i + EXTRACT_PARALLEL_LIMIT]
            results = await asyncio.gather(*[process_batch(b) for b in parallel])
            for profile in results:
                if profile:
                    all_profiles.append(profile)

            processed = min((i + EXTRACT_PARALLEL_LIMIT) * EXTRACT_BATCH_SIZE, total)
            logger.info(f"Extracted {processed}/{total}")
            self._update_job(job_id, {
                "progress": {"phase": "extracting", "extracted": processed, "total": total},
            })

        return all_profiles

    # -----------------------------------------------------------------
    # Validation & Merge
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_and_clean(profile: Dict) -> Dict:
        if not profile or not isinstance(profile, dict):
            return {}
        try:
            validated = UserProfile.model_validate(profile)
            return validated.model_dump(exclude_none=True, exclude_defaults=True)
        except Exception as e:
            logger.warning(f"Profile validation failed: {e}")
            return {}

    def _merge_all(self, profiles: List[Dict]) -> Dict:
        if not profiles:
            return {}
        merged: Dict = {}
        for p in profiles:
            if p and isinstance(p, dict):
                cleaned = self._validate_and_clean(p)
                merged = _merge_profiles(merged, cleaned)
        return self._validate_and_clean(merged)

    # -----------------------------------------------------------------
    # Gmail scanning (raw API)
    # -----------------------------------------------------------------

    @staticmethod
    async def _gmail_fetch_ids(
        client: httpx.AsyncClient, access_token: str, query: str, max_results: int,
    ) -> List[str]:
        message_ids: List[str] = []
        page_token = None
        while len(message_ids) < max_results:
            params: Dict[str, Any] = {"q": query, "maxResults": min(500, max_results - len(message_ids))}
            if page_token:
                params["pageToken"] = page_token
            resp = await client.get(
                f"{GMAIL_API_BASE}/users/me/messages",
                headers={"Authorization": f"Bearer {access_token}"},
                params=params,
                timeout=60.0,
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            messages = data.get("messages", [])
            if not messages:
                break
            message_ids.extend(m["id"] for m in messages)
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        return message_ids[:max_results]

    @staticmethod
    async def _gmail_fetch_metadata_batch(
        client: httpx.AsyncClient, access_token: str, message_ids: List[str],
    ) -> List[Dict]:
        async def fetch_one(msg_id: str):
            try:
                resp = await client.get(
                    f"{GMAIL_API_BASE}/users/me/messages/{msg_id}",
                    headers={"Authorization": f"Bearer {access_token}"},
                    params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date", "To"]},
                    timeout=30.0,
                )
                if resp.status_code != 200:
                    return None
                data = resp.json()
                headers = {h["name"]: h["value"] for h in data.get("payload", {}).get("headers", [])}
                return {
                    "id": msg_id,
                    "subject": headers.get("Subject", ""),
                    "sender": headers.get("From", ""),
                    "to": headers.get("To", ""),
                    "date": headers.get("Date", ""),
                    "snippet": data.get("snippet", ""),
                }
            except Exception:
                return None

        results = await asyncio.gather(*[fetch_one(mid) for mid in message_ids], return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]

    @staticmethod
    async def _gmail_fetch_content(
        client: httpx.AsyncClient, access_token: str, msg_id: str,
    ) -> Optional[Dict]:
        try:
            resp = await client.get(
                f"{GMAIL_API_BASE}/users/me/messages/{msg_id}",
                headers={"Authorization": f"Bearer {access_token}"},
                params={"format": "full"},
                timeout=30.0,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            payload = data.get("payload", {})
            headers = {h["name"]: h["value"] for h in payload.get("headers", [])}

            body = ""
            if payload.get("body", {}).get("data"):
                body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="ignore")
            if not body and "parts" in payload:
                for part in payload["parts"]:
                    if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                        body = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
                        break
                    if "parts" in part:
                        for nested in part["parts"]:
                            if nested.get("mimeType") == "text/plain" and nested.get("body", {}).get("data"):
                                body = base64.urlsafe_b64decode(nested["body"]["data"]).decode("utf-8", errors="ignore")
                                break

            if len(body) > 3000:
                body = body[:3000] + "..."

            return {
                "id": msg_id,
                "subject": headers.get("Subject", ""),
                "sender": headers.get("From", ""),
                "to": headers.get("To", ""),
                "date": headers.get("Date", ""),
                "body": body,
            }
        except Exception:
            return None

    async def _scan_gmail(self, provider, max_emails: int = MAX_EMAILS_TO_SCAN) -> List[Dict]:
        """Scan Gmail messages for metadata using raw API."""
        all_metadata: List[Dict] = []
        seen_ids: set = set()

        try:
            if not await provider.ensure_valid_token():
                return []

            async with httpx.AsyncClient(limits=httpx.Limits(max_connections=50)) as client:
                queries = ["category:primary in:inbox", "category:primary -in:inbox -in:trash -in:spam"]
                all_ids: List[str] = []

                for query in queries:
                    ids = await self._gmail_fetch_ids(client, provider.access_token, query, max_emails // 2)
                    for mid in ids:
                        if mid not in seen_ids:
                            seen_ids.add(mid)
                            all_ids.append(mid)
                    logger.info(f"Found {len(ids)} IDs for: {query[:40]}...")

                logger.info(f"Total: {len(all_ids)} message IDs")

                for i in range(0, len(all_ids), METADATA_BATCH_SIZE):
                    batch = all_ids[i:i + METADATA_BATCH_SIZE]
                    metadata = await self._gmail_fetch_metadata_batch(client, provider.access_token, batch)
                    all_metadata.extend(metadata)
                    if (i + METADATA_BATCH_SIZE) % 500 == 0:
                        logger.info(f"Metadata: {min(i + METADATA_BATCH_SIZE, len(all_ids))}/{len(all_ids)}")

            logger.info(f"Scanned {len(all_metadata)} emails")
            return all_metadata
        except Exception as e:
            logger.error(f"Gmail scan failed: {e}", exc_info=True)
            return all_metadata

    async def _fetch_gmail_contents(self, provider, emails: List[Dict]) -> List[Dict]:
        """Fetch full email bodies from Gmail."""
        all_contents: List[Dict] = []
        try:
            if not await provider.ensure_valid_token():
                return []

            async with httpx.AsyncClient(limits=httpx.Limits(max_connections=30)) as client:
                for i in range(0, len(emails), CONTENT_BATCH_SIZE):
                    batch = emails[i:i + CONTENT_BATCH_SIZE]
                    results = await asyncio.gather(
                        *[self._gmail_fetch_content(client, provider.access_token, e["id"]) for e in batch],
                    )
                    for r in results:
                        if isinstance(r, dict) and r.get("body"):
                            all_contents.append(r)
                    if (i + CONTENT_BATCH_SIZE) % 100 == 0:
                        logger.info(f"Content: {min(i + CONTENT_BATCH_SIZE, len(emails))}/{len(emails)}")

            logger.info(f"Fetched {len(all_contents)} contents")
            return all_contents
        except Exception as e:
            logger.error(f"Content fetch failed: {e}")
            return all_contents

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def start_extraction(self, tenant_id: str, providers: list, llm_client) -> str:
        """
        Start a background extraction job.

        Args:
            tenant_id: The user/tenant ID
            providers: List of email provider instances (from EmailProviderFactory)
            llm_client: The OneValet LLM client (BaseLLMClient)

        Returns:
            job_id for status polling
        """
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "job_id": job_id,
            "tenant_id": tenant_id,
            "status": "started",
            "progress": {},
            "profile": None,
            "error": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
        }

        asyncio.create_task(self._run_extraction(job_id, tenant_id, providers, llm_client))
        return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        return self._jobs.get(job_id)

    # -----------------------------------------------------------------
    # Main pipeline
    # -----------------------------------------------------------------

    async def _run_extraction(
        self, job_id: str, tenant_id: str, providers: list, llm_client,
    ):
        start_time = datetime.now(timezone.utc)
        try:
            logger.info(f"Starting extraction for tenant {tenant_id}, job {job_id}")
            self._update_job(job_id, {"status": "scanning"})

            # Determine user email from first provider's credentials
            user_email = ""
            if providers:
                creds = getattr(providers[0], "credentials", {})
                user_email = creds.get("account_identifier", "") or creds.get("email", "")

            # Phase 1: Scan all providers
            logger.info("Phase 1: Scanning...")
            scan_start = datetime.now(timezone.utc)
            all_metadata: List[Dict] = []

            for provider in providers:
                provider_type = getattr(provider, "provider", "").lower()

                if provider_type in ("google", "gmail"):
                    metadata = await self._scan_gmail(provider)
                    all_metadata.extend(metadata)
                else:
                    logger.warning(f"Unsupported provider type for scanning: {provider_type}")

            scan_time = (datetime.now(timezone.utc) - scan_start).total_seconds()
            logger.info(f"Scanned {len(all_metadata)} emails in {scan_time:.1f}s")

            if not all_metadata:
                self._update_job(job_id, {
                    "status": "completed",
                    "profile": {},
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
                return

            # Phase 2: Rule-based split
            logger.info("Phase 2: Rule-based filtering...")
            rule_matched, needs_llm = _split_emails_by_rules(all_metadata)
            rule_matched = _select_diverse_emails(rule_matched, MAX_RULE_MATCHED, MAX_PER_SENDER)

            self._update_job(job_id, {
                "status": "llm_filtering",
                "progress": {"phase": "llm_filtering", "rule_matched": len(rule_matched), "needs_llm": len(needs_llm)},
            })

            # Phase 3: LLM filter remaining
            logger.info("Phase 3: LLM filtering remaining emails...")
            filter_start = datetime.now(timezone.utc)
            llm_matched = await self._llm_filter_emails(llm_client, needs_llm, job_id)
            filter_time = (datetime.now(timezone.utc) - filter_start).total_seconds()
            llm_matched = _select_diverse_emails(llm_matched, MAX_LLM_MATCHED, MAX_PER_SENDER)
            logger.info(f"LLM filter: {len(llm_matched)} relevant in {filter_time:.1f}s")

            # Phase 4: Merge selections
            logger.info("Phase 4: Merging selections...")
            all_selected_ids: set = set()
            merged_selection: List[Dict] = []
            for email in rule_matched:
                if email["id"] not in all_selected_ids:
                    all_selected_ids.add(email["id"])
                    merged_selection.append(email)
            for email in llm_matched:
                if email["id"] not in all_selected_ids:
                    all_selected_ids.add(email["id"])
                    merged_selection.append(email)
            merged_selection = merged_selection[:MAX_TOTAL_EMAILS]
            logger.info(f"Total selected: {len(merged_selection)} emails")

            if not merged_selection:
                self._update_job(job_id, {
                    "status": "completed",
                    "profile": {},
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
                return

            self._update_job(job_id, {
                "status": "fetching",
                "progress": {"phase": "fetching", "selected": len(merged_selection)},
            })

            # Phase 5: Fetch content (use first Gmail provider)
            logger.info("Phase 5: Fetching content...")
            fetch_start = datetime.now(timezone.utc)
            email_contents: List[Dict] = []
            for provider in providers:
                provider_type = getattr(provider, "provider", "").lower()
                if provider_type in ("google", "gmail"):
                    contents = await self._fetch_gmail_contents(provider, merged_selection)
                    email_contents.extend(contents)
                    break
            fetch_time = (datetime.now(timezone.utc) - fetch_start).total_seconds()
            logger.info(f"Fetched {len(email_contents)} contents in {fetch_time:.1f}s")

            if not email_contents:
                self._update_job(job_id, {
                    "status": "completed",
                    "profile": {},
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
                return

            self._update_job(job_id, {
                "status": "extracting",
                "progress": {"phase": "extracting", "emails": len(email_contents)},
            })

            # Phase 6: Extract
            logger.info("Phase 6: Extracting profiles...")
            extract_start = datetime.now(timezone.utc)
            profiles = await self._extract_profiles_batch(llm_client, email_contents, job_id, user_email)
            extract_time = (datetime.now(timezone.utc) - extract_start).total_seconds()
            logger.info(f"Extracted {len(profiles)} profiles in {extract_time:.1f}s")

            # Phase 7: Merge
            logger.info("Phase 7: Merging...")
            final_profile = self._merge_all(profiles)

            total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self._update_job(job_id, {
                "status": "completed",
                "profile": final_profile,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "progress": {
                    "phase": "completed",
                    "emails_scanned": len(all_metadata),
                    "rule_matched": len(rule_matched),
                    "llm_matched": len(llm_matched),
                    "total_selected": len(merged_selection),
                    "emails_processed": len(email_contents),
                    "total_time_seconds": total_time,
                    "scan_time": scan_time,
                    "filter_time": filter_time,
                    "fetch_time": fetch_time,
                    "extract_time": extract_time,
                },
            })
            logger.info(f"Extraction completed in {total_time:.1f}s")

        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            self._update_job(job_id, {
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
