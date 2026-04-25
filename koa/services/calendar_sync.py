"""CalendarSyncService — daily background task that pulls events from all
bound calendar providers (Google Calendar today; Outlook etc. in future) into
the shared ``tenant_default.local_calendar_events`` table so the local
CalendarAgent can answer schedule questions without round-tripping to the
upstream provider.

Design
------
* Runs every 24h (initial delay 60s so the app finishes startup).
* For every tenant that has at least one calendar credential, iterates over
  every account, instantiates the appropriate provider via
  ``CalendarProviderFactory``, fetches events in the window
  ``[now - LOOKBACK_DAYS, now + LOOKAHEAD_DAYS]``, then upserts each event
  into ``tenant_default.local_calendar_events``.
* ``event_id`` is namespaced per source ("google:<calendar_id>:<event_id>")
  so iOS EventKit ingestion (which uses raw EventKit UUIDs) never collides
  with Google rows.
* Deduplication: Google version wins.  Before upserting a Google event we
  delete any other row for the same user with a matching ``metadata->>
  'ical_uid'`` (which is stable across iOS subscribed-from-Google copies).
* Token persistence: each provider is created with a callback that writes the
  refreshed credentials back to ``CredentialStore``.  This fixes the
  pre-existing bug where refreshed access_tokens were never persisted.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from ..constants import CALENDAR_SERVICES
from ..providers.calendar.factory import CalendarProviderFactory

logger = logging.getLogger(__name__)

# Default daily cadence
SYNC_INTERVAL_S = 24 * 60 * 60

# Time window we keep mirrored locally
LOOKBACK_DAYS = 7
LOOKAHEAD_DAYS = 60

# Cap per-account events per run (Google paginates; we just need agenda volume)
MAX_RESULTS_PER_ACCOUNT = 250

# Map credential.service -> provider name expected by CalendarProviderFactory
_SERVICE_TO_PROVIDER = {
    "google_calendar": "google",
    "outlook_calendar": "microsoft",
}


class CalendarSyncService:
    """Polls every bound calendar account and mirrors events into
    ``tenant_default.local_calendar_events``.
    """

    def __init__(self, db, credential_store, interval_s: int = SYNC_INTERVAL_S):
        """
        Args:
            db: koa.db.Database instance (asyncpg pool).
            credential_store: koa.credentials.CredentialStore instance.
            interval_s: Seconds between full sync passes (default 24h).
        """
        self._db = db
        self._credential_store = credential_store
        self._interval_s = interval_s
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "CalendarSyncService started (interval=%ds, window=-%dd..+%dd)",
            self._interval_s,
            LOOKBACK_DAYS,
            LOOKAHEAD_DAYS,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("CalendarSyncService stopped")

    async def _loop(self) -> None:
        # Brief initial delay so app finishes startup before we hammer external APIs
        await asyncio.sleep(60)
        while self._running:
            try:
                await self.sync_all_tenants()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("CalendarSyncService loop error: %s", e, exc_info=True)
            await asyncio.sleep(self._interval_s)

    # ------------------------------------------------------------------
    # Public sync APIs
    # ------------------------------------------------------------------

    async def sync_all_tenants(self) -> Dict[str, Any]:
        """Iterate every tenant that has at least one calendar credential and
        sync each of their calendar accounts. Returns a summary dict.
        """
        tenant_ids = await self._list_tenants_with_calendars()
        logger.info("CalendarSync: %d tenant(s) with calendar accounts", len(tenant_ids))

        total_events = 0
        total_accounts = 0
        for tenant_id in tenant_ids:
            try:
                result = await self.sync_tenant(tenant_id)
                total_events += result.get("events", 0)
                total_accounts += result.get("accounts", 0)
            except Exception as e:
                logger.error("CalendarSync: tenant %s failed: %s", tenant_id, e, exc_info=True)

        return {"tenants": len(tenant_ids), "accounts": total_accounts, "events": total_events}

    async def sync_tenant(self, tenant_id: str) -> Dict[str, Any]:
        """Sync every calendar account for a single tenant. Public so it can
        also be invoked from the manual /api/sensing/calendar/sync endpoint.
        """
        accounts_synced = 0
        events_synced = 0

        for service in CALENDAR_SERVICES:
            try:
                creds_list = await self._credential_store.list(tenant_id, service=service)
            except Exception as e:
                logger.warning(
                    "CalendarSync: failed to list %s for %s: %s", service, tenant_id, e
                )
                continue

            for creds in creds_list or []:
                try:
                    n = await self._sync_account(tenant_id, service, creds)
                    accounts_synced += 1
                    events_synced += n
                except Exception as e:
                    logger.error(
                        "CalendarSync: account %s/%s for %s failed: %s",
                        service,
                        creds.get("account_name"),
                        tenant_id,
                        e,
                        exc_info=True,
                    )

        logger.info(
            "CalendarSync: tenant=%s accounts=%d events=%d",
            tenant_id,
            accounts_synced,
            events_synced,
        )
        return {"tenant_id": tenant_id, "accounts": accounts_synced, "events": events_synced}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _list_tenants_with_calendars(self) -> List[str]:
        """Return distinct tenant_ids that have at least one calendar credential."""
        placeholders = ",".join(f"${i + 1}" for i in range(len(CALENDAR_SERVICES)))
        rows = await self._db.fetch(
            f"SELECT DISTINCT tenant_id FROM credentials WHERE service IN ({placeholders})",
            *CALENDAR_SERVICES,
        )
        return [r["tenant_id"] for r in rows or []]

    async def _sync_account(
        self, tenant_id: str, service: str, creds: dict
    ) -> int:
        """Fetch + upsert events for a single calendar account. Returns count written."""
        provider_name = _SERVICE_TO_PROVIDER.get(service)
        if not provider_name:
            logger.debug("CalendarSync: no provider mapping for %s", service)
            return 0

        # Inject the provider field expected by the factory
        creds_for_factory = dict(creds)
        creds_for_factory.setdefault("provider", provider_name)

        account_name = creds.get("account_name") or "primary"

        # Persist refreshed tokens back to the store. Factory invokes this
        # callback synchronously after _do_refresh succeeds, so we schedule
        # the async DB write as a fire-and-forget task.
        callback = self._make_persist_callback(tenant_id, service, account_name)

        provider = CalendarProviderFactory.create_provider(creds_for_factory, on_token_refreshed=callback)
        if not provider:
            logger.warning(
                "CalendarSync: factory returned no provider for %s/%s", service, account_name
            )
            return 0

        now = datetime.now(timezone.utc)
        time_min = now - timedelta(days=LOOKBACK_DAYS)
        time_max = now + timedelta(days=LOOKAHEAD_DAYS)

        result = await provider.list_events(
            time_min=time_min, time_max=time_max, max_results=MAX_RESULTS_PER_ACCOUNT
        )
        if not result.get("success"):
            logger.warning(
                "CalendarSync: list_events failed for %s/%s: %s",
                service,
                account_name,
                result.get("error"),
            )
            return 0

        events = result.get("data") or []
        written = 0
        for ev in events:
            try:
                if await self._upsert_event(
                    user_id=tenant_id,
                    source=provider_name,
                    account_name=account_name,
                    raw_event=ev,
                    calendar_id=getattr(provider, "calendar_id", None) or "primary",
                ):
                    written += 1
            except Exception as e:
                logger.error(
                    "CalendarSync: upsert failed for event %s: %s",
                    ev.get("event_id"),
                    e,
                )
        return written

    def _make_persist_callback(
        self, tenant_id: str, service: str, account_name: str
    ) -> Callable[[dict], None]:
        store = self._credential_store

        def _cb(updated: dict) -> None:
            # _do_refresh calls this synchronously. Spawn an async task to
            # write back to the credential store without blocking.
            try:
                # Strip the synthetic 'provider' field we injected for the factory
                payload = {k: v for k, v in updated.items() if k != "provider"}
                asyncio.create_task(
                    store.save(tenant_id, service, payload, account_name=account_name)
                )
                logger.info(
                    "CalendarSync: persisted refreshed token for %s/%s/%s",
                    tenant_id,
                    service,
                    account_name,
                )
            except Exception as e:
                logger.error("CalendarSync: persist refreshed token failed: %s", e)

        return _cb

    async def _upsert_event(
        self,
        *,
        user_id: str,
        source: str,
        account_name: str,
        raw_event: dict,
        calendar_id: str,
    ) -> bool:
        """Upsert a single provider event row, applying iCalUID-based dedup.

        Returns True if a row was written, False if skipped (e.g., missing
        timestamps).
        """
        starts_at = raw_event.get("start")
        ends_at = raw_event.get("end")
        if not starts_at or not ends_at:
            return False

        # Provider-specific id fields
        provider_event_id = raw_event.get("event_id") or raw_event.get("id")
        if not provider_event_id:
            return False

        # Namespace the local event_id so different sources never collide
        event_id = f"{source}:{calendar_id}:{provider_event_id}"

        ical_uid = (
            raw_event.get("ical_uid")
            or raw_event.get("iCalUID")
            or raw_event.get("uid")
        )

        title = raw_event.get("summary") or raw_event.get("title") or "(No title)"
        location = raw_event.get("location") or None
        notes = raw_event.get("description") or raw_event.get("notes") or None
        attendees = raw_event.get("attendees") or []
        all_day = bool(raw_event.get("all_day", False))

        metadata = {
            "source": source,
            "account_name": account_name,
            "calendar_id": calendar_id,
        }
        if ical_uid:
            metadata["ical_uid"] = ical_uid
        if raw_event.get("html_link"):
            metadata["html_link"] = raw_event["html_link"]
        if raw_event.get("organizer"):
            metadata["organizer"] = raw_event["organizer"]
        if raw_event.get("status"):
            metadata["status"] = raw_event["status"]

        # Dedup: if Google version, drop any other-source rows with same iCalUID
        # so the Google copy becomes authoritative.
        if source == "google" and ical_uid:
            await self._db.execute(
                """
                DELETE FROM tenant_default.local_calendar_events
                WHERE user_id = $1
                  AND event_id <> $2
                  AND metadata->>'ical_uid' = $3
                """,
                user_id,
                event_id,
                ical_uid,
            )

        await self._db.execute(
            """
            INSERT INTO tenant_default.local_calendar_events
                (user_id, event_id, calendar_name, title, starts_at, ends_at,
                 all_day, location, notes, attendees, metadata, updated_at)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::jsonb,$11::jsonb, NOW())
            ON CONFLICT (user_id, event_id) DO UPDATE SET
                calendar_name = EXCLUDED.calendar_name,
                title = EXCLUDED.title,
                starts_at = EXCLUDED.starts_at,
                ends_at = EXCLUDED.ends_at,
                all_day = EXCLUDED.all_day,
                location = EXCLUDED.location,
                notes = EXCLUDED.notes,
                attendees = EXCLUDED.attendees,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """,
            user_id,
            event_id,
            account_name,
            title,
            starts_at,
            ends_at,
            all_day,
            location,
            notes,
            json.dumps(attendees),
            json.dumps(metadata),
        )
        return True
