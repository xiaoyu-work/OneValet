#!/usr/bin/env python3
"""
Credential Migration Script - Supabase oauth_accounts -> CredentialStore (Postgres)

One-time, idempotent migration script.

Mapping:
    oauth_accounts.user_id   -> credentials.tenant_id
    oauth_accounts.provider  -> credentials.service
    oauth_accounts.account_name -> credentials.account_name (default: "primary")
    oauth_accounts fields    -> credentials.credentials_json

Supabase oauth_accounts schema (source):
    user_id, provider, account_identifier, account_name,
    access_token, refresh_token, token_expiry, scopes, metadata

CredentialStore schema (target):
    tenant_id, service, account_name, credentials_json

Usage:
    # Set environment variables:
    #   SUPABASE_URL          - Supabase project URL
    #   SUPABASE_SERVICE_KEY  - Supabase service role key
    #   CREDENTIAL_STORE_DSN  - Postgres DSN for CredentialStore
    #
    # Then run:
    python scripts/migrate_credentials.py [--dry-run]
"""

import os
import sys
import json
import asyncio
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_from_supabase(supabase_url: str, supabase_key: str) -> list:
    """Fetch all oauth_accounts from Supabase."""
    try:
        from supabase import create_client
    except ImportError:
        logger.error("supabase-py is required. Install with: pip install supabase")
        sys.exit(1)

    client = create_client(supabase_url, supabase_key)

    # Paginate through all records (Supabase default limit is 1000)
    all_rows = []
    offset = 0
    page_size = 1000
    while True:
        response = (
            client.table("oauth_accounts")
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        rows = response.data or []
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size

    logger.info(f"Fetched {len(all_rows)} rows from Supabase oauth_accounts")
    return all_rows


def transform_row(row: dict) -> dict:
    """Transform a Supabase oauth_accounts row to CredentialStore format.

    Returns:
        {
            "tenant_id": str,
            "service": str,
            "account_name": str,
            "credentials": dict,  # the JSON blob to store
        }
    """
    credentials = {
        "account_identifier": row.get("account_identifier"),
        "access_token": row.get("access_token"),
        "refresh_token": row.get("refresh_token"),
        "token_expiry": row.get("token_expiry"),
        "scopes": row.get("scopes"),
    }

    # Merge any extra metadata
    extra_meta = row.get("metadata")
    if extra_meta and isinstance(extra_meta, dict):
        credentials["metadata"] = extra_meta

    # Remove None values for cleaner storage
    credentials = {k: v for k, v in credentials.items() if v is not None}

    return {
        "tenant_id": row["user_id"],
        "service": row["provider"],
        "account_name": row.get("account_name") or "primary",
        "credentials": credentials,
    }


async def write_to_credential_store(dsn: str, records: list, dry_run: bool = False) -> int:
    """Write transformed records to CredentialStore. Uses upsert for idempotency."""
    # Import inline to avoid hard dependency at module level
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from onevalet.credentials import CredentialStore

    store = CredentialStore(dsn=dsn)
    await store.initialize()

    count = 0
    for rec in records:
        if dry_run:
            logger.info(
                f"[DRY RUN] Would upsert: "
                f"tenant={rec['tenant_id']}, service={rec['service']}, "
                f"account={rec['account_name']}"
            )
        else:
            await store.save(
                tenant_id=rec["tenant_id"],
                service=rec["service"],
                credentials=rec["credentials"],
                account_name=rec["account_name"],
            )
        count += 1

    await store.close()
    return count


async def main():
    parser = argparse.ArgumentParser(description="Migrate Supabase oauth_accounts to CredentialStore")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be migrated without writing")
    args = parser.parse_args()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    cred_dsn = os.getenv("CREDENTIAL_STORE_DSN")

    if not supabase_url or not supabase_key:
        logger.error("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        sys.exit(1)

    if not cred_dsn and not args.dry_run:
        logger.error("CREDENTIAL_STORE_DSN must be set (or use --dry-run)")
        sys.exit(1)

    # Step 1: Fetch from Supabase
    rows = fetch_from_supabase(supabase_url, supabase_key)
    if not rows:
        logger.info("No records to migrate.")
        return

    # Step 2: Transform
    records = [transform_row(row) for row in rows]

    # Step 3: Write to CredentialStore
    count = await write_to_credential_store(cred_dsn or "", records, dry_run=args.dry_run)

    action = "Would migrate" if args.dry_run else "Migrated"
    logger.info(f"{action} {count} credential records")


if __name__ == "__main__":
    asyncio.run(main())
