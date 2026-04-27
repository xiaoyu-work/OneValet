"""Fix pgvector extension location and repair MOMEX tenant schemas.

pgvector was installed without specifying a schema, so it ended up in
tenant_default instead of extensions.  This caused "type vector does not
exist" for every per-user tenant schema, breaking MOMEX memory
storage/retrieval.

Fix:
  1. Move pgvector to the 'extensions' schema (Supabase standard).
  2. Re-create any missing MOMEX tables in all existing tenant_* schemas.

Revision ID: 008
Revises: 007
"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_SIZE = 1536


def upgrade() -> None:
    # ── Step 1: Move pgvector to extensions schema ──────────────────────
    # ALTER EXTENSION SET SCHEMA is non-destructive (no CASCADE data loss).
    op.execute(
        text("""
        DO $$
        DECLARE
            current_schema TEXT;
        BEGIN
            SELECT n.nspname INTO current_schema
            FROM pg_extension e
            JOIN pg_namespace n ON e.extnamespace = n.oid
            WHERE e.extname = 'vector';

            IF current_schema IS NULL THEN
                CREATE EXTENSION IF NOT EXISTS vector SCHEMA extensions;
                RAISE NOTICE 'Created pgvector in extensions schema';
            ELSIF current_schema != 'extensions' THEN
                ALTER EXTENSION vector SET SCHEMA extensions;
                RAISE NOTICE 'Moved pgvector from % to extensions', current_schema;
            ELSE
                RAISE NOTICE 'pgvector already in extensions — no-op';
            END IF;
        END $$;
    """)
    )

    # ── Step 2: Repair tenant_default (re-create anything CASCADE may
    #            have dropped when we tested earlier) ────────────────────
    _create_momex_tables("tenant_default")

    # ── Step 3: Repair every per-user tenant schema ─────────────────────
    op.execute(
        text("""
        DO $$
        DECLARE
            s TEXT;
        BEGIN
            FOR s IN
                SELECT nspname FROM pg_namespace
                WHERE nspname LIKE 'tenant_%' AND nspname != 'tenant_default'
            LOOP
                RAISE NOTICE 'Will repair MOMEX tables in schema: %', s;
            END LOOP;
        END $$;
    """)
    )
    # The DO block above is just for logging; actual table creation is
    # done per-schema via _repair_tenant_schemas() helper below.
    _repair_tenant_schemas()

    # Reset search_path
    op.execute("RESET search_path;")


def downgrade() -> None:
    # No destructive downgrade — tables are IF NOT EXISTS anyway.
    pass


# ── Helpers ─────────────────────────────────────────────────────────────


def _create_momex_tables(schema: str) -> None:
    """Create all 9 MOMEX tables + indexes in the given schema."""
    op.execute(f'SET search_path TO "{schema}", public, extensions;')

    # Core tables (no vector dependency)
    op.execute("""
        CREATE TABLE IF NOT EXISTS ConversationMetadata (
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (key, value)
        );
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS Messages (
            msg_id SERIAL PRIMARY KEY,
            chunks JSONB NULL,
            chunk_uri TEXT NULL,
            start_timestamp TIMESTAMPTZ NULL,
            tags JSONB NULL,
            metadata JSONB NULL,
            extra JSONB NULL
        );
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_start_timestamp ON Messages(start_timestamp);"
    )
    op.execute("""
        CREATE TABLE IF NOT EXISTS SemanticRefs (
            semref_id INTEGER PRIMARY KEY,
            range_json JSONB NOT NULL,
            knowledge_type TEXT NOT NULL,
            knowledge_json JSONB NOT NULL
        );
    """)
    op.execute("""
        CREATE TABLE IF NOT EXISTS SemanticRefIndex (
            term TEXT NOT NULL,
            semref_id INTEGER NOT NULL
                REFERENCES SemanticRefs(semref_id) ON DELETE CASCADE
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_semantic_ref_index_term ON SemanticRefIndex(term);")

    # Tables with vector columns (these previously failed)
    op.execute(f"""
        CREATE TABLE IF NOT EXISTS MessageTextIndex (
            id SERIAL PRIMARY KEY,
            msg_id INTEGER NOT NULL
                REFERENCES Messages(msg_id) ON DELETE CASCADE,
            chunk_ordinal INTEGER NOT NULL,
            embedding vector({EMBEDDING_SIZE}) NOT NULL,
            UNIQUE (msg_id, chunk_ordinal)
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS PropertyIndex (
            prop_name TEXT NOT NULL,
            value_str TEXT NOT NULL,
            score REAL NOT NULL DEFAULT 1.0,
            semref_id INTEGER NOT NULL
                REFERENCES SemanticRefs(semref_id) ON DELETE CASCADE
        );
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_property_index_prop_name ON PropertyIndex(prop_name);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_property_index_value_str ON PropertyIndex(value_str);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_property_index_combined "
        "ON PropertyIndex(prop_name, value_str);"
    )

    op.execute("""
        CREATE TABLE IF NOT EXISTS RelatedTermsAliases (
            term TEXT NOT NULL,
            alias TEXT NOT NULL,
            PRIMARY KEY (term, alias)
        );
    """)
    op.execute("CREATE INDEX IF NOT EXISTS idx_related_aliases_term ON RelatedTermsAliases(term);")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_related_aliases_alias ON RelatedTermsAliases(alias);"
    )

    op.execute(f"""
        CREATE TABLE IF NOT EXISTS RelatedTermsFuzzy (
            term TEXT NOT NULL PRIMARY KEY,
            term_embedding vector({EMBEDDING_SIZE}) NOT NULL
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS IngestedSources (
            source_id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'Ingested'
        );
    """)


def _repair_tenant_schemas() -> None:
    """Dynamically find and repair all per-user tenant schemas."""
    # We use a DO block to iterate schemas, but alembic op.execute()
    # can't easily loop + call Python.  Instead we generate all the
    # DDL in a single PL/pgSQL block.
    op.execute(
        text(
            """
        DO $$
        DECLARE
            s TEXT;
        BEGIN
            FOR s IN
                SELECT nspname FROM pg_namespace
                WHERE nspname LIKE 'tenant_%' AND nspname != 'tenant_default'
            LOOP
                EXECUTE format('SET search_path TO %I, public, extensions', s);

                -- Core tables
                EXECUTE 'CREATE TABLE IF NOT EXISTS ConversationMetadata (
                    key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY (key, value))';
                EXECUTE 'CREATE TABLE IF NOT EXISTS Messages (
                    msg_id SERIAL PRIMARY KEY, chunks JSONB NULL, chunk_uri TEXT NULL,
                    start_timestamp TIMESTAMPTZ NULL, tags JSONB NULL,
                    metadata JSONB NULL, extra JSONB NULL)';
                EXECUTE 'CREATE INDEX IF NOT EXISTS idx_messages_start_timestamp ON Messages(start_timestamp)';
                EXECUTE 'CREATE TABLE IF NOT EXISTS SemanticRefs (
                    semref_id INTEGER PRIMARY KEY, range_json JSONB NOT NULL,
                    knowledge_type TEXT NOT NULL, knowledge_json JSONB NOT NULL)';
                EXECUTE 'CREATE TABLE IF NOT EXISTS SemanticRefIndex (
                    term TEXT NOT NULL,
                    semref_id INTEGER NOT NULL REFERENCES SemanticRefs(semref_id) ON DELETE CASCADE)';
                EXECUTE 'CREATE INDEX IF NOT EXISTS idx_semantic_ref_index_term ON SemanticRefIndex(term)';

                -- Vector tables (previously failed without pgvector)
                EXECUTE 'CREATE TABLE IF NOT EXISTS MessageTextIndex (
                    id SERIAL PRIMARY KEY,
                    msg_id INTEGER NOT NULL REFERENCES Messages(msg_id) ON DELETE CASCADE,
                    chunk_ordinal INTEGER NOT NULL,
                    embedding vector("""
            + str(EMBEDDING_SIZE)
            + """) NOT NULL,
                    UNIQUE (msg_id, chunk_ordinal))';

                EXECUTE 'CREATE TABLE IF NOT EXISTS PropertyIndex (
                    prop_name TEXT NOT NULL, value_str TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 1.0,
                    semref_id INTEGER NOT NULL REFERENCES SemanticRefs(semref_id) ON DELETE CASCADE)';
                EXECUTE 'CREATE INDEX IF NOT EXISTS idx_property_index_prop_name ON PropertyIndex(prop_name)';
                EXECUTE 'CREATE INDEX IF NOT EXISTS idx_property_index_value_str ON PropertyIndex(value_str)';
                EXECUTE 'CREATE INDEX IF NOT EXISTS idx_property_index_combined ON PropertyIndex(prop_name, value_str)';

                EXECUTE 'CREATE TABLE IF NOT EXISTS RelatedTermsAliases (
                    term TEXT NOT NULL, alias TEXT NOT NULL, PRIMARY KEY (term, alias))';
                EXECUTE 'CREATE INDEX IF NOT EXISTS idx_related_aliases_term ON RelatedTermsAliases(term)';
                EXECUTE 'CREATE INDEX IF NOT EXISTS idx_related_aliases_alias ON RelatedTermsAliases(alias)';

                EXECUTE 'CREATE TABLE IF NOT EXISTS RelatedTermsFuzzy (
                    term TEXT NOT NULL PRIMARY KEY,
                    term_embedding vector("""
            + str(EMBEDDING_SIZE)
            + """) NOT NULL)';

                EXECUTE 'CREATE TABLE IF NOT EXISTS IngestedSources (
                    source_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT ''Ingested'')';

                RAISE NOTICE 'Repaired MOMEX tables in schema: %', s;
            END LOOP;
        END $$;
    """
        )
    )
