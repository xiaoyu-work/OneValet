"""Create Momex (TypeAgent) tables for long-term memory.

Enables pgvector extension and creates all tables required by TypeAgent's
structured RAG system under the 'tenant_default' schema.

Revision ID: 006
Revises: 005
"""

from typing import Sequence, Union

from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Embedding dimension for text-embedding-3-small
EMBEDDING_SIZE = 1536

SCHEMA = "tenant_default"


def upgrade() -> None:
    # 1. pgvector extension (database-level, idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Tenant schema
    op.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}";')

    # 3. Set search_path so tables are created inside the tenant schema.
    #    Include 'extensions' for Supabase where pgvector lives there.
    op.execute(f'SET search_path TO "{SCHEMA}", public, extensions;')

    # 4. Tables (matching TypeAgent's init_db_schema exactly)
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
            extra JSONB NULL,
            CONSTRAINT chunks_xor_chunkuri CHECK (
                (chunks IS NOT NULL AND chunk_uri IS NULL) OR
                (chunks IS NULL AND chunk_uri IS NOT NULL)
            )
        );
    """)

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

    op.execute("""
        CREATE TABLE IF NOT EXISTS RelatedTermsAliases (
            term TEXT NOT NULL,
            alias TEXT NOT NULL,
            PRIMARY KEY (term, alias)
        );
    """)

    op.execute(f"""
        CREATE TABLE IF NOT EXISTS RelatedTermsFuzzy (
            term TEXT NOT NULL PRIMARY KEY,
            term_embedding vector({EMBEDDING_SIZE}) NOT NULL
        );
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS IngestedSources (
            source_id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'ingested'
        );
    """)

    # 5. Indexes
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_messages_start_timestamp "
        "ON Messages(start_timestamp);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_semantic_ref_index_term "
        "ON SemanticRefIndex(term);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_property_index_prop_name "
        "ON PropertyIndex(prop_name);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_property_index_value_str "
        "ON PropertyIndex(value_str);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_property_index_combined "
        "ON PropertyIndex(prop_name, value_str);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_related_aliases_term "
        "ON RelatedTermsAliases(term);"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_related_aliases_alias "
        "ON RelatedTermsAliases(alias);"
    )

    # Reset search_path
    op.execute("SET search_path TO public;")


def downgrade() -> None:
    op.execute(f'DROP SCHEMA IF EXISTS "{SCHEMA}" CASCADE;')
