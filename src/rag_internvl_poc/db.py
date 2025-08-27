import os
from typing import Optional

import psycopg


def get_dsn(explicit_dsn: Optional[str] = None) -> str:
    if explicit_dsn:
        return explicit_dsn
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("Aucun DSN fourni. Passez --dsn ou définissez DATABASE_URL dans l'environnement.")
    return dsn


def init_db(dsn: str) -> None:
    """Initialise le schéma DB : extension pgvector, table chunks, index.
    Utilise vector(1024) par défaut (embeddings BAAI/bge-m3).
    """
    sql = r"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS chunks (
      id           BIGSERIAL PRIMARY KEY,
      doc_id       TEXT NOT NULL,
      page_num     INTEGER NOT NULL,
      chunk_index  INTEGER NOT NULL,
      content      TEXT NOT NULL,
      content_tsv  tsvector GENERATED ALWAYS AS (to_tsvector('french', content)) STORED,
      image_path   TEXT,
      embedding    vector(1024)
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id, page_num, chunk_index);
    CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (content_tsv);
    -- ivfflat nécessite un ANALYZE et donne de bons résultats avec des listes adaptées à votre dataset
    DO $$ BEGIN
      CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    EXCEPTION WHEN duplicate_table THEN
      NULL;
    END $$;
    """
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()

    # Conseillé après une grosse ingestion
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("ANALYZE chunks;")
        conn.commit()

