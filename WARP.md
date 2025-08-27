# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Repository: rag-internvl-poc (Python 3.11, uv)

Overview
- Purpose: Proof-of-concept RAG pipeline over PDFs containing text and images. Ingests PDFs, chunks text, renders page images, stores data in Postgres with pgvector and FTS, retrieves hybrid contexts, and optionally calls InternVL3.5-8B to generate an answer.
- Tooling: Python 3.11, uv for env and dependency management; Postgres ≥15 with pgvector; optional GPU for InternVL3.5-8B.
- Entry point: Console script rag-internvl (Typer CLI) defined in pyproject.

Environment setup
- Create env (Python 3.11) and install deps using uv:
  - uv venv -p 3.11
  - uv sync
- Database DSN is read from the DATABASE_URL environment variable, or pass --dsn explicitly to CLI commands.

Common commands
- CLI help
  - uv run rag-internvl --help

- Initialize database schema (creates extension/table/indexes if missing)
  - uv run rag-internvl init-db --dsn "$DATABASE_URL"

- Ingest PDFs (extract text + render images, compute embeddings, index to DB)
  - uv run rag-internvl ingest --dsn "$DATABASE_URL" --pdf-dir data/pdfs
  - Notes: Default embedding model is BAAI/bge-m3 (dim=1024). Images are written under data/images/<pdf_stem>.

- Query (hybrid retrieval only)
  - uv run rag-internvl query --dsn "$DATABASE_URL" --question "Votre question" --top-k 4 --alpha 0.5

- Query with InternVL3.5-8B generation (requires suitable GPU)
  - uv run rag-internvl query --dsn "$DATABASE_URL" --question "Votre question" --top-k 4 --alpha 0.5 --run-model

- Lint and format (dev tools declared in pyproject [tool.uv].dev-dependencies)
  - Lint: uv run ruff check .
  - Format: uv run ruff format
  - Type-check: uv run mypy src/

- Build package (optional)
  - uv build

Tests
- There is no tests/ directory or test configuration in this repository at present.

Architecture (high-level)
- CLI (src/rag_internvl_poc/cli.py)
  - Typer app exposing commands: init-db, ingest, query.
  - Console script rag-internvl maps to rag_internvl_poc.cli:app.

- Database layer (src/rag_internvl_poc/db.py)
  - Initializes Postgres schema using pgvector and FTS:
    - Table chunks(id, doc_id, page_num, chunk_index, content, content_tsv GENERATED via to_tsvector('french', content), image_path, embedding vector(1024)).
    - Indexes: GIN on content_tsv; ivfflat on embedding (vector_cosine_ops) with lists=100; and a doc/page index.
  - Provides get_dsn() to resolve DSN from env or argument.

- Ingestion (src/rag_internvl_poc/ingest.py)
  - PyMuPDF renders each page to PNG and extracts page text.
  - Text is chunked with sliding window (defaults: max_chars=1500, overlap=200).
  - Embeddings are computed in batch via SentenceTransformers (BAAI/bge-m3) and stored as pgvector literals.
  - Device selection heuristic prefers CUDA, then Apple MPS, else CPU.

- Embedding model wrapper (src/rag_internvl_poc/embeddings.py)
  - Thin adapter over SentenceTransformer with normalization and dim metadata (bge-m3 → 1024).

- Retrieval (src/rag_internvl_poc/retrieve.py)
  - Hybrid SQL over chunks:
    - dense_score = 1 - (embedding <=> query_vec)
    - fts_score = ts_rank_cd(content_tsv, plainto_tsquery('french', :query))
    - hybrid_score = alpha * dense_score + (1 - alpha) * fts_score
  - Returns top_k contexts with scores and image paths.

- Generation (optional) (src/rag_internvl_poc/model_internvl.py)
  - Builds a French system prompt and concatenates retrieved contexts.
  - If --run-model: loadsOpenGVLab/InternVL3_5-2B via Transformers with trust_remote_code, attempts model.chat(...). Falls back to a generic generate(...) path if chat is unavailable.
  - Supplies retrieved page images to the model when available.
  - If not running the model, returns a dry-run structure with the prompt and image list.

Conventions and notes
- Python version: pyproject requires >=3.11,<3.12. Use uv venv -p 3.11.
- Changing embedding model: The DB uses vector(1024). If you switch to a model with a different dimension, update the schema accordingly.
- GPU: InternVL3.5-8B is designed for GPU inference. Without a suitable GPU, prefer running queries without --run-model to validate the retrieval chain.
- Source layout: src/ packaging with primary module rag_internvl_poc/ re-exported from __init__.py.

External rules and docs
- README.md contains end-to-end setup and example commands; the key workflow is reflected above.
- No CLAUDE.md, Cursor rules, or Copilot instruction files were found in this repository.

