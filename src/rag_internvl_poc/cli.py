from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from .db import init_db, get_dsn
from .ingest import IngestConfig, ingest_pdfs
from .rag import answer_question

app = typer.Typer(add_completion=False, help="RAG POC avec InternVL3.5-8B, Postgres pgvector + FTS")


@app.callback()
def _configure(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Affiche des logs détaillés (DEBUG) sur la pipeline RAG",
    )
):
    """Configure le logging global avant chaque commande."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@app.command()
def init_db_cmd(
    dsn: Optional[str] = typer.Option(None, help="DSN Postgres, sinon lu depuis DATABASE_URL"),
):
    """Initialise le schéma (extensions, tables, index)."""
    init_db(get_dsn(dsn))
    typer.echo("Schéma initialisé.")


@app.command()
def ingest(
    pdf_dir: Path = typer.Option(Path("data/pdfs"), exists=True, file_okay=False, help="Dossier contenant les PDF"),
    images_dir: Path = typer.Option(Path("data/images"), help="Dossier de sortie pour les images de pages"),
    max_chars: int = typer.Option(1500, help="Taille max (caractères) d'un chunk"),
    overlap: int = typer.Option(200, help="Chevauchement entre chunks (caractères)"),
    image_dpi: int = typer.Option(150, help="Résolution de rendu des pages en PNG"),
    embed_model_id: str = typer.Option("BAAI/bge-m3", help="Modèle d'embedding (dense)"),
    dsn: Optional[str] = typer.Option(None, help="DSN Postgres, sinon lu depuis DATABASE_URL"),
):
    """Ingestion des PDF : extraction texte+images et indexation (pgvector + FTS)."""
    cfg = IngestConfig(
        pdf_dir=pdf_dir,
        images_dir=images_dir,
        max_chars=max_chars,
        overlap=overlap,
        image_dpi=image_dpi,
        embed_model_id=embed_model_id,
    )
    ingest_pdfs(get_dsn(dsn), cfg)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question en français"),
    top_k: int = typer.Option(4, help="Nombre de contextes à récupérer"),
    alpha: float = typer.Option(0.5, min=0.0, max=1.0, help="Poids dense vs FTS (0..1)"),
    run_model: bool = typer.Option(False, help="Tenter l'inférence InternVL (GPU requis)"),
    dsn: Optional[str] = typer.Option(None, help="DSN Postgres, sinon lu depuis DATABASE_URL"),
):
    """Recherche hybride et, optionnellement, génération via InternVL."""
    out = answer_question(get_dsn(dsn), question, top_k=top_k, alpha=alpha, run_model=run_model)
    typer.echo(json.dumps(out, ensure_ascii=False, indent=2))

