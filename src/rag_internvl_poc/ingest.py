from __future__ import annotations

import logging
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
import psycopg
from tqdm import tqdm
import pytesseract
from PIL import Image

from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


def _chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def _render_page_image(doc: fitz.Document, page_num: int, out_dir: Path, dpi: int = 150) -> str:
    page = doc.load_page(page_num)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"page_{page_num+1:04d}.png"
    pix.save(out_path)
    return str(out_path)


@dataclass
class IngestConfig:
    pdf_dir: Path
    images_dir: Path
    max_chars: int = 1500
    overlap: int = 200
    image_dpi: int = 150
    embed_model_id: str = "BAAI/bge-m3"
    images_input_dir: Optional[Path] = None


def ingest_pdfs(dsn: str, cfg: IngestConfig) -> None:
    device = _pick_device()
    logger.info("[ingest] start: pdf_dir=%s images_dir=%s device=%s model=%s", cfg.pdf_dir, cfg.images_dir, device, cfg.embed_model_id)
    embed = EmbeddingModel(model_id=cfg.embed_model_id, device=device)

    pdf_paths = sorted([p for p in cfg.pdf_dir.glob("**/*.pdf") if p.is_file()])
    if not pdf_paths:
        logger.warning("[ingest] aucun PDF trouvé dans %s", cfg.pdf_dir)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for pdf_path in pdf_paths:
                t_doc = time.perf_counter()
                doc = fitz.open(pdf_path)
                doc_id = str(pdf_path.resolve())
                logger.info("[ingest] %s pages=%d", pdf_path, doc.page_count)
                rows: List[Tuple[str, int, int, str, str, str]] = []
                # On stocke d'abord les contenus, puis on calculera l'embedding en batch
                texts: List[str] = []

                for page_num in tqdm(range(doc.page_count), desc=f"Pages {pdf_path.name}"):
                    page = doc.load_page(page_num)
                    text = page.get_text("text")
                    image_path = _render_page_image(doc, page_num, cfg.images_dir / pdf_path.stem, cfg.image_dpi)

                    chunks = _chunk_text(text, cfg.max_chars, cfg.overlap)
                    for ci, chunk in enumerate(chunks):
                        rows.append((doc_id, page_num, ci, chunk, image_path, None))
                        texts.append(chunk)

                if not rows:
                    continue

                # Embeddings en batch
                t_emb = time.perf_counter()
                embs = embed.encode(texts)  # (N, D) float32 normalisés
                logger.debug("[ingest] embeddings: N=%d dim=%d in %.1f ms", len(texts), embs.shape[1], (time.perf_counter() - t_emb) * 1000)

                # Insertion en DB
                insert_sql = (
                    "INSERT INTO chunks (doc_id, page_num, chunk_index, content, image_path, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)"
                )

                # pgvector accepte les littéraux du style '[0.1, 0.2, ...]'
                def to_vector_literal(v: np.ndarray) -> str:
                    return "[" + ",".join(str(float(x)) for x in v.tolist()) + "]"

                batch: List[Tuple[str, int, int, str, str, str]] = []
                for (doc_id, page_num, ci, chunk, image_path, _), vec in zip(rows, embs):
                    batch.append((doc_id, page_num, ci, chunk, image_path, to_vector_literal(vec)))

                # psycopg3: use executemany (execute_batch is psycopg2-specific)
                t_db = time.perf_counter()
                cur.executemany(insert_sql, batch)
                conn.commit()
                logger.info(
                    "[ingest] inserted rows=%d for %s in %.1f ms (total %.1f ms)",
                    len(batch),
                    pdf_path.name,
                    (time.perf_counter() - t_db) * 1000,
                    (time.perf_counter() - t_doc) * 1000,
                )

            # Ingestion d'images standalone via OCR si dossier fourni
            if cfg.images_input_dir and cfg.images_input_dir.exists():
                image_files = sorted([p for p in cfg.images_input_dir.glob("**/*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
                for img_path in image_files:
                    t_img = time.perf_counter()
                    try:
                        pil_img = Image.open(img_path).convert("RGB")
                    except Exception:
                        continue
                    text = pytesseract.image_to_string(pil_img, lang="fra+eng")
                    chunks = _chunk_text(text, cfg.max_chars, cfg.overlap) or [""]
                    rows: List[Tuple[str, int, int, str, str, str]] = []
                    texts: List[str] = []
                    doc_id = str(img_path.resolve())
                    page_num = 0
                    for ci, chunk in enumerate(chunks):
                        rows.append((doc_id, page_num, ci, chunk, str(img_path), None))
                        texts.append(chunk)
                    if not rows:
                        continue
                    t_emb = time.perf_counter()
                    embs = embed.encode(texts)
                    logger.debug("[ingest-img] embeddings: N=%d dim=%d in %.1f ms", len(texts), embs.shape[1], (time.perf_counter() - t_emb) * 1000)
                    insert_sql = (
                        "INSERT INTO chunks (doc_id, page_num, chunk_index, content, image_path, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s)"
                    )
                    def to_vector_literal(v: np.ndarray) -> str:
                        return "[" + ",".join(str(float(x)) for x in v.tolist()) + "]"
                    batch: List[Tuple[str, int, int, str, str, str]] = []
                    for (_doc, _pg, ci, chunk, image_path, _), vec in zip(rows, embs):
                        batch.append((doc_id, page_num, ci, chunk, image_path, to_vector_literal(vec)))
                    t_db = time.perf_counter()
                    cur.executemany(insert_sql, batch)
                    conn.commit()
                    logger.info(
                        "[ingest-img] inserted rows=%d for %s in %.1f ms",
                        len(batch),
                        img_path.name,
                        (time.perf_counter() - t_img) * 1000,
                    )

    # Analyse pour l'index ivfflat
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("ANALYZE chunks;")
        conn.commit()
    logger.info("[ingest] done")


def _pick_device() -> str:
    # Device simple: si CUDA, 'cuda', sinon si MPS (mac), 'mps', sinon 'cpu'
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

