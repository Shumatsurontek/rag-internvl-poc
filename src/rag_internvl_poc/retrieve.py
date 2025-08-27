from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

import psycopg

from .embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    top_k: int = 4
    alpha: float = 0.5  # poids du dense vs FTS (0..1)


def _to_vector_literal(vec) -> str:
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def hybrid_search(dsn: str, question: str, cfg: RetrievalConfig) -> List[dict]:
    """Retourne les meilleurs chunks selon un score hybride dense + FTS.

    Score = alpha * dense_score + (1 - alpha) * fts_score
    dense_score = 1 - cosine_distance
    fts_score = ts_rank_cd(...)
    """
    t0 = time.perf_counter()
    embed = EmbeddingModel()
    q_vec = embed.encode([question])[0]
    q_vec_lit = _to_vector_literal(q_vec)
    logger.debug("[retrieve] query embedding dim=%d in %.1f ms", len(q_vec), (time.perf_counter() - t0) * 1000)

    sql = (
        "SELECT id, doc_id, page_num, chunk_index, content, image_path, "
        "       (1 - (embedding <=> %s::vector)) AS dense_score, "
        "       ts_rank_cd(content_tsv, plainto_tsquery('french', %s)) AS fts_score, "
        "       (%s * (1 - (embedding <=> %s::vector)) + (1 - %s) * "
        "        ts_rank_cd(content_tsv, plainto_tsquery('french', %s))) AS hybrid_score "
        "FROM chunks "
        "ORDER BY hybrid_score DESC "
        "LIMIT %s"
    )

    results: List[dict] = []
    t1 = time.perf_counter()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # psycopg3 uses %s placeholders; order must match the SQL above
            # Order: q_vec_lit (vector), question (fts), alpha, q_vec_lit (vector), alpha, question (fts), top_k
            cur.execute(
                sql,
                (
                    q_vec_lit,
                    question,
                    cfg.alpha,
                    q_vec_lit,
                    cfg.alpha,
                    question,
                    cfg.top_k,
                ),
            )
            rows = cur.fetchall()
            for row in rows:
                (rid, doc_id, page_num, chunk_index, content, image_path, dense, fts, hybrid) = row
                results.append(
                    {
                        "id": rid,
                        "doc_id": doc_id,
                        "page_num": page_num,
                        "chunk_index": chunk_index,
                        "content": content,
                        "image_path": image_path,
                        "dense_score": float(dense if dense is not None else 0.0),
                        "fts_score": float(fts if fts is not None else 0.0),
                        "hybrid_score": float(hybrid if hybrid is not None else 0.0),
                    }
                )
    logger.debug(
        "[retrieve] db search top_k=%d alpha=%.2f -> %d rows in %.1f ms",
        cfg.top_k,
        cfg.alpha,
        len(results),
        (time.perf_counter() - t1) * 1000,
    )
    return results

