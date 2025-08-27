from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import psycopg

from .embeddings import EmbeddingModel


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
    embed = EmbeddingModel()
    q_vec = embed.encode([question])[0]
    q_vec_lit = _to_vector_literal(q_vec)

    sql = (
        "SELECT id, doc_id, page_num, chunk_index, content, image_path, "
        "       (1 - (embedding <=> $2::vector)) AS dense_score, "
        "       ts_rank_cd(content_tsv, plainto_tsquery('french', $1)) AS fts_score, "
        "       ($3 * (1 - (embedding <=> $2::vector)) + (1 - $3) * "
        "        ts_rank_cd(content_tsv, plainto_tsquery('french', $1))) AS hybrid_score "
        "FROM chunks "
        "ORDER BY hybrid_score DESC "
        "LIMIT $4"
    )

    results: List[dict] = []
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (question, q_vec_lit, cfg.alpha, cfg.top_k))
            for row in cur.fetchall():
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
    return results

