from __future__ import annotations

import logging
import time
from typing import List

from .retrieve import RetrievalConfig, hybrid_search
from .model_internvl import InternVL, build_prompt

logger = logging.getLogger(__name__)

def answer_question(
    dsn: str,
    question: str,
    top_k: int = 4,
    alpha: float = 0.5,
    run_model: bool = False,
) -> dict:
    logger.info("[rag] question='%s' top_k=%d alpha=%.2f run_model=%s", question, top_k, alpha, run_model)
    t0 = time.perf_counter()
    results = hybrid_search(dsn, question, RetrievalConfig(top_k=top_k, alpha=alpha))
    logger.info("[rag] retrieved %d contexts in %.1f ms", len(results), (time.perf_counter() - t0) * 1000)
    # Optionnel: dédupliquer par page/doc si nécessaire

    internvl = InternVL()
    t1 = time.perf_counter()
    out = internvl.generate(question=question, contexts=results, run_model=run_model)
    logger.info("[rag] generation done dry_run=%s in %.1f ms", out.get("dry_run"), (time.perf_counter() - t1) * 1000)
    return {
        "question": question,
        "contexts": results,
        **out,
    }

