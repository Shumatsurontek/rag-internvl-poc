from __future__ import annotations

from typing import List

from .retrieve import RetrievalConfig, hybrid_search
from .model_internvl import InternVL, build_prompt


def answer_question(
    dsn: str,
    question: str,
    top_k: int = 4,
    alpha: float = 0.5,
    run_model: bool = False,
) -> dict:
    results = hybrid_search(dsn, question, RetrievalConfig(top_k=top_k, alpha=alpha))
    # Optionnel: dédupliquer par page/doc si nécessaire

    internvl = InternVL()
    out = internvl.generate(question=question, contexts=results, run_model=run_model)
    return {
        "question": question,
        "contexts": results,
        **out,
    }

