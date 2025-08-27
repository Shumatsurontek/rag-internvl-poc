from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModel:
    model_id: str = "BAAI/bge-m3"
    device: str = "cpu"  # "cuda", "mps" ou "cpu"

    def __post_init__(self) -> None:
        start = time.perf_counter()
        self.model = SentenceTransformer(self.model_id, device=self.device)
        dur = (time.perf_counter() - start) * 1000
        logger.debug("Loaded embedding model '%s' on device=%s in %.1f ms", self.model_id, self.device, dur)

    @property
    def dim(self) -> int:
        # bge-m3 -> 1024
        # On interroge le modèle si possible, sinon on renvoie 1024 par défaut
        try:
            return int(self.model.get_sentence_embedding_dimension())
        except Exception:
            return 1024

    def encode(self, texts: List[str]) -> np.ndarray:
        start = time.perf_counter()
        embs = self.model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        dur = (time.perf_counter() - start) * 1000
        logger.debug("Encoded %d texts -> shape=%s in %.1f ms", len(texts), tuple(embs.shape), dur)
        return embs.astype(np.float32)

