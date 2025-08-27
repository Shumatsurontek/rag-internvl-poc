from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingModel:
    model_id: str = "BAAI/bge-m3"
    device: str = "cpu"  # "cuda", "mps" ou "cpu"

    def __post_init__(self) -> None:
        self.model = SentenceTransformer(self.model_id, device=self.device)

    @property
    def dim(self) -> int:
        # bge-m3 -> 1024
        # On interroge le modèle si possible, sinon on renvoie 1024 par défaut
        try:
            return int(self.model.get_sentence_embedding_dimension())
        except Exception:
            return 1024

    def encode(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        return embs.astype(np.float32)

