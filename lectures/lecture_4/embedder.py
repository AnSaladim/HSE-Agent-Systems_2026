"""
Обёртка над embeddings API.

По умолчанию используется `text-embedding-3-small`.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np


class OpenAIEmbedder:
    def __init__(
        self,
        client: Any,
        model: str = "text-embedding-3-small",
        batch_size: int = 32,
        fallback_dim: int = 256,
    ) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.fallback_dim = fallback_dim

    def _fallback_embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.fallback_dim, dtype=np.float32)
        tokens = re.findall(r"[\w]+", text.lower())

        if not tokens:
            return vec.tolist()

        for token in tokens:
            h = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % self.fallback_dim
            sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def _fallback_embed(self, texts: list[str]) -> list[list[float]]:
        return [self._fallback_embed_one(text) for text in texts]

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            result: list[list[float]] = []
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                result.extend([item.embedding for item in response.data])
            return result
        except Exception:
            return self._fallback_embed(texts)

    def encode_query(self, text: str) -> list[float]:
        return self.encode([text])[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)

    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return 0.0

    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0

    return float(np.dot(va, vb) / (na * nb))
