from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class BaseVectorStore(ABC):
    """벡터를 저장하고, 쿼리 벡터와의 유사도로 검색하는 공통 인터페이스"""

    @abstractmethod
    def add(self, texts: List[str], vectors: List[List[float]]) -> None:
        """텍스트와 그 임베딩 벡터들을 스토어에 추가"""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vec: List[float], k: int = 3) -> List[Tuple[str, float]]:
        """쿼리 벡터와 가장 유사한 상위 k개의 (텍스트, 점수)를 반환"""
        raise NotImplementedError


class InMemoryVectorStore(BaseVectorStore):
    """가장 간단한 인메모리 벡터 스토어 (numpy + 코사인 유사도)"""

    def __init__(self) -> None:
        self.texts: List[str] = []
        self.vectors: np.ndarray | None = None  # shape: (N, D)

    def add(self, texts: List[str], vectors: List[List[float]]) -> None:
        vecs = np.asarray(vectors, dtype=np.float32)

        if self.vectors is None:
            self.vectors = vecs
            self.texts = list(texts)
        else:
            self.vectors = np.vstack([self.vectors, vecs])
            self.texts.extend(texts)

    def search(self, query_vec: List[float], k: int = 3) -> List[Tuple[str, float]]:
        if self.vectors is None or len(self.texts) == 0:
            return []

        q = np.asarray(query_vec, dtype=np.float32)

        # 혹시 임베딩이 미리 정규화 안 되어 있어도 코사인 유사도 되게 방어 코드
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        mat = self.vectors
        row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1.0
        mat_norm = mat / row_norms

        sims = mat_norm @ q  # (N,) 코사인 유사도

        topk_idx = np.argsort(-sims)[:k]  # 내림차순 상위 k개
        results: List[Tuple[str, float]] = [
            (self.texts[i], float(sims[i])) for i in topk_idx
        ]
        return results
