from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, texts: List[str], vectors: List[List[float]]):
        """텍스트 + 임베딩 저장"""
        raise NotImplementedError()

    @abstractmethod
    def search(self, query_vector: List[float], k: int = 3) -> List[Tuple[str, float]]:
        """query와 가장 유사한 텍스트 k개 찾기"""
        raise NotImplementedError()


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self):
        self.texts = []
        self.vectors = None  # np.ndarray 형태로 저장됨

    def add(self, texts, vectors):
        vecs = np.array(vectors)

        if self.vectors is None:
            self.vectors = vecs
        else:
            self.vectors = np.vstack([self.vectors, vecs])

        self.texts.extend(texts)

    def search(self, query_vector, k=3):
        if self.vectors is None:
            return []

        q = np.array(query_vector)

        dot = self.vectors @ q
        norm_docs = np.linalg.norm(self.vectors, axis=1)
        norm_q = np.linalg.norm(q) + 1e-8
        sims = dot / (norm_docs * norm_q)

        idx = np.argsort(-sims)[:k]
        return [(self.texts[i], float(sims[i])) for i in idx]
