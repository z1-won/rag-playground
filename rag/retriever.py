from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

from .base_embedder import BaseEmbedder
from .vector_store import BaseVectorStore


class BaseRetriever(ABC):
    """모든 리트리버가 따라야 하는 공통 인터페이스"""

    @abstractmethod
    def add_documents(self, texts: List[str]) -> None:
        """문서(텍스트)를 내부 스토어에 추가"""
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """쿼리로부터 상위 k개의 (텍스트, 점수)를 반환"""
        raise NotImplementedError


class SimilarityRetriever(BaseRetriever):
    """
    Dense 임베딩 + 벡터 스토어 기반
    → 코사인 유사도로 문서 검색하는 가장 기본 리트리버
    """

    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def add_documents(self, texts: List[str]) -> None:
        vectors = self.embedder.encode(texts)
        self.store.add(texts, vectors)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        query_vec = self.embedder.encode([query])[0]
        return self.store.search(query_vec, k=k)
