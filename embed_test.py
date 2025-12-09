from abc import ABC, abstractmethod
from typing import List

from sentence_transformers import SentenceTransformer


# 1) 공통 인터페이스 (추상 클래스)
class BaseEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """여러 문장을 벡터로 바꿔서 반환"""
        raise NotImplementedError()


# 2) BGE 구현체
class BGEEmbedder(BaseEmbedder):
    def __init__(self):
        # 한국어 포함 멀티랭 RAG에 좋은 BGE-m3 모델
        self.model = SentenceTransformer("BAAI/bge-m3")

    def encode(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,  # 코사인 유사도용 정규화
        )
        return vectors.tolist()


# 3) 실제로 돌려보는 테스트 코드
if __name__ == "__main__":
    embedder = BGEEmbedder()

    texts = ["안녕", "지원이 요즘 트랜스포머 공부 중이야"]
    vectors = embedder.encode(texts)

    print("문장 개수:", len(texts))
    print("첫 번째 벡터 길이:", len(vectors[0]))
    print("첫 번째 벡터 앞 5개:", vectors[0][:5])
