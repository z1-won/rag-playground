from typing import List
from sentence_transformers import SentenceTransformer
from .base_embedder import BaseEmbedder

class BGEEmbedder(BaseEmbedder):
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")

    def encode(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
