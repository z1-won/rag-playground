from typing import List, Tuple
from base_embedder import BaseEmbedder
from vector_store import BaseVectorStore

class SimpleRetriever:
    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore):
        self.embedder = embedder
        self.store = store

    def add_documents(self, texts: List[str]):
        vectors = self.embedder.encode(texts)
        self.store.add(texts, vectors)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        query_vec = self.embedder.encode([query])[0]
        return self.store.search(query_vec, k=k)
