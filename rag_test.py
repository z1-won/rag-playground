
from bge_embedder import BGEEmbedder
from vector_store import InMemoryVectorStore
from retriever import SimpleRetriever

if __name__ == "__main__":
    embedder = BGEEmbedder()
    store = InMemoryVectorStore()
    retriever = SimpleRetriever(embedder, store)

    docs = [
        "트랜스포머는 어텐션 기반 모델이다.",
        "BERT는 트랜스포머 인코더 구조를 사용한다.",
        "YOLO는 CNN 기반 객체 탐지 모델이다."
    ]

    retriever.add_documents(docs)

    query = "트랜스포머와 BERT의 관계가 뭐야?"
    results = retriever.retrieve(query, k=2)

    print("질문:", query)
    print("검색 결과:")
    for text, score in results:
        print(f"- ({score:.3f}) {text}")
