from bge_embedder import BGEEmbedder
from retriever import SimilarityRetriever
from vector_store import InMemoryVectorStore


if __name__ == "__main__":
    embedder = BGEEmbedder()
    store = InMemoryVectorStore()
    retriever = SimilarityRetriever(embedder, store)

    docs = [
        "정은원은 한화 이글스의 2루수이며 등번호는 43번이다.",
        "류현진은 한화 이글스의 투수이며 등번호는 99번이다.",
        "야구는 9이닝으로 구성된 스포츠이다."
        "야구는 투수 1명과 야수 9명으로, 총 10명이서 진행한다.",
        "야구는 9이닝으로 구성된 스포츠이다.",
        "폰세는 삼진 17개를 잡아냈다.",
        "류현진은 한화 이글스의 영구 결번이 될 선수이다.",
    ]

    retriever.add_documents(docs)

    # 🔥 터미널에서 사용자 입력 받기
    while True:
        query = input("\n질문을 입력하세요 (종료하려면 exit 입력): ")

        if query.lower() in ["exit", "quit"]:
            print("종료합니다.")
            break

        results = retriever.retrieve(query, k=3)

        print(f"\n검색 결과:")
        for text, score in results:
            print(f"- ({score:.3f}) {text}")
