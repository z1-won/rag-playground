from rag.bge_embedder import BGEEmbedder
from rag.vector_store import InMemoryVectorStore
from rag.retriever import SimilarityRetriever
from rag.pdf_parser import extract_kbo_chunks, chunk_text


if __name__ == "__main__":
    pdf_path = "data/2025_리그규정.pdf"

    # 1) 조항 단위 chunk 추출
    kbo_chunks = extract_kbo_chunks(pdf_path)
    print(f"추출된 조항 단위 청크 개수: {len(kbo_chunks)}")

    embedder = BGEEmbedder()
    store = InMemoryVectorStore()
    retriever = SimilarityRetriever(embedder, store)

    docs = []

    for c in kbo_chunks:
        prefix_parts = []
        if c.chapter:
            prefix_parts.append(c.chapter)
        if c.article:
            prefix_parts.append(c.article)
        location = " / ".join(prefix_parts) if prefix_parts else "KBO 규정"
        location += f" (p.{c.page_start}-{c.page_end})"

        # 조항 chunk가 너무 길면 다시 잘라서 300~400자 chunk로 변환
        small_chunks = chunk_text(c.text, chunk_size=350, overlap=50)

        for sc in small_chunks:
            full_text = f"[{location}]\n{sc}"
            docs.append(full_text)

    print(f"최종 문장 단위 청크 개수: {len(docs)}")

    retriever.add_documents(docs)

    # 인터렉티브 질의
    while True:
        query = input("\n질문 입력(exit 종료): ")
        if query.lower() == "exit":
            print("종료합니다.")
            break

        results = retriever.retrieve(query, k=3)

        print("\n검색 결과:")
        for text, score in results:
            preview = text[:300].replace("\n", " ")
            print(f"- ({score:.3f}) {preview}...")
