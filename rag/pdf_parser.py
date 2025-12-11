# rag/pdf_parser.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader


def pdf_to_text(path: str) -> str:
    """PDF 전체 텍스트 추출 (단순 버전)"""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size=300, overlap=50) -> List[str]:
    """긴 텍스트를 문자 길이 기준으로 청킹"""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


# ----------------------------
# 1) 구조화된 조항 단위 청킹
# ----------------------------

@dataclass
class KBOChunk:
    text: str
    chapter: str | None
    article: str | None
    page_start: int
    page_end: int


def extract_kbo_chunks(path: str) -> List[KBOChunk]:
    """
    KBO 리그 규정 PDF를:
    - 제n장 / 제n조 패턴 기준으로
    - 조항 단위로 청킹하고
    - 장/조/페이지 정보까지 담아서 반환
    """
    reader = PdfReader(path)

    chapter_pattern = re.compile(r"^제\s*\d+\s*장\s*.*")
    article_pattern = re.compile(r"^제\s*\d+\s*조\s*.*")

    chunks: List[KBOChunk] = []

    current_chapter: str | None = None
    current_article: str | None = None
    current_lines: List[str] = []
    current_page_start: int | None = None
    current_page_end: int | None = None

    def flush_chunk():
        nonlocal current_lines, current_page_start, current_page_end
        if not current_lines:
            return
        text = "\n".join(current_lines).strip()
        if not text:
            return
        chunk = KBOChunk(
            text=text,
            chapter=current_chapter,
            article=current_article,
            page_start=current_page_start if current_page_start is not None else 1,
            page_end=current_page_end if current_page_end is not None else current_page_start or 1,
        )
        chunks.append(chunk)
        current_lines = []
        current_page_start = None
        current_page_end = None

    # 페이지 순회
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_text = page.extract_text() or ""
        lines = page_text.splitlines()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 제n장 패턴
            if chapter_pattern.match(stripped):
                # 앞에 쌓인 내용이 있으면 이전 조항 청크로 flush
                flush_chunk()
                current_chapter = stripped
                # 장 제목 줄도 포함시키고 싶으면 아래 주석 해제
                # current_lines.append(stripped)
                continue

            # 제n조 패턴
            if article_pattern.match(stripped):
                # 앞에 쌓인 내용이 있으면 이전 조항 청크로 flush
                flush_chunk()
                current_article = stripped
                current_page_start = page_num
                current_page_end = page_num
                current_lines.append(stripped)
                continue

            # 일반 내용 줄
            if current_page_start is None:
                current_page_start = page_num
            current_page_end = page_num
            current_lines.append(stripped)

    # 마지막 청크 flush
    flush_chunk()

    return chunks
