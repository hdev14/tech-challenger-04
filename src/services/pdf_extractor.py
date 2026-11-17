from dataclasses import dataclass
from pathlib import Path

import pymupdf

from src.core.errors import ReviewRequired


@dataclass(frozen=True)
class PdfPage:
    page: int
    text: str


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def extract_pdf_pages(path: Path, max_text_chars: int) -> list[PdfPage]:
    pages: list[PdfPage] = []
    total = 0
    with pymupdf.open(path) as document:
        for index, page in enumerate(document):
            text = page.get_text("text")
            total += len(text)
            if total > max_text_chars:
                raise ReviewRequired(
                    "PDF_TEXT_LIMIT",
                    "O texto do PDF excede o limite configurado; revisão manual necessária.",
                )
            pages.append(PdfPage(page=index + 1, text=text))
    return pages


def classify_pdf_pages(pages: list[PdfPage]) -> tuple[list[int], list[int]]:
    with_text = [page.page for page in pages if page.text.strip()]
    without_text = [page.page for page in pages if not page.text.strip()]
    return with_text, without_text


def chunk_pdf_pages(pages: list[PdfPage], max_chars: int) -> list[list[PdfPage]]:
    chunks: list[list[PdfPage]] = []
    current: list[PdfPage] = []
    used = 0
    for page in pages:
        text = normalize_whitespace(page.text)
        position = 0
        while position < len(text):
            remaining = max_chars - used
            if remaining == 0:
                chunks.append(current)
                current, used = [], 0
                remaining = max_chars
            end = min(position + remaining, len(text))
            if end < len(text):
                last_space = text.rfind(" ", position, end)
                if last_space > position:
                    end = last_space + 1
            piece = text[position:end]
            current.append(PdfPage(page=page.page, text=piece))
            used += len(piece)
            position = end
    if current:
        chunks.append(current)
    return chunks
