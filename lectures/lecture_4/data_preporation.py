"""
Подготовка PDF-данных для векторного хранилища.

Основная идея для этой домашней работы — page-aware ingestion:
- один PDF -> набор страниц;
- одна страница -> один retrievable chunk.

"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
from pypdf import PdfReader


_YEAR_BY_FILENAME = {
    "Potrebitelskie_ozhidaniya_2_2019.pdf": 2019,
    "Consumer_sentiment_2Q2025.pdf": 2025,
}

_MOJIBAKE_RE = re.compile(r"[ÐÑÒÓÃÅËÏðñòóãåëï]{8,}")


def infer_year_from_path(pdf_path: str | Path) -> int | None:
    """Определить год документа по имени файла."""
    name = Path(pdf_path).name
    if name in _YEAR_BY_FILENAME:
        return _YEAR_BY_FILENAME[name]

    match = re.search(r"20\d{2}", name)
    if match:
        return int(match.group(0))
    return None


def clean_pdf_text(text: str) -> str:
    """Очистить типовой шум после извлечения текста из PDF."""
    if not text:
        return ""

    text = text.replace("\x0c", "\n").replace("\u00ad", "")
    text = text.replace("\xa0", " ")
    text = re.sub(r"([A-Za-zА-Яа-яЁё])-\n([A-Za-zА-Яа-яЁё])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Убираем типичные «дыры» внутри слов из PDF-слоя.
    text = re.sub(r"(?<=[А-Яа-яЁё])\s{2,}(?=[А-Яа-яЁё])", "", text)
    text = re.sub(r"(?<=[A-Za-z])\s{2,}(?=[A-Za-z])", "", text)

    # Косметика вокруг пунктуации.
    text = re.sub(r"\s+([,.;:%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return text.strip()


def build_search_text(text: str) -> str:
    """Нормализованный текст для lexical retrieval."""
    text = clean_pdf_text(text).replace("ё", "е").lower()
    text = re.sub(r"[^\w\s%.-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_like_mojibake(text: str) -> bool:
    """
    Грубая эвристика для страниц, которые pypdf прочитал в битой кодировке.
    На таких страницах часто встречаются последовательности вида 'ÐàÑ...' / 'Ïî...'.
    """
    if not text:
        return False

    cleaned = text.strip()
    if not cleaned:
        return False

    if _MOJIBAKE_RE.search(cleaned):
        return True

    bad_chars = sum(ch in "ÐÑÒÓÃÅËÏðñòóãåëï" for ch in cleaned)
    return bad_chars >= 20 and bad_chars / max(len(cleaned), 1) >= 0.015


def _extract_text_pypdf(pdf_path: Path) -> list[str]:
    """Извлечение текста по страницам через pypdf."""
    reader = PdfReader(str(pdf_path))
    return [clean_pdf_text(page.extract_text() or "") for page in reader.pages]


def _extract_text_pymupdf(pdf_path: Path) -> list[str]:
    """Fallback-извлечение текста по страницам через PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages: list[str] = []
    for page in doc:
        pages.append(clean_pdf_text(page.get_text("text") or ""))
    doc.close()
    return pages


def _strip_repeated_lines(pages: list[dict]) -> list[dict]:
    """
    Удаляем короткие строки, которые повторяются на большинстве страниц одного PDF:
    running headers, одинаковые подзаголовки, номера страниц.
    """
    line_counter = Counter()

    split_pages: list[list[str]] = []
    for page in pages:
        lines = [ln.strip() for ln in str(page["text"]).splitlines() if ln.strip()]
        split_pages.append(lines)
        for ln in set(lines):
            line_counter[ln] += 1

    min_repeat = max(3, int(len(pages) * 0.6))
    repeated = {
        ln
        for ln, cnt in line_counter.items()
        if cnt >= min_repeat
        and len(ln) <= 120
    }

    cleaned_pages = []
    for page, lines in zip(pages, split_pages):
        new_lines = []
        for ln in lines:
            if ln in repeated:
                continue
            if re.fullmatch(r"\d+", ln):
                continue
            new_lines.append(ln)

        cleaned_text = clean_pdf_text("\n".join(new_lines))
        if not cleaned_text:
            continue

        page = dict(page)
        page["text"] = cleaned_text
        page["raw_text"] = cleaned_text
        page["search_text"] = build_search_text(cleaned_text)
        page["text_len"] = len(cleaned_text)
        cleaned_pages.append(page)

    return cleaned_pages


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """
    Извлечь текст по страницам.

    Сначала используем pypdf. Если конкретная страница получилась пустой
    или выглядит как mojibake/битая кодировка, дочитываем её через PyMuPDF.
    """
    pdf_path = Path(pdf_path)
    pypdf_pages = _extract_text_pypdf(pdf_path)
    pymupdf_pages: list[str] | None = None

    pages: list[dict] = []
    for page_idx, text in enumerate(pypdf_pages):
        raw_text = text
        visual_text = ""

        need_fallback = not raw_text.strip() or _looks_like_mojibake(raw_text)
        if need_fallback:
            if pymupdf_pages is None:
                pymupdf_pages = _extract_text_pymupdf(pdf_path)
            fallback_text = pymupdf_pages[page_idx] if page_idx < len(pymupdf_pages) else ""
            if fallback_text.strip():
                raw_text = fallback_text
                visual_text = fallback_text

        cleaned = clean_pdf_text(raw_text)
        if not cleaned:
            continue

        pages.append(
            {
                "page": page_idx + 1,
                "text": cleaned,
                "raw_text": cleaned,
                "visual_text": visual_text,
                "has_visual_text": bool(visual_text.strip()),
                "search_text": build_search_text(cleaned),
                "text_len": len(cleaned),
            }
        )

    return pages


def build_page_documents(dataset_dir: str | Path) -> list[dict]:
    """
    Собрать page-level документы из всех PDF в dataset_dir.

    Дополнительно сохраняем несколько служебных колонок для совместимости
    с разными версиями ноутбука и для отладки ingestion.
    """
    dataset_dir = Path(dataset_dir)
    pdf_paths = sorted(dataset_dir.glob("*.pdf"))

    documents: list[dict] = []
    chunk_id = 0

    for pdf_path in pdf_paths:
        year = infer_year_from_path(pdf_path)
        pages = extract_pdf_pages(pdf_path)
        pages = _strip_repeated_lines(pages)

        for page in pages:
            documents.append(
                {
                    "chunk_id": chunk_id,
                    "source": pdf_path.name,
                    "source_file": pdf_path.name,
                    "year": year,
                    **page,
                }
            )
            chunk_id += 1

    return documents
