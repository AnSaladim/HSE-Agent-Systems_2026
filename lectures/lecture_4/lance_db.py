"""
Функции для работы с LanceDB.

Храним page-level чанки и заранее считаем embeddings.
Основной retrieval затем идёт in-memory.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import lancedb
import pandas as pd

from data_preporation import build_page_documents, infer_year_from_path
from embedder import OpenAIEmbedder


REQUIRED_COLUMNS = {
    "chunk_id",
    "source",
    "source_file",
    "year",
    "page",
    "text",
    "search_text",
    "vector",
}


def connect_db(vectorstore_dir: str | Path):
    return lancedb.connect(str(vectorstore_dir))


def table_exists(db: Any, table_name: str) -> bool:
    try:
        return table_name in set(db.table_names())
    except Exception:
        return False


def read_all(table: Any) -> pd.DataFrame:
    """
    Считать всю таблицу в DataFrame.

    В некоторых версиях LanceDB вызов `table.to_pandas()` или чтение через
    `search()` без надёжного обходного пути на практике может возвращать не всю
    таблицу, а ограниченный срез. Поэтому здесь читаем первые `count_rows()`
    строк через `head(...)`.
    """
    total_rows = int(table.count_rows())
    if total_rows == 0:
        return pd.DataFrame()

    try:
        df = table.head(total_rows).to_pandas()
    except Exception:
        # Fallback на случай несовместимой версии LanceDB.
        df = table.search().limit(total_rows).to_pandas()

    if "chunk_id" in df.columns:
        df = df.sort_values("chunk_id").reset_index(drop=True)
    return df


def _expected_sources_and_years(dataset_dir: Path) -> tuple[set[str], set[int]]:
    pdf_paths = sorted(dataset_dir.glob("*.pdf"))
    sources = {p.name for p in pdf_paths}
    years = {year for p in pdf_paths if (year := infer_year_from_path(p)) is not None}
    return sources, years


def _table_is_valid(table: Any, dataset_dir: Path) -> bool:
    try:
        df = read_all(table)
    except Exception:
        return False

    if df.empty:
        return False

    if not REQUIRED_COLUMNS.issubset(set(df.columns)):
        return False

    expected_sources, expected_years = _expected_sources_and_years(dataset_dir)
    actual_sources = set(df["source"].dropna().astype(str))
    actual_years = set(df["year"].dropna().astype(int))

    if expected_sources and actual_sources != expected_sources:
        return False
    if expected_years and actual_years != expected_years:
        return False

    expected_docs = build_page_documents(dataset_dir)
    if len(df) != len(expected_docs):
        return False

    return True


def build_vectorstore(
    client: Any,
    dataset_dir: str | Path = "dataset",
    vectorstore_dir: str | Path = "lance_db/vectorstore",
    table_name: str = "chunks",
    embedding_model: str = "text-embedding-3-small",
    force_rebuild: bool = False,
):
    """
    Построить или переоткрыть локальное хранилище LanceDB.

    Если таблица уже существует, мы валидируем её against current
    dataset. Это защищает от случая, когда в старой базе остался только один PDF
    или таблица была собрана до исправления ingestion.
    """
    vectorstore_dir = Path(vectorstore_dir)
    dataset_dir = Path(dataset_dir)

    if force_rebuild and vectorstore_dir.exists():
        shutil.rmtree(vectorstore_dir, ignore_errors=True)

    db = connect_db(vectorstore_dir)

    if table_exists(db, table_name) and not force_rebuild:
        table = db.open_table(table_name)
        if _table_is_valid(table, dataset_dir):
            return table

        # stale / incomplete table -> rebuild
        shutil.rmtree(vectorstore_dir, ignore_errors=True)
        db = connect_db(vectorstore_dir)

    records = build_page_documents(dataset_dir)
    embedder = OpenAIEmbedder(client=client, model=embedding_model)
    vectors = embedder.encode([row["text"] for row in records])

    for row, vector in zip(records, vectors):
        row["vector"] = vector

    return db.create_table(table_name, data=records, mode="overwrite")
