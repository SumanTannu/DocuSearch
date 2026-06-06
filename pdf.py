import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz

from file import active_embedding_model_name, cache_collection, documents_collection, make_embedding, normalize_text


def extract_text_from_pdf(pdf_path: Path) -> str:
    text_parts: list[str] = []
    with fitz.open(pdf_path) as document:
        for page in document:
            text_parts.append(page.get_text())
    return normalize_text(" ".join(text_parts))


def chunk_text(text: str, max_words: int = 350, overlap_words: int = 60) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max_words - overlap_words
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks


def upload_pdfs_as_vectors(
    pdf_dir: str = "pdf",
    force: bool = False,
    filenames: list[str] | None = None,
) -> dict[str, Any]:
    directory = Path(pdf_dir)
    if not directory.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {directory}")

    inserted = 0
    skipped = 0
    updated_at = datetime.now(timezone.utc)
    allowed_filenames = set(filenames or [])

    for pdf_path in sorted(directory.glob("*.pdf")):
        if allowed_filenames and pdf_path.name not in allowed_filenames:
            continue

        text_content = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text_content)

        if force:
            documents_collection.delete_many({"filename": pdf_path.name})

        existing_count = documents_collection.count_documents({"filename": pdf_path.name})
        if existing_count and not force:
            skipped += existing_count
            continue

        records = [
            {
                "filename": pdf_path.name,
                "chunk_index": index,
                "content": chunk,
                "vector": make_embedding(chunk),
                "embedding_model": active_embedding_model_name(),
                "updated_at": updated_at,
            }
            for index, chunk in enumerate(chunks)
        ]
        if records:
            documents_collection.insert_many(records)
            inserted += len(records)

    if inserted:
        cache_collection.delete_many({})

    return {
        "pdf_dir": os.fspath(directory),
        "processed_files": sorted(allowed_filenames) if allowed_filenames else "all",
        "inserted_chunks": inserted,
        "skipped_chunks": skipped,
        "documents_in_index": documents_collection.count_documents({}),
    }


if __name__ == "__main__":
    result = upload_pdfs_as_vectors(os.getenv("PDF_DIR", "pdf"), force=os.getenv("FORCE_INGEST") == "1")
    print(result)
