import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "documentDB")
DOCUMENT_COLLECTION = os.getenv("DOCUMENT_COLLECTION", "pdf_vectors")
CACHE_COLLECTION = os.getenv("CACHE_COLLECTION", "query_cache")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
ANSWER_MODEL_NAME = os.getenv("ANSWER_MODEL_NAME", "google/flan-t5-small")
ENABLE_TRANSFORMER_EMBEDDINGS = os.getenv("ENABLE_TRANSFORMER_EMBEDDINGS", "1") == "1"
ENABLE_LOCAL_ANSWER_MODEL = os.getenv("ENABLE_LOCAL_ANSWER_MODEL", "1") == "1"
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "pdf")
HASH_EMBEDDING_DIMENSIONS = 384
ANSWER_VERSION = "extractive-v2"


mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = mongo_client[MONGO_DB]
documents_collection = db[DOCUMENT_COLLECTION]
cache_collection = db[CACHE_COLLECTION]

embedding_tokenizer = None
embedding_model = None
embedding_load_error = None
answer_generator = None
answer_generator_load_error = None

app = FastAPI(title="Context-Aware Document Retrieval API")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class IngestRequest(BaseModel):
    pdf_dir: str = Field(default="pdf")
    force: bool = False


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_answer_sentence(sentence: str) -> str:
    sentence = normalize_text(sentence)
    sentence = re.sub(r"^[A-Z][A-Za-z\s-]{8,80}\s+(Introduction|Overview|Conclusion)\.?\s+", "", sentence)
    sentence = re.sub(r"^(Introduction|Overview|Conclusion|Business Context|Data Pipeline|Deployment)\.?\s+", "", sentence)
    return sentence.strip()


def query_terms(query: str) -> set[str]:
    terms = {term.lower() for term in re.findall(r"\w+", query) if len(term) > 2}
    if "rag" in terms:
        terms.update({"retrieval", "augmented", "generation", "pipeline", "context", "documents"})
    if "llm" in terms:
        terms.update({"large", "language", "model"})
    return terms


def is_definition_query(query: str) -> bool:
    return bool(re.search(r"\b(what is|what are|define|explain)\b", query.lower()))


def load_embedding_model() -> bool:
    global embedding_load_error, embedding_model, embedding_tokenizer

    if not ENABLE_TRANSFORMER_EMBEDDINGS:
        embedding_load_error = "disabled"
        return False
    if embedding_tokenizer is not None and embedding_model is not None:
        return True
    if embedding_load_error is not None:
        return False

    try:
        embedding_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        embedding_model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)
        embedding_model.eval()
        return True
    except Exception as exc:
        embedding_load_error = str(exc)
        embedding_tokenizer = None
        embedding_model = None
        return False


def make_hash_embedding(text: str) -> list[float]:
    vector = np.zeros(HASH_EMBEDDING_DIMENSIONS, dtype=np.float32)
    terms = re.findall(r"\w+", text.lower())

    for term in terms:
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % HASH_EMBEDDING_DIMENSIONS
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return vector.tolist()


def make_transformer_embedding(text: str) -> list[float]:
    if embedding_tokenizer is None or embedding_model is None:
        return make_hash_embedding(text)

    inputs = embedding_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = embedding_model(**inputs)

    token_embeddings = outputs.last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.squeeze().cpu().numpy().astype(np.float32).tolist()


def make_embedding(text: str) -> list[float]:
    if load_embedding_model():
        return make_transformer_embedding(text)
    return make_hash_embedding(text)


def embedding_status() -> str:
    if embedding_tokenizer is not None and embedding_model is not None:
        return MODEL_NAME
    if load_embedding_model():
        return MODEL_NAME
    return f"hash-fallback-{HASH_EMBEDDING_DIMENSIONS}d"


def active_embedding_model_name() -> str:
    return embedding_status()


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def cache_key(query: str, top_k: int) -> str:
    total_docs = documents_collection.count_documents({})
    latest_doc = documents_collection.find_one(sort=[("updated_at", -1)])
    latest_update = latest_doc.get("updated_at", "") if latest_doc else ""
    raw_key = (
        f"{query.strip().lower()}::{top_k}::{total_docs}::{latest_update}::"
        f"{embedding_status()}::{ANSWER_MODEL_NAME}::{ANSWER_VERSION}"
    )
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def rank_documents(query: str, top_k: int) -> list[dict[str, Any]]:
    query_vector = np.array(make_embedding(query), dtype=np.float32)
    results: list[dict[str, Any]] = []

    active_model = active_embedding_model_name()
    for document in documents_collection.find({"vector": {"$exists": True}}):
        if document.get("embedding_model") != active_model:
            continue
        doc_vector = np.array(document["vector"], dtype=np.float32)
        if doc_vector.shape != query_vector.shape:
            continue
        score = cosine_similarity(query_vector, doc_vector)
        results.append(
            {
                "filename": document.get("filename", "unknown"),
                "chunk_index": document.get("chunk_index", 0),
                "content": document.get("content", ""),
                "score": score,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def compatible_document_count() -> int:
    expected_size = len(make_embedding("dimension check"))
    active_model = active_embedding_model_name()
    count = 0
    for document in documents_collection.find(
        {"vector": {"$exists": True}},
        {"vector": 1, "embedding_model": 1},
    ):
        if document.get("embedding_model") == active_model and len(document.get("vector", [])) == expected_size:
            count += 1
    return count


def build_extractive_answer(query: str, ranked_docs: list[dict[str, Any]]) -> str:
    terms = query_terms(query)
    wants_definition = is_definition_query(query)
    candidate_sentences: list[tuple[float, float, str, str]] = []

    for doc in ranked_docs:
        sentences = re.split(r"(?<=[.!?])\s+", doc["content"])
        for sentence in sentences:
            normalized = clean_answer_sentence(sentence)
            if not normalized:
                continue
            if len(normalized.split()) < 6:
                continue
            lowered = normalized.lower()
            overlap = sum(1 for term in terms if term in lowered)
            answer_score = float(overlap)

            if wants_definition:
                if "retrieval augmented generation" in lowered:
                    answer_score += 4
                if any(marker in lowered for marker in (" usually starts", " is ", " are ", " approach")):
                    answer_score += 2
                if lowered.startswith(("evaluation.", "caching.", "chunking strategy.", "embedding search.")):
                    answer_score -= 2

            candidate_sentences.append((answer_score, doc["score"], doc["filename"], normalized))

    candidate_sentences.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [item for item in candidate_sentences if item[0] > 0][:3]
    if not selected:
        selected = candidate_sentences[:3]

    if not selected:
        return "I could not find enough document context to answer this query."

    answer_lines: list[str] = []
    for _, _, _, sentence in selected:
        if any(sentence in existing or existing in sentence for existing in answer_lines):
            continue
        answer_lines.append(sentence)

    sources = sorted({filename for _, _, filename, _ in selected})
    return " ".join(answer_lines) + f"\n\nSources: {', '.join(sources)}"


def get_answer_generator():
    global answer_generator, answer_generator_load_error

    if not ENABLE_LOCAL_ANSWER_MODEL:
        return None
    if answer_generator is None:
        if answer_generator_load_error is not None:
            return None
        try:
            answer_tokenizer = AutoTokenizer.from_pretrained(ANSWER_MODEL_NAME, local_files_only=True)
            answer_model = AutoModelForSeq2SeqLM.from_pretrained(ANSWER_MODEL_NAME, local_files_only=True)
            answer_generator = pipeline(
                "text2text-generation",
                model=answer_model,
                tokenizer=answer_tokenizer,
            )
        except Exception as exc:
            answer_generator_load_error = str(exc)
            return None
    return answer_generator


def build_generated_answer(query: str, ranked_docs: list[dict[str, Any]]) -> str | None:
    context = "\n".join(
        f"Source {index + 1} ({doc['filename']}): {doc['content'][:1200]}"
        for index, doc in enumerate(ranked_docs[:4])
    )
    prompt = (
        "Answer the question using only the context. "
        "Write 2 concise sentences. Do not mention information that is not in the context.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:"
    )

    try:
        generator = get_answer_generator()
        if generator is None:
            return None
        generated = generator(
            prompt,
            max_new_tokens=120,
            do_sample=False,
            truncation=True,
        )[0]["generated_text"]
    except Exception:
        return None

    generated = normalize_text(generated)
    if not generated or len(generated.split()) < 8:
        return None

    sources = sorted({doc["filename"] for doc in ranked_docs[:3]})
    return f"{generated}\n\nSources: {', '.join(sources)}"


def build_answer(query: str, ranked_docs: list[dict[str, Any]]) -> tuple[str, str]:
    generated_answer = build_generated_answer(query, ranked_docs)
    if generated_answer:
        return generated_answer, "local-generator"
    return build_extractive_answer(query, ranked_docs), "extractive-fallback"


def serialize_search(query: str, top_k: int, results: list[dict[str, Any]], cached: bool) -> dict[str, Any]:
    context = "\n\n".join(
        f"[{index + 1}] {doc['filename']} (score: {doc['score']:.4f})\n{doc['content']}"
        for index, doc in enumerate(results)
    )
    answer, answer_mode = build_answer(query, results)
    return {
        "query": query,
        "top_k": top_k,
        "cached": cached,
        "answer": answer,
        "answer_mode": answer_mode,
        "context": context,
        "results": results,
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    try:
        total_documents = documents_collection.count_documents({})
    except ServerSelectionTimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"MongoDB unavailable at {MONGO_URI}. Start MongoDB and try again.",
        ) from exc

    return {
        "status": "alive",
        "documents": total_documents,
        "compatible_documents": compatible_document_count() if total_documents else 0,
        "model": embedding_status(),
        "answer_model": ANSWER_MODEL_NAME if answer_generator is not None else "extractive-fallback",
    }


@app.post("/search")
def search_endpoint(request: SearchRequest) -> dict[str, Any]:
    query = normalize_text(request.query)
    key = cache_key(query, request.top_k)
    cached_result = cache_collection.find_one({"cache_key": key}, {"_id": 0})
    if cached_result:
        cached_result["cached"] = True
        return cached_result

    ranked_docs = rank_documents(query, request.top_k)
    if not ranked_docs:
        raise HTTPException(status_code=404, detail="No indexed documents found.")

    response = serialize_search(query, request.top_k, ranked_docs, cached=False)
    cache_collection.update_one(
        {"cache_key": key},
        {
            "$set": {
                **response,
                "cache_key": key,
                "created_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )
    return response


@app.post("/ingest")
def ingest_endpoint(request: IngestRequest) -> dict[str, Any]:
    from pdf import upload_pdfs_as_vectors

    return upload_pdfs_as_vectors(request.pdf_dir, force=request.force)


@app.post("/upload")
async def upload_endpoint(
    files: list[UploadFile] = File(...),
    force: bool = Form(False),
) -> dict[str, Any]:
    from pdf import upload_pdfs_as_vectors

    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []

    for uploaded_file in files:
        filename = Path(uploaded_file.filename or "").name
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {filename}")

        destination = upload_dir / filename
        content = await uploaded_file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {filename}")

        destination.write_bytes(content)
        saved_files.append(filename)

    ingest_result = upload_pdfs_as_vectors(os.fspath(upload_dir), force=force, filenames=saved_files)
    return {
        "uploaded_files": saved_files,
        **ingest_result,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
