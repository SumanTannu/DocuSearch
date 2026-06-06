# Context-Aware Document Retrieval

This repo implements a document retrieval system with:

- PDF upload, ingestion, and chunking
- Semantic embeddings with cosine similarity search
- Local answer generation from retrieved context, with extractive fallback
- MongoDB storage for document vectors
- MongoDB query-result caching
- FastAPI backend
- Streamlit UI

## Requirements

- Python 3.11+
- MongoDB running locally on `mongodb://localhost:27017/`

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run Locally

Start MongoDB first. If the Windows service is working:

```powershell
Start-Service MongoDB
```

Start the FastAPI backend in one terminal:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn file:app --host 127.0.0.1 --port 8000
```

Start the Streamlit UI in a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run stream.py
```

Open the UI:

```text
http://localhost:8501
```

The backend API is available at:

```text
http://localhost:8000
```

## Use The App

1. Open `http://localhost:8501`.
2. Upload one or more PDFs from the sidebar.
3. Check `Rebuild existing index` when you want a fresh vector index.
4. Click `Upload and index`.
5. Ask a question in the search box.

## Troubleshooting

If Hugging Face downloads are blocked or reset, the backend still works with local fallbacks. The embedding model is only loaded from your local Hugging Face cache. If it is not cached, the UI health line will show:

```text
Embedding model: hash-fallback-384d
```

If you already have the Hugging Face model cached locally, the backend uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

If the UI says MongoDB is unavailable at `mongodb://localhost:27017/`, verify MongoDB is listening:

```powershell
mongosh "mongodb://127.0.0.1:27017/" --eval "db.runCommand({ ping: 1 })"
```

## Models

The default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The default local answer model is:

```text
google/flan-t5-small
```

The first run can take longer because Python packages and model files may need to download.

The answer model is also loaded from the local Hugging Face cache only. If `google/flan-t5-small` is not cached, searches automatically use the extractive fallback instead of retrying a Hugging Face download.

To force the no-download fallbacks:

```powershell
$env:ENABLE_TRANSFORMER_EMBEDDINGS = "0"
$env:ENABLE_LOCAL_ANSWER_MODEL = "0"
uvicorn file:app --host 127.0.0.1 --port 8000
```
