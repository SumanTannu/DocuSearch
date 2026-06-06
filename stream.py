import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="Document Retrieval", layout="wide")
st.title("Context-Aware Document Retrieval")


def call_api(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, str(exc)


health, health_error = call_api("GET", "/health")
if health_error:
    st.error(f"Backend unavailable: {health_error}")
    st.stop()
else:
    st.caption(
        f"Backend ready | Indexed chunks: {health['documents']} | "
        f"Searchable chunks: {health['compatible_documents']} | "
        f"Embedding model: {health['model']} | Answer model: {health['answer_model']}"
    )
    if health["documents"] and not health["compatible_documents"]:
        st.warning("Existing vectors use a different embedding model. Rebuild the index before searching.")


with st.sidebar:
    st.header("Index")
    uploaded_pdfs = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    force = st.checkbox("Rebuild existing index")
    if st.button("Upload and index", use_container_width=True):
        if not uploaded_pdfs:
            st.warning("Choose at least one PDF.")
        else:
            files = [
                ("files", (pdf.name, pdf.getvalue(), "application/pdf"))
                for pdf in uploaded_pdfs
            ]
            result, error = call_api("POST", "/upload", files=files, data={"force": str(force).lower()})
        if uploaded_pdfs and error:
            st.error(error)
        elif uploaded_pdfs:
            uploaded_names = ", ".join(result["uploaded_files"])
            st.success(
                f"Uploaded {uploaded_names}. "
                f"Inserted {result['inserted_chunks']} chunks. "
                f"Skipped {result['skipped_chunks']}. "
                f"Index total: {result['documents_in_index']}."
            )

    with st.expander("Server-side indexing"):
        pdf_dir = st.text_input("PDF directory", value="pdf")
        if st.button("Index existing folder", use_container_width=True):
            payload = {"pdf_dir": pdf_dir, "force": force}
            result, error = call_api("POST", "/ingest", json=payload)
            if error:
                st.error(error)
            else:
                st.success(
                    f"Inserted {result['inserted_chunks']} chunks. "
                    f"Skipped {result['skipped_chunks']}. "
                    f"Index total: {result['documents_in_index']}."
                )


query = st.text_input("Search query", placeholder="Ask something about the indexed documents")
indexed_chunks = health.get("compatible_documents", 0) if health else 0
if indexed_chunks < 1:
    st.info("Upload and index PDFs before searching.")
    top_k = 1
elif indexed_chunks == 1:
    top_k = 1
    st.caption("Results: 1")
else:
    max_results = min(20, indexed_chunks)
    default_results = min(5, max_results)
    top_k = st.slider("Results", min_value=1, max_value=max_results, value=default_results)

if st.button("Search", type="primary"):
    if not query.strip():
        st.warning("Enter a query first.")
    else:
        payload = {"query": query, "top_k": top_k}
        result, error = call_api("POST", "/search", json=payload)
        if error:
            st.error(error)
        else:
            cache_label = "cached" if result.get("cached") else "fresh"
            answer_mode = result.get("answer_mode", "unknown")
            st.subheader(f"Answer ({cache_label}, {answer_mode})")
            st.write(result["answer"])

            st.subheader("Retrieved Context")
            for item in result["results"]:
                with st.expander(f"{item['filename']} | score {item['score']:.4f}"):
                    st.write(item["content"])
