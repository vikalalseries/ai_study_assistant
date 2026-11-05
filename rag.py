"""Retrieval pipeline: load -> chunk -> embed -> store (ingest), then embed query -> search."""
import re
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parent
LIBRARY_DIR = ROOT / "library"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "study_notes"
MAX_CHUNK_CHARS = 600
SUPPORTED = {".txt", ".md"}


def _client():
    # Chroma's default embedding function is a local ONNX MiniLM model (no API key needed).
    return chromadb.PersistentClient(path=str(DB_DIR))


def chunk_text(text: str) -> list[str]:
    """Split on blank lines, then break any paragraph longer than MAX_CHUNK_CHARS by sentence."""
    chunks = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        current = ""
        for sentence in re.split(r"(?<=[.!?])\s+", para):
            if current and len(current) + len(sentence) > MAX_CHUNK_CHARS:
                chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            chunks.append(current)
    return chunks


def ingest() -> int:
    """Rebuild the vector database from every .txt/.md file in library/. Returns chunk count."""
    LIBRARY_DIR.mkdir(exist_ok=True)
    client = _client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # first run: nothing to delete
    collection = client.get_or_create_collection(COLLECTION)

    ids, docs, metas = [], [], []
    for path in sorted(LIBRARY_DIR.iterdir()):
        if path.suffix.lower() not in SUPPORTED:
            continue
        for i, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8", errors="ignore"))):
            ids.append(f"{path.name}-{i}")
            docs.append(chunk)
            metas.append({"source": path.name})
    if docs:
        collection.add(ids=ids, documents=docs, metadatas=metas)
    return len(docs)


def search(query: str, k: int = 4) -> list[tuple[str, str]]:
    """Return the k closest (source, text) chunks to the query."""
    collection = _client().get_or_create_collection(COLLECTION)
    if collection.count() == 0:
        return []
    result = collection.query(query_texts=[query], n_results=min(k, collection.count()))
    return [
        (meta["source"], doc)
        for doc, meta in zip(result["documents"][0], result["metadatas"][0])
    ]
