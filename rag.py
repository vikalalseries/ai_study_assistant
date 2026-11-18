"""Retrieval pipeline: load -> chunk -> embed (Gemini) -> store (ChromaDB), then embed query -> search.

Notes are kept in Vercel Blob when BLOB_READ_WRITE_TOKEN is set (see storage.py) and cached in LIBRARY_DIR;
ChromaDB is rebuilt from that cache whenever the Blob contents change (e.g. on a cold start).
"""
import os
import re
import threading
from pathlib import Path

import chromadb
from google import genai
from google.genai import types

try:
    from . import storage
    from .config import DATA_DIR, ROOT  # also loads .env
except ImportError:  # run as a plain script from inside the folder
    import storage
    from config import DATA_DIR, ROOT

LIBRARY_DIR = DATA_DIR / "library"
DB_DIR = DATA_DIR / "chroma_db"
COLLECTION = "study_notes_gemini"  # new name: the old MiniLM index has a different vector size
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMS = 768
EMBED_BATCH = 100  # max texts per embed request
MAX_CHUNK_CHARS = 600
SUPPORTED = {".txt", ".md"}

_lock = threading.RLock()
_files_sig: str | None = None    # Blob listing the local cache matches
_from_blob: set[str] = set()     # files this process downloaded; only these may be removed when Blob drops them
_indexed_sig: str | None = None  # Blob listing the Chroma index was built from
_genai: genai.Client | None = None


def _client():
    return chromadb.PersistentClient(path=str(DB_DIR))


def _gemini() -> genai.Client:
    global _genai
    if _genai is None:
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("gemini_api_key")
        _genai = genai.Client(api_key=key)
    return _genai


def embed(texts: list[str], task: str) -> list[list[float]]:
    """Gemini embeddings. task: RETRIEVAL_DOCUMENT for notes, RETRIEVAL_QUERY for questions."""
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = _gemini().models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i:i + EMBED_BATCH],
            config=types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIMS),
        )
        out.extend(e.values for e in resp.embeddings)
    return out


def _collection(client=None):
    # Embeddings are supplied by us, so Chroma needs no embedding function (and no ONNX model).
    return (client or _client()).get_or_create_collection(
        COLLECTION, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )


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


def sync_files() -> str | None:
    """Make LIBRARY_DIR match Blob (no-op without a token). Returns a signature of the Blob contents."""
    global _files_sig
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    if not storage.enabled():
        return None
    with _lock:
        remote = {n: i for n, i in storage.list_files().items() if Path(n).suffix.lower() in SUPPORTED}
        sig = "|".join(f"{n}:{i['size']}:{i['uploaded']}" for n, i in sorted(remote.items()))
        if sig == _files_sig:
            return sig
        for name, info in remote.items():
            (LIBRARY_DIR / name).write_bytes(storage.download(info))  # library is small: refresh all on any change
        for name in _from_blob - remote.keys():  # deleted from Blob elsewhere; never touch local-only files
            (LIBRARY_DIR / name).unlink(missing_ok=True)
        _from_blob.clear()
        _from_blob.update(remote)
        _files_sig = sig
        return sig


def ingest() -> int:
    """Rebuild the vector database from every .txt/.md file in the library. Returns chunk count."""
    global _indexed_sig
    with _lock:
        sig = sync_files()
        client = _client()
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass  # first run: nothing to delete
        collection = _collection(client)

        ids, docs, metas = [], [], []
        for path in sorted(LIBRARY_DIR.iterdir()):
            if path.suffix.lower() not in SUPPORTED:
                continue
            for i, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8", errors="ignore"))):
                ids.append(f"{path.name}-{i}")
                docs.append(chunk)
                metas.append({"source": path.name})
        if docs:
            collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embed(docs, "RETRIEVAL_DOCUMENT"))
        _indexed_sig = sig
        return len(docs)


def prepare() -> None:
    """Before searching: if Blob changed since the index was built (cold start, other instance), re-index."""
    if not storage.enabled():
        return
    with _lock:
        sig = sync_files()
        if sig != _indexed_sig:
            ingest()


def search_scored(query: str, k: int = 4) -> list[tuple[str, str, int]]:
    """Return the k closest (source, text, match%) chunks. Match% is cosine similarity (0-100)."""
    prepare()
    collection = _collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=embed([query], "RETRIEVAL_QUERY"), n_results=min(k, collection.count())
    )
    out = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        # Cosine space: distance = 1 - cosine similarity.
        out.append((meta["source"], doc, max(0, min(100, round((1 - dist) * 100)))))
    return out


def search(query: str, k: int = 4) -> list[tuple[str, str]]:
    """Return the k closest (source, text) chunks to the query."""
    return [(s, t) for s, t, _ in search_scored(query, k)]
