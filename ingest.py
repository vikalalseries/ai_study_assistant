"""Run once (and again whenever library/ changes):  python ingest.py"""
try:
    from .rag import LIBRARY_DIR, ingest
except ImportError:  # run as a plain script from inside the folder
    from rag import LIBRARY_DIR, ingest


def ingest_library() -> int:
    count = ingest()
    print(f"Indexed {count} chunks from {LIBRARY_DIR}")
    return count


if __name__ == "__main__":
    ingest_library()
