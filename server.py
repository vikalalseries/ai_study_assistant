"""StudyBuddy web server.

    python server.py        then open http://127.0.0.1:8000

A thin FastAPI layer over the existing agent / memory / rag modules. Single-user: one agent session lives in memory.
"""
import ast
import asyncio
import io
import json
import os
import re
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader

try:
    from . import memory, rag, storage
    from .agent import NOTES_FILE
    from .main import start_session
except ImportError:  # run as a plain script from inside the folder
    import memory
    import rag
    import storage
    from agent import NOTES_FILE
    from main import start_session

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_BYTES = 2_000_000
MAX_PDF_BYTES = 10_000_000

app = FastAPI(title="StudyBuddy")
_lock = asyncio.Lock()  # one chat turn at a time (the session is shared)
_agent, _session = start_session()


class ChatIn(BaseModel):
    message: str


def _safe_name(name: str) -> str:
    """Strip any path and odd characters from an uploaded / requested filename."""
    return re.sub(r"[^\w.\- ]", "_", Path(name).name).strip()


def _parse_result(result) -> dict | None:
    """Turn a tool result (dict, JSON string, repr string or list of content parts) back into a dict."""
    if isinstance(result, dict):
        return result
    if isinstance(result, (list, tuple)):
        result = "".join(str(getattr(p, "text", p)) for p in result)
    text = str(result)
    for parse in (json.loads, ast.literal_eval):
        try:
            data = parse(text)
            if isinstance(data, dict):
                return data
        except (ValueError, SyntaxError):
            pass
    return None


def _tool_trace(response) -> list[dict]:
    """Which tools the agent used this turn, plus the note passages it read. Defensive: shape varies by version."""
    trace, calls = [], {}
    for msg in getattr(response, "messages", []) or []:
        for c in getattr(msg, "contents", []) or []:
            kind = getattr(c, "type", "")
            if kind == "function_call":
                entry = {"tool": getattr(c, "name", "?"), "sources": [], "passages": []}
                calls[getattr(c, "call_id", id(c))] = entry
                trace.append(entry)
            elif kind == "function_result":
                entry = calls.get(getattr(c, "call_id", None))
                result = getattr(c, "result", None)
                if entry and entry["tool"] == "search_study_material" and result:
                    data = _parse_result(result)
                    if data and data.get("passages"):
                        entry["passages"] = [
                            {"source": p.get("source", "?"), "text": p.get("text", ""), "match": p.get("match")}
                            for p in data["passages"]
                        ]
                        entry["sources"] = sorted({p["source"] for p in entry["passages"]})
                    else:
                        entry["sources"] = sorted(set(re.findall(r'"source":\s*"([^"]+)"', str(result))))
    return trace


def _library() -> list[dict]:
    rag.sync_files()  # pull the latest notes from Vercel Blob (no-op when Blob isn't configured)
    return [
        {"name": p.name, "size": p.stat().st_size}
        for p in sorted(rag.LIBRARY_DIR.iterdir())
        if p.suffix.lower() in rag.SUPPORTED
    ]


def _notes() -> list[dict]:
    if not NOTES_FILE.exists():
        return []
    out = []
    for line in NOTES_FILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\[(.*?)\]\s*(.*)", line)
        if m:
            out.append({"topic": m.group(1), "note": m.group(2)})
    return out


def _state() -> dict:
    s = _session.state
    return {
        "quiz": {"correct": s.get("quiz_correct", 0), "total": s.get("quiz_total", 0)},
        "summary": memory.load_summary(),
        "recent_turns": len(memory.load_turns()),
        "library": _library(),
        "notes": _notes(),
    }


NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}  # always fetch the latest page on refresh


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers=NO_CACHE)


@app.delete("/api/notes/{index}")
def delete_note(index: int):
    """Remove one saved note; `index` is its position in the list shown by /api/state."""
    if not NOTES_FILE.exists():
        raise HTTPException(404, "No saved notes")
    lines = NOTES_FILE.read_text(encoding="utf-8").splitlines()
    valid = [i for i, line in enumerate(lines) if re.match(r"\[(.*?)\]\s*(.*)", line)]
    if not 0 <= index < len(valid):
        raise HTTPException(404, "That note no longer exists")
    del lines[valid[index]]
    NOTES_FILE.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return {"notes": _notes()}


@app.get("/api/state")
def state():
    return _state()


@app.get("/api/history")
def history():
    """Unsummarised turns, so a page refresh can show the recent conversation."""
    return [{"user": u, "assistant": a} for u, a in memory.load_turns()]


@app.post("/api/chat")
async def chat(body: ChatIn):
    global _agent, _session
    question = body.message.strip()
    if not question:
        raise HTTPException(400, "Empty message")
    async with _lock:
        try:
            response = await _agent.run(question, session=_session)
        except Exception as exc:  # model / network errors should reach the UI, not crash the server
            raise HTTPException(502, f"The model call failed: {exc}") from exc
        answer = response.text
        if await memory.record_turn(question, answer):
            _agent, _session = start_session(dict(_session.state))
    return {"answer": answer, "tools": _tool_trace(response), "state": _state()}


@app.post("/api/memory/clear")
async def clear_memory():
    global _agent, _session
    async with _lock:
        memory.clear()
        _agent, _session = start_session()
    return _state()


@app.post("/api/memory/summarize")
async def summarize_now():
    """Fold the current turns into the summary immediately (the CLI does this on exit)."""
    global _agent, _session
    async with _lock:
        done = await memory.fold_into_summary()
        if done:
            _agent, _session = start_session(dict(_session.state))
    return {"folded": done, "state": _state()}


def _store(name: str, data: bytes) -> None:
    """Save a note locally (cache) and to Vercel Blob (durable storage) when it is configured."""
    rag.LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    if storage.enabled():
        try:
            storage.put(name, data)
        except Exception as exc:
            raise HTTPException(502, f"Could not save {name} to Vercel Blob: {exc}") from exc
    (rag.LIBRARY_DIR / name).write_bytes(data)


def _pdf_to_markdown(name: str, data: bytes) -> tuple[str, int]:
    """Extract text from a PDF. Returns (markdown, pages_without_text). Raises ValueError if unreadable."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError(f"{name}: password-protected PDF, not readable")
        pages = [(p.extract_text() or "").strip() for p in reader.pages]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{name}: not a readable PDF ({exc.__class__.__name__})") from exc
    # Join hard-wrapped lines inside a paragraph; keep blank lines as paragraph breaks.
    paras = [re.sub(r"(?<!\n)\n(?!\n)", " ", t) for t in pages if t]
    if sum(len(t) for t in paras) < 20:
        raise ValueError(f"{name}: not readable. The PDF has only images or scans and no text.")
    return f"# {Path(name).stem}\n\n" + "\n\n".join(paras) + "\n", pages.count("")


@app.post("/api/library/upload")
async def upload(files: list[UploadFile] = File(...)):
    rag.LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    saved, warnings = [], []
    for f in files:
        name = _safe_name(f.filename or "")
        suffix = Path(name).suffix.lower()
        if suffix not in rag.SUPPORTED and suffix != ".pdf":
            raise HTTPException(400, f"{name or 'file'}: only .txt, .md and .pdf files are supported")
        data = await f.read()
        limit = MAX_PDF_BYTES if suffix == ".pdf" else MAX_UPLOAD_BYTES
        if len(data) > limit:
            raise HTTPException(413, f"{name}: larger than {limit // 1_000_000} MB")
        if suffix == ".pdf":
            try:
                text, blank = await asyncio.to_thread(_pdf_to_markdown, name, data)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            name = Path(name).stem + ".md"  # stored as Markdown so the indexer can read it
            data = text.encode("utf-8")
            if blank:
                warnings.append(f"{name}: {blank} page(s) were images and could not be read")
        await asyncio.to_thread(_store, name, data)
        saved.append(name)
    return {"saved": saved, "warnings": warnings, "library": _library()}


class NoteIn(BaseModel):
    title: str
    text: str
    filename: str | None = None  # set when editing an existing file
    overwrite: bool = False


@app.post("/api/library/text")
def save_typed_note(body: NoteIn):
    """Save a note typed in the browser as a library file, so it can be indexed and searched."""
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "The note is empty")
    if body.filename:
        name = _safe_name(body.filename)
        if Path(name).suffix.lower() not in rag.SUPPORTED:
            raise HTTPException(400, "Only .txt and .md files can be edited")
    else:
        stem = _safe_name(body.title).strip(". ")
        if not stem:
            raise HTTPException(400, "Give the note a title")
        name = Path(stem).stem + ".md" if stem.lower().endswith((".md", ".txt")) else stem + ".md"
    if len(text.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Note is larger than {MAX_UPLOAD_BYTES // 1_000_000} MB")
    rag.LIBRARY_DIR.mkdir(exist_ok=True)
    path = rag.LIBRARY_DIR / name
    if path.exists() and not (body.overwrite or body.filename):
        raise HTTPException(409, f"{name} already exists")
    path.write_text(text + "\n", encoding="utf-8")
    return {"saved": name, "library": _library()}


@app.post("/api/quiz/reset")
async def reset_quiz():
    async with _lock:
        _session.state["quiz_correct"] = 0
        _session.state["quiz_total"] = 0
    return _state()


@app.get("/api/library/{name}")
def read_file(name: str):
    rag.sync_files()
    path = rag.LIBRARY_DIR / _safe_name(name)
    if not path.is_file():
        raise HTTPException(404, "Not found")
    return {"name": path.name, "text": path.read_text(encoding="utf-8", errors="ignore")}


@app.delete("/api/library/{name}")
def delete_file(name: str):
    path = rag.LIBRARY_DIR / _safe_name(name)
    if not path.is_file():
        raise HTTPException(404, "Not found")
    if storage.enabled():
        storage.delete(path.name)
    path.unlink()
    return {"library": _library()}


@app.post("/api/ingest")
async def ingest():
    chunks = await asyncio.to_thread(rag.ingest)
    return {"chunks": chunks, "library": _library()}


@app.get("/api/export")
def export():
    return {
        "summary": memory.load_summary(),
        "turns": [{"user": u, "assistant": a} for u, a in memory.load_turns()],
        "notes": _notes(),
        "quiz": _state()["quiz"],
    }


def _deepgram_key() -> str:
    key = os.getenv("DEEPGRAM_API_KEY") or os.getenv("text_to_speech")
    if not key:
        raise HTTPException(503, "Voice typing is not set up: DEEPGRAM_API_KEY is missing")
    return key


@app.post("/api/transcribe")
async def transcribe(request: Request):
    """Speech to text for one short recording (about 4 seconds), sent as the raw request body.

    Plain HTTP, so it works on Vercel. The page uses it when it cannot stream live (see /api/voice-token).
    """
    audio = await request.body()
    if len(audio) < 1000:  # almost nothing recorded: treat as silence
        return {"text": ""}
    if len(audio) > 3_000_000:
        raise HTTPException(413, "That recording is too long")
    headers = {"Authorization": f"Token {_deepgram_key()}", "Content-Type": request.headers.get("content-type", "audio/webm")}
    params = {"model": "nova-3", "language": "multi", "smart_format": "true", "punctuate": "true"}
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            r = await client.post("https://api.deepgram.com/v1/listen", params=params, headers=headers, content=audio)
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Could not reach the speech service") from exc
    if r.status_code != 200:
        raise HTTPException(502, f"The speech service refused the request ({r.status_code})")
    alts = r.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])
    return {"text": (alts[0].get("transcript") or "").strip()}


@app.post("/api/voice-token")
async def voice_token():
    """Live voice typing without a WebSocket on the server (Vercel cannot hold one).

    The browser asks for a short-lived Deepgram token here, then streams the microphone straight to Deepgram.
    The real API key never leaves the server; the token expires after a minute.
    """
    key = _deepgram_key()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post("https://api.deepgram.com/v1/auth/grant",
                                  headers={"Authorization": f"Token {key}"}, json={"ttl_seconds": 60})
    except httpx.HTTPError as exc:
        raise HTTPException(502, "Could not reach the speech service") from exc
    if r.status_code != 200:
        raise HTTPException(502, f"The speech service refused the request ({r.status_code})")
    return {"token": r.json()["access_token"]}


app.mount("/static", StaticFiles(directory=STATIC), name="static")  # deepgram_mic.js; must come before the catch-all


@app.get("/{path:path}", include_in_schema=False)
def spa_fallback(path: str):
    """Refreshing a deep link (e.g. /chat or /notes) should reopen the app, not a 404. API paths still 404."""
    if path.startswith("api/"):
        raise HTTPException(404, "Not found")
    return FileResponse(STATIC / "index.html", headers=NO_CACHE)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
