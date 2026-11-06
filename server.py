"""StudyBuddy web backend (FastAPI). Wraps the existing agent, RAG, and memory modules and serves the UI.

    python server.py            ->  http://127.0.0.1:8000
"""
import asyncio
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import memory
import rag
from agent import NOTES_FILE, build_agent

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_UPLOAD = 2 * 1024 * 1024
RETRIES = 3

state = {"agent": None, "session": None, "conv": None, "lock": asyncio.Lock()}

CONV_DIR = Path(__file__).resolve().parent / "conversations"
MAX_CONVERSATIONS = 10
CONTEXT_MESSAGES = 20


# ---------- conversation history (one JSON file per chat, newest 10 kept) ----------

def conv_path(cid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", cid):
        raise HTTPException(404, "No such chat")
    return CONV_DIR / f"{cid}.json"


def load_conv(cid: str) -> dict | None:
    try:
        return json.loads(conv_path(cid).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_conv(conv: dict) -> None:
    CONV_DIR.mkdir(exist_ok=True)
    conv_path(conv["id"]).write_text(json.dumps(conv, ensure_ascii=False, indent=1), encoding="utf-8")


def all_convs() -> list[dict]:
    CONV_DIR.mkdir(exist_ok=True)
    convs = []
    for p in CONV_DIR.glob("*.json"):
        try:
            convs.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(convs, key=lambda c: c.get("updated", 0), reverse=True)


def prune_convs() -> None:
    for old in all_convs()[MAX_CONVERSATIONS:]:
        if old["id"] != state["conv"]:
            conv_path(old["id"]).unlink(missing_ok=True)


def create_conv(first_message: str) -> dict:
    title = " ".join(first_message.split())
    now = time.time()
    conv = {"id": uuid.uuid4().hex[:12], "title": title[:40] + ("…" if len(title) > 40 else ""),
            "created": now, "updated": now, "messages": [], "quiz": {}}
    save_conv(conv)
    state["conv"] = conv["id"]
    prune_convs()
    return conv


def new_session(quiz_state: dict | None = None) -> None:
    context = memory.context()
    conv = load_conv(state["conv"]) if state["conv"] else None
    if conv and conv["messages"]:
        lines = "\n".join(f"{'Student' if m['role'] == 'user' else 'StudyBuddy'}: {m['text']}"
                          for m in conv["messages"][-CONTEXT_MESSAGES:])
        context += f"\n\nThis chat so far (continue it naturally):\n{lines}"
    agent = build_agent(context)
    session = agent.create_session()
    session.state.update(quiz_state or {})
    state["agent"], state["session"] = agent, session


@asynccontextmanager
async def lifespan(_: FastAPI):
    new_session()
    # Load the local embedding model now so the first real question isn't the slow one.
    try:
        await asyncio.to_thread(rag.search, "warm up")
    except Exception:  # noqa: BLE001 - warm-up is only an optimisation
        pass
    yield


app = FastAPI(title="StudyBuddy", lifespan=lifespan)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def friendly_error(exc: Exception) -> str:
    text = str(exc)
    if "503" in text or "UNAVAILABLE" in text:
        return "The AI model is busy right now. Please try again in a moment."
    if "429" in text or "RESOURCE_EXHAUSTED" in text:
        return "Rate limit reached. Wait a bit and try again."
    if "API key" in text or "401" in text or "403" in text:
        return "The Gemini API key looks invalid. Check gemini_api_key in .env."
    return "Something went wrong talking to the AI model."


# ---------- chat ----------

background: set[asyncio.Task] = set()


async def remember(message: str, answer: str) -> None:
    """Save the turn to long-term memory (and summarise when due) without blocking the reply."""
    async with state["lock"]:
        try:
            if await memory.record_turn(message, answer):
                new_session(dict(state["session"].state))
        except Exception:  # noqa: BLE001 - memory is best-effort
            pass


class ChatIn(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(body: ChatIn):
    message = body.message.strip()
    if not message:
        raise HTTPException(400, "Empty message")

    async def events():
        async with state["lock"]:
            for attempt in range(RETRIES):
                streamed = False
                seen_calls: set[str] = set()
                try:
                    stream = state["agent"].run(message, session=state["session"], stream=True)
                    async for update in stream:
                        for c in update.contents:
                            if c.type == "function_call" and c.name:
                                key = c.call_id or f"{c.name}-{len(seen_calls)}"
                                if key not in seen_calls:
                                    seen_calls.add(key)
                                    yield sse("tool", {"name": c.name})
                                    streamed = True
                            elif c.type == "text" and c.text:
                                yield sse("token", {"text": c.text})
                                streamed = True
                    final = await stream.get_final_response()
                    answer = final.text
                    quiz = {k: state["session"].state.get(k, 0) for k in ("quiz_correct", "quiz_total")}
                    conv = (load_conv(state["conv"]) if state["conv"] else None) or create_conv(message)
                    conv["messages"] += [{"role": "user", "text": message}, {"role": "assistant", "text": answer}]
                    conv["quiz"], conv["updated"] = quiz, time.time()
                    save_conv(conv)
                    yield sse("done", {"text": answer, "quiz": quiz, "conversation": conv["id"]})
                    # Summarising can take seconds, so do it after the reply instead of holding the UI.
                    task = asyncio.create_task(remember(message, answer))
                    background.add(task)
                    task.add_done_callback(background.discard)
                    return
                except Exception as exc:  # noqa: BLE001 - surface any model failure to the UI
                    retryable = "503" in str(exc) or "UNAVAILABLE" in str(exc)
                    if retryable and not streamed and attempt < RETRIES - 1:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    yield sse("error", {"message": friendly_error(exc)})
                    return

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------- library (RAG) ----------

def safe_name(name: str) -> str:
    base = re.sub(r"[^\w.\- ]", "_", Path(name).name).strip()
    if Path(base).suffix.lower() not in rag.SUPPORTED:
        raise HTTPException(400, "Only .txt and .md files are supported")
    return base


def list_library() -> list[dict]:
    rag.LIBRARY_DIR.mkdir(exist_ok=True)
    return [
        {"name": p.name, "size": p.stat().st_size}
        for p in sorted(rag.LIBRARY_DIR.iterdir())
        if p.suffix.lower() in rag.SUPPORTED
    ]


@app.get("/api/library")
async def library():
    return {"files": list_library()}


@app.post("/api/library")
async def upload(file: UploadFile = File(...)):
    name = safe_name(file.filename or "")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "File too large (2 MB max)")
    (rag.LIBRARY_DIR / name).write_bytes(data)
    chunks = await asyncio.to_thread(rag.ingest)
    return {"files": list_library(), "chunks": chunks}


@app.delete("/api/library/{name}")
async def remove(name: str):
    path = rag.LIBRARY_DIR / safe_name(name)
    if not path.exists():
        raise HTTPException(404, "No such file")
    path.unlink()
    chunks = await asyncio.to_thread(rag.ingest)
    return {"files": list_library(), "chunks": chunks}


@app.get("/api/library/{name}")
async def read_file(name: str):
    path = rag.LIBRARY_DIR / safe_name(name)
    if not path.exists():
        raise HTTPException(404, "No such file")
    return {"name": path.name, "text": path.read_text(encoding="utf-8", errors="ignore")}


# ---------- notes, memory, session ----------

@app.get("/api/notes")
async def notes():
    if not NOTES_FILE.exists():
        return {"notes": []}
    out = []
    for line in NOTES_FILE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\[(.*?)\]\s*(.*)", line)
        if m:
            out.append({"topic": m.group(1), "note": m.group(2)})
    return {"notes": out}


@app.get("/api/memory")
async def get_memory():
    return {"summary": memory.load_summary(), "recent_turns": len(memory.load_turns())}


@app.delete("/api/memory")
async def clear_memory():
    async with state["lock"]:
        memory.clear()
        new_session()
    return {"ok": True}


@app.post("/api/session/new")
async def new_chat():
    """Start a fresh conversation; what was said so far is folded into the long-term summary."""
    async with state["lock"]:
        try:
            await memory.fold_into_summary()
        except Exception:  # noqa: BLE001 - a failed summary must not block a new chat
            pass
        state["conv"] = None  # the chat itself is created when the first message is sent
        new_session()
    return {"ok": True}


@app.get("/api/conversations")
async def conversations():
    return {"active": state["conv"],
            "conversations": [{"id": c["id"], "title": c["title"], "updated": c["updated"]} for c in all_convs()]}


@app.post("/api/conversations/{cid}/open")
async def open_conversation(cid: str):
    async with state["lock"]:
        conv = load_conv(cid)
        if not conv:
            raise HTTPException(404, "No such chat")
        state["conv"] = cid
        new_session(conv.get("quiz") or {})
    return {"id": cid, "title": conv["title"], "messages": conv["messages"]}


@app.delete("/api/conversations/{cid}")
async def delete_conversation(cid: str):
    async with state["lock"]:
        conv_path(cid).unlink(missing_ok=True)
        was_active = state["conv"] == cid
        if was_active:
            state["conv"] = None
            new_session()
    return {"ok": True, "was_active": was_active}


@app.get("/api/status")
async def status():
    s = state["session"].state
    return {
        "quiz_correct": s.get("quiz_correct", 0),
        "quiz_total": s.get("quiz_total", 0),
        "files": len(list_library()),
        "notes": len((await notes())["notes"]),
    }


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000)
