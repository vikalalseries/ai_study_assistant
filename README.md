# StudyBuddy

A study assistant that answers from **your own notes**. It is built on the [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) and uses Google **Gemini** for both chat and embeddings. Notes are indexed in **ChromaDB** and can optionally be stored durably in **Vercel Blob**.

You can use it from the terminal or through a web UI.

## Features

- **Answers from your notes (RAG):** it searches your notes first and cites the source file. If nothing relevant turns up, it says so before giving a general answer.
- **Calculator tool:** arithmetic goes through a safe expression evaluator, never "in the model's head".
- **Save notes:** ask it to remember something and it appends the note to your notes file.
- **Quizzes:** it asks one question at a time from your material, judges each answer and keeps a running score.
- **Long-term memory:** recent turns are saved to disk and older ones are rolled into a summary, so it remembers you between runs.
- **Web UI:** upload `.txt`, `.md` and `.pdf` files, type notes in the browser, chat, view the tools the agent used, and export your data.

## How it works

```
library/*.md|txt ──chunk──▶ Gemini embeddings ──▶ ChromaDB
                                                      │
question ──▶ Agent (Gemini) ──tool call──▶ search_study_material
                  │
                  ├─ calculate          (safe arithmetic)
                  ├─ save_note          (append to notes file)
                  └─ track_quiz_score   (per-session score)
```

## Project layout

| File | Purpose |
| --- | --- |
| [main.py](main.py) | Terminal chat entry point (`python main.py`, `python main.py ingest`) |
| [server.py](server.py) | FastAPI web server and REST API |
| [agent.py](agent.py) | Agent definition, instructions and the four tools |
| [rag.py](rag.py) | Chunking, Gemini embeddings, ChromaDB indexing and search |
| [ingest.py](ingest.py) | Rebuilds the vector index from `library/` |
| [memory.py](memory.py) | Saved turns plus rolling summary |
| [storage.py](storage.py) | Vercel Blob client (REST, via `httpx`) |
| [config.py](config.py) | `.env` loading and the Gemini chat client |
| [static/index.html](static/index.html) | Web front end |
| [library/](library) | Sample notes |
| [STUDYBUDDY_NOTES.md](STUDYBUDDY_NOTES.md) | Line-by-line walkthrough of the code |

## Getting started

### Prerequisites

- Python 3.10+
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

### Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```env
gemini_api_key=your-gemini-api-key
# Optional: store notes in Vercel Blob instead of only on local disk
BLOB_READ_WRITE_TOKEN=your-blob-token
```

`GOOGLE_API_KEY` also works in place of `gemini_api_key`.

### Run

Index your notes (do this again whenever `library/` changes):

```bash
python main.py ingest
```

**Terminal chat.** Type `clear` to wipe saved memory and `exit` to quit.

```bash
python main.py
```

**Web UI.** Then open <http://127.0.0.1:8000>.

```bash
python server.py
```

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/api/state` | Library, notes and quiz state |
| GET | `/api/history` | Recent unsummarised turns |
| POST | `/api/chat` | Send a message to the agent |
| POST | `/api/memory/clear` | Wipe saved memory |
| POST | `/api/memory/summarize` | Fold current turns into the summary |
| POST | `/api/quiz/reset` | Reset the quiz score |
| POST | `/api/library/upload` | Upload `.txt`, `.md` or `.pdf` files |
| POST | `/api/library/text` | Save a typed note as a library file |
| GET / DELETE | `/api/library/{name}` | Read or delete a library file |
| POST | `/api/ingest` | Rebuild the vector index |
| GET | `/api/export` | Export your data |

## Deploying to Vercel

The repo includes a [vercel.json](vercel.json) that bundles `static/` and excludes heavy, unneeded packages to keep the function small.

1. Import the repo into Vercel.
2. Set `gemini_api_key` and `BLOB_READ_WRITE_TOKEN` in the project's environment variables.
3. Deploy.

Vercel's filesystem is read-only apart from `/tmp`, which is temporary. On Vercel the app therefore writes to `/tmp` and relies on Blob for durable notes. It re-indexes automatically when the Blob contents change, for example after a cold start. Without a Blob token, uploaded notes will not survive a restart.

## Tech stack

Python, FastAPI, Uvicorn, Microsoft Agent Framework, Google Gemini (`gemini-3.5-flash-lite` for chat, `gemini-embedding-001` for embeddings), ChromaDB, pypdf, httpx.

## License

Apache License 2.0. See [LICENSE](LICENSE).
