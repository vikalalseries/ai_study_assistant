"""Long-term memory for the study agent: recent turns on disk plus a rolling summary of older ones.

The first FIRST_SUMMARY_TURNS turns become the first summary. After that, every NEXT_SUMMARY_TURNS new turns
are merged into the existing summary. On exit, whatever is left is merged too, so the summary covers the whole chat.
"""
import json
from collections.abc import Sequence

from agent_framework import Agent

try:
    from .config import ROOT, make_client
except ImportError:  # run as a plain script from inside the folder
    from config import ROOT, make_client

MEMORY_FILE = ROOT / "chat_memory.json"
SUMMARY_FILE = ROOT / "chat_summary.txt"
FIRST_SUMMARY_TURNS = 10
NEXT_SUMMARY_TURNS = 9

_summarizer: Agent | None = None


def load_turns() -> list[tuple[str, str]]:
    """Saved turns that are not yet part of the summary."""
    if not MEMORY_FILE.exists():
        return []
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        return [(turn["user"], turn["assistant"]) for turn in data]
    except (json.JSONDecodeError, KeyError, TypeError):
        # A corrupted memory file shouldn't stop the agent from starting.
        return []


def save_turns(turns: Sequence[tuple[str, str]]) -> None:
    data = [{"user": u, "assistant": a} for u, a in turns]
    MEMORY_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_summary() -> str:
    return SUMMARY_FILE.read_text(encoding="utf-8") if SUMMARY_FILE.exists() else ""


def clear() -> None:
    MEMORY_FILE.unlink(missing_ok=True)
    SUMMARY_FILE.unlink(missing_ok=True)


def _format(turns: Sequence[tuple[str, str]]) -> str:
    return "\n".join(f"Student: {u}\nStudyBuddy: {a}" for u, a in turns)


def context() -> str:
    """Memory text to give the agent at the start of a session ('' when there is none)."""
    parts = []
    if summary := load_summary():
        parts.append(f"Summary of earlier conversations:\n{summary}")
    if turns := load_turns():
        parts.append(f"Most recent turns:\n{_format(turns)}")
    return "\n\n".join(parts)


async def fold_into_summary() -> bool:
    """Merge the saved turns into the summary and empty the turn file. False if there was nothing to merge."""
    global _summarizer
    turns = load_turns()
    if not turns:
        return False
    if _summarizer is None:
        _summarizer = Agent(make_client(), "You summarize study conversations concisely.", name="summarizer")

    previous = load_summary()
    prompt = (
        "Update the running summary of this study chat so it covers the topics studied, questions asked, "
        "quiz results and anything the student shared about themselves, from both the earlier summary and "
        "the new turns. Keep it concise.\n\n"
        + (f"Summary of earlier conversation:\n{previous}\n\n" if previous else "")
        + f"New turns:\n{_format(turns)}"
    )
    SUMMARY_FILE.write_text((await _summarizer.run(prompt)).text, encoding="utf-8")
    save_turns([])
    return True


async def record_turn(question: str, answer: str) -> bool:
    """Save a turn and roll the summary when due. True if the summary was just updated."""
    turns = [*load_turns(), (question, answer)]
    save_turns(turns)
    threshold = NEXT_SUMMARY_TURNS if SUMMARY_FILE.exists() else FIRST_SUMMARY_TURNS
    return len(turns) >= threshold and await fold_into_summary()
