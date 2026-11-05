"""StudyBuddy entry point.

    python main.py          chat with the study agent (remembers you between runs)
    python main.py ingest   rebuild the notes database from library/

Type 'clear' while chatting to wipe the saved memory, 'exit' to quit.
"""
import asyncio
import sys

try:
    from . import memory
    from .agent import build_agent
    from .ingest import ingest_library
except ImportError:  # run as a plain script from inside the folder
    import memory
    from agent import build_agent
    from ingest import ingest_library


def start_session(quiz_state: dict | None = None):
    """A fresh agent whose instructions carry the saved memory, plus a session that keeps the quiz score."""
    agent = build_agent(memory.context())
    session = agent.create_session()
    session.state.update(quiz_state or {})
    return agent, session


async def run_study_agent() -> None:
    agent, session = start_session()
    print("StudyBuddy ready. Type 'clear' to forget everything, 'exit' to quit.")
    try:
        while True:
            question = (await asyncio.to_thread(input, "You: ")).strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            if question.lower() == "clear":
                memory.clear()
                agent, session = start_session()
                print("Memory cleared.")
                continue
            answer = (await agent.run(question, session=session)).text
            print(f"StudyBuddy: {answer}\n")
            if await memory.record_turn(question, answer):
                # Old turns now live in the summary: restart the session so the history stays short.
                agent, session = start_session(dict(session.state))
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        # Session is over: fold the remaining turns into the summary of the whole chat.
        if await memory.fold_into_summary():
            print(f"Memory saved to {memory.SUMMARY_FILE.name}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "study"
    if mode == "ingest":
        ingest_library()
    elif mode == "study":
        asyncio.run(run_study_agent())
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
