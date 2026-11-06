import ast
import operator

from agent_framework import Agent, FunctionInvocationContext, tool

try:
    from . import rag
    from .config import ROOT, make_client
except ImportError:  # run as a plain script from inside the folder
    import rag
    from config import ROOT, make_client

NOTES_FILE = ROOT / "study_notes.txt"

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / ( ) are allowed")


@tool
def calculate(expression: str) -> dict:
    """Safely evaluate an arithmetic expression such as "12 * 4 + 7".

    Args:
        expression: Numbers combined with + - * / and parentheses only.

    Returns:
        A dict with status "success" and the numeric result, or status "error" and a message.
    """
    try:
        return {"status": "success", "result": _eval(ast.parse(expression.strip(), mode="eval").body)}
    except ZeroDivisionError:
        return {"status": "error", "message": "division by zero"}
    except (ValueError, SyntaxError):
        return {"status": "error", "message": "only numbers and + - * / ( ) are allowed"}


@tool
def save_note(topic: str, note: str) -> dict:
    """Append a note to the student's local notes file.

    Args:
        topic: Short topic label for the note, e.g. "biology".
        note: The text of the note to remember.

    Returns:
        A dict confirming the note was saved.
    """
    with NOTES_FILE.open("a", encoding="utf-8") as f:
        f.write(f"[{topic.strip()}] {note.strip()}\n")
    return {"status": "saved", "topic": topic, "file": NOTES_FILE.name}


@tool
def track_quiz_score(was_correct: bool, ctx: FunctionInvocationContext) -> dict:
    """Update the running quiz score for this session after the student answers a quiz question.

    Args:
        was_correct: True if the student's last answer was correct.

    Returns:
        A dict with correct, total and a readable score string.
    """
    state = ctx.session.state
    correct = state.get("quiz_correct", 0) + int(was_correct)
    total = state.get("quiz_total", 0) + 1
    state["quiz_correct"] = correct
    state["quiz_total"] = total
    return {"correct": correct, "total": total, "score": f"{correct}/{total}"}


@tool
def search_study_material(question: str) -> dict:
    """Search the student's own notes for passages related to a question.

    Args:
        question: The student's question or topic.

    Returns:
        A dict with matching passages (each with its source file), or status "no_results".
    """
    hits = rag.search(question)
    if not hits:
        return {"status": "no_results", "message": "Nothing found. The notes library may be empty or not ingested."}
    return {"status": "success", "passages": [{"source": s, "text": t} for s, t in hits]}


INSTRUCTIONS = """You are StudyBuddy, a friendly, concise study assistant. StudyBuddy is its own product with its own name
and personality; always introduce and refer to yourself simply as "StudyBuddy".

Identity rules:
- Never mention Google, Gemini, any other AI company, model name, or API unless the student's question is only
  about those things in general (not about you).
- If asked who made you, what model or technology powers you, or whether you use another company's API, say you are
  StudyBuddy, an AI study assistant, and that you don't share details about the technology behind you. Keep it short,
  then offer to help with studying.
- Always be honest that you are an AI. Never claim to be human, and never state specific false facts about how you
  were built (for example, do not say you were trained from scratch or that no other technology is involved).

- For ANY arithmetic, call calculate. Never compute in your head. If it returns an error, tell the student plainly.
- Before answering anything that might be covered by the student's notes, call search_study_material first.
  Answer from the returned passages and name the source file. If nothing relevant comes back, say so plainly
  ("I could not find that in your notes") and only then offer a general answer, clearly labelled as not from their notes.
- When asked to remember or save something, call save_note with a short topic and the note, then confirm.
- When asked to quiz the student: use search_study_material to get material, ask ONE question at a time, and wait
  for the answer. After each answer, judge it against the notes, call track_quiz_score, then share the running score
  and ask the next question.
- Otherwise just chat naturally."""


MEMORY_NOTE = """

Your memory of this student from earlier conversations is below. Use it to remember who they are and what they
have studied, but it is only a summary: for facts from their notes, still call search_study_material.

{memory}"""


def build_agent(memory: str = "") -> Agent:
    return Agent(
        make_client(),
        INSTRUCTIONS + (MEMORY_NOTE.format(memory=memory) if memory else ""),
        name="study_buddy",
        description="A study assistant that answers from the student's own notes, does maths, saves notes and quizzes.",
        tools=[calculate, save_note, track_quiz_score, search_study_material],
    )
