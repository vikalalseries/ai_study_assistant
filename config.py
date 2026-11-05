"""Shared settings: .env loading and the Gemini chat client used by every Agent Framework agent."""
import os
import sys
from pathlib import Path

from agent_framework_gemini import GeminiChatClient
from dotenv import load_dotenv

for stream in (sys.stdout, sys.stderr):
    # Windows consoles default to cp1252 and crash on emoji in model replies.
    stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
MODEL = "gemini-3.5-flash-lite"

load_dotenv(ROOT / ".env")
# The Gemini client reads GOOGLE_API_KEY; reuse the key already stored in .env as gemini_api_key.
if os.getenv("gemini_api_key"):
    os.environ.setdefault("GOOGLE_API_KEY", os.environ["gemini_api_key"])


def make_client() -> GeminiChatClient:
    return GeminiChatClient(model=MODEL)
