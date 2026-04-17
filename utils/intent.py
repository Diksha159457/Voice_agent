# utils/intent.py — Intent Detection module
# Sends the user's text to the Groq cloud LLM and asks it to classify
# what the user wants to do, returning a structured dict we can act on.
#
# Why Groq instead of a local Ollama model?
#   • Groq runs on specialised LPU hardware — responses arrive in < 1 second
#   • No local model download (Ollama llama3 = ~4 GB RAM + disk)
#   • Free tier is generous for development / demo use
#   • Sign up for a free key at https://console.groq.com

import os    # reads environment variables (GROQ_API_KEY)
import json  # parses the LLM's JSON response into a Python dict

from groq import Groq
# Groq: the official Python client for the Groq REST API.
# Wraps HTTP calls so we don't need to write raw requests.

# ── Lazy client initialisation ────────────────────────────────────────────────
# We do NOT create the client at module-load time.
# Previously, `client = Groq(...)` ran the moment Python imported this file.
# If GROQ_API_KEY wasn't set yet it crashed with a GroqError BEFORE the app
# even started — confusing for new users. Now _get_client() is called only
# when an actual API request is made, giving a clear, actionable error message.

_client = None   # cached after first successful creation

def _get_client() -> Groq:
    """
    Return the shared Groq client, creating it on the very first call.
    Raises RuntimeError with clear setup instructions if the key is missing.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")   # read from shell environment or .env
        if not api_key:
            raise RuntimeError(
                "\n\n  GROQ_API_KEY is not set!\n"
                "  Get a free key at https://console.groq.com then run:\n\n"
                "    export GROQ_API_KEY=gsk_...   # Mac / Linux (paste in terminal)\n"
                "    set    GROQ_API_KEY=gsk_...   # Windows CMD\n\n"
                "  Or put  GROQ_API_KEY=gsk_...  in a .env file in the project root.\n"
            )
        _client = Groq(api_key=api_key)   # create and cache — reused on every subsequent call
    return _client


# ── System prompt ─────────────────────────────────────────────────────────────
# Injected as the "system" role in every API call.
# Being very precise here is critical: the LLM must return ONLY valid JSON,
# otherwise json.loads() throws and we fall back to general_chat.
SYSTEM_PROMPT = """
You are a precise intent classifier for a voice assistant.

Given user input, respond with ONLY a raw JSON object — no markdown fences,
no explanation, no preamble. The JSON must have exactly these three keys:

{
  "intent":  one of ["create_file", "write_code", "summarize", "general_chat"],
  "target":  the filename, folder name, topic, or empty string if not applicable,
  "details": any extra context (language, framework, text to summarize, etc.)
}

Examples:
  Input:  "Create a Python file called app.py"
  Output: {"intent":"write_code","target":"app.py","details":"empty Python file"}

  Input:  "Make a folder called reports"
  Output: {"intent":"create_file","target":"reports","details":"directory"}

  Input:  "What is recursion?"
  Output: {"intent":"general_chat","target":"","details":"explain recursion"}
"""


def detect_intent(user_text: str) -> dict:
    """
    Classify the user's input into a structured intent dict.

    Args:
        user_text: transcribed (or typed) user command.

    Returns:
        dict with keys 'intent', 'target', 'details'.
        Falls back to general_chat if LLM returns invalid JSON.
    """
    response = _get_client().chat.completions.create(
        model="llama3-8b-8192",
        # llama3-8b-8192:
        #   8B parameter LLaMA 3, 8192-token context window.
        #   Extremely fast on Groq LPU hardware (~200 ms round-trip).
        #   Free tier: 30 requests/min, 14 400/day.
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},  # classification rules
            {"role": "user",   "content": user_text},       # what the user said
        ],
        temperature=0,    # fully deterministic — we need consistent JSON, not creativity
        max_tokens=200,   # intent JSON is tiny; cap to avoid wasting tokens
    )

    raw = response.choices[0].message.content.strip()
    # choices[0]  → first (only) completion
    # .message.content → the raw text the model produced
    # .strip() → remove any leading/trailing whitespace

    try:
        return json.loads(raw)       # convert JSON string → Python dict
    except json.JSONDecodeError:
        # LLM occasionally wraps JSON in markdown fences or adds prose.
        # Gracefully fall back so the app never crashes on a bad response.
        return {
            "intent":  "general_chat",
            "target":  "",
            "details": user_text,    # pass original text so tools.py can still use it
        }