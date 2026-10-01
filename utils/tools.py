# utils/tools.py — Tool Executor module
# Reads the intent dict from intent.py and performs the correct action.
# All generated files go into output/ — the agent can never write outside it.

import os
import re

from config import MODEL_NAME
from utils.client import _get_client, has_api_key
from utils.sandbox import SandboxError, safe_path

OUTPUT_DIR = os.environ.get("AGENT_OUTPUT_DIR", "output")
# All files the agent creates go here — see utils/sandbox.py for the rules.

_FENCE = re.compile(r"^```[\w+-]*\s*\n(.*?)\n?```\s*$", re.DOTALL)


def strip_code_fences(text: str) -> str:
    """Models often wrap code in ``` fences despite instructions; remove them."""
    match = _FENCE.match(text.strip())
    return match.group(1) if match else text.strip()


# ── Tool: create an empty file or directory ───────────────────────────────────
def create_file(intent_data: dict) -> str:
    """Create a blank file or folder inside output/."""
    target  = intent_data.get("target", "")    # e.g. "my_project" or "notes.txt"
    details = str(intent_data.get("details", ""))   # e.g. "directory"

    try:
        path = safe_path(OUTPUT_DIR, target)
    except SandboxError as e:
        return f"⚠️ {e}"
    if path.exists() and path.is_dir() != ("dir" in details.lower() or "folder" in details.lower()):
        return f"⚠️ '{path.name}' already exists as a different type."

    if "dir" in details.lower() or "folder" in details.lower():
        path.mkdir(exist_ok=True)
        return f"✅ Created folder: {OUTPUT_DIR}/{path.name}"
    else:
        with open(path, "a"):              # 'a' mode: creates if absent, no-op if present
            pass
        return f"✅ Created file: {OUTPUT_DIR}/{path.name}"


# ── Tool: generate code with the LLM and save it ─────────────────────────────
def write_code(intent_data: dict) -> str:
    """Ask Groq LLaMA3 to write code and save it to output/<target>."""
    if not has_api_key():
        return "Groq API key is missing. Add GROQ_API_KEY to your environment to generate code."

    target  = intent_data.get("target") or "output.py"   # filename to save to
    details = intent_data.get("details", "")            # what the code should do

    try:
        path = safe_path(OUTPUT_DIR, target)
    except SandboxError as e:
        return f"⚠️ {e}"
    if path.is_dir():
        return f"⚠️ '{path.name}' is a folder; choose a file name."

    prompt = (
        f"Write complete, working code for a file named '{target}'. "
        f"{details}. "
        "Return ONLY the code — no markdown fences, no explanations."
    )

    response = _get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,   # low but not zero: allows natural code style variation
        max_tokens=1000,   # enough for a reasonably-sized source file
    )

    code = strip_code_fences(response.choices[0].message.content or "")

    with open(path, "w", encoding="utf-8") as f:
        f.write(code)   # save to disk

    preview = "\n".join(code.splitlines()[:3])   # first 3 lines for the bubble preview
    return f"✅ Wrote code to {OUTPUT_DIR}/{path.name}:\n\n{preview}\n..."


# ── Tool: summarize text ──────────────────────────────────────────────────────
def summarize(intent_data: dict) -> str:
    """Summarize the text in intent_data['details'] into 2-4 sentences."""
    if not has_api_key():
        return "Groq API key is missing. Add GROQ_API_KEY to your environment to summarize text."

    details = intent_data.get("details", "")

    if not details:
        return "Please provide the text you'd like me to summarize."

    prompt = f"Summarize the following in 2-4 concise sentences:\n\n{details}"

    response = _get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,   # slight warmth for readable prose
        max_tokens=300,
    )

    summary = response.choices[0].message.content.strip()
    return f"📝 Summary:\n\n{summary}"


# ── Tool: general conversational reply ───────────────────────────────────────
def general_chat(intent_data: dict) -> str:
    """Answer any question that doesn't fit a specific tool."""
    if not has_api_key():
        return "Groq API key is missing. Add GROQ_API_KEY to your environment or .env file."

    details = intent_data.get("details", "")
    target  = intent_data.get("target", "")
    question = details or target or "Hello"   # reconstruct the original question

    response = _get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful, concise voice assistant. Keep answers under 150 words.",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.7,   # higher creativity for natural conversation
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()

# ── Dispatch table: intent string → tool function ─────────────────────────────
TOOL_MAP = {
    "create_file":  create_file,    # "make a folder …" / "create a file …"
    "write_code":   write_code,     # "write a Python script that …"
    "summarize":    summarize,      # "summarize this: …"
    "general_chat": general_chat,   # "what is …" / everything else
}


def execute_tool(intent_data: dict) -> str:
    """
    Route intent_data to the correct tool and return its result string.
    Falls back to general_chat for any unrecognised intent.
    """
    intent  = intent_data.get("intent", "general_chat")
    DEFAULT_TOOL = general_chat
    tool_fn = TOOL_MAP.get(intent, DEFAULT_TOOL)  # safe fallback

    try:
        return tool_fn(intent_data)
    except Exception as e:
        return f"⚠️ Error running '{intent}': {str(e)}"
