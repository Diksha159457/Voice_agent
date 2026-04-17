# utils/tools.py — Tool Executor module
# Reads the intent dict from intent.py and performs the correct action.
# All generated files go into output/ — the agent can never write outside it.

import os       # path operations and directory creation

from config import MODEL_NAME
from utils.client import _get_client

OUTPUT_DIR = "output"
# All files the agent creates go here.
# One safe folder = easy cleanup, no risk of overwriting system files.

# ── Helper: create output/ if it doesn't exist ───────────────────────────────
def _ensure_output_dir() -> None:
    """Create output/ directory. exist_ok=True = no error if already there."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helper: build a safe path inside output/ ─────────────────────────────────
def _safe_path(filename: str) -> str:
    """
    Strips directory components from filename to block path-traversal attacks.
    e.g. '../../etc/passwd' → 'output/passwd'  (harmless)
    """
    safe_name = os.path.basename(filename) if filename else "output_file"
    return os.path.join(OUTPUT_DIR, safe_name)


# ── Tool: create an empty file or directory ───────────────────────────────────
def create_file(intent_data: dict) -> str:
    """Create a blank file or folder inside output/."""
    _ensure_output_dir()

    target  = intent_data.get("target", "")    # e.g. "my_project" or "notes.txt"
    details = intent_data.get("details", "")   # e.g. "directory"

    if not target:
        return "Please specify a file or folder name."

    path = _safe_path(target)

    if "dir" in details.lower() or "folder" in details.lower():
        os.makedirs(path, exist_ok=True)   # create directory (and any parents)
        return f"✅ Created folder: {path}"
    else:
        with open(path, "a"):              # 'a' mode: creates if absent, no-op if present
            pass
        return f"✅ Created file: {path}"


# ── Tool: generate code with the LLM and save it ─────────────────────────────
def write_code(intent_data: dict) -> str:
    """Ask Groq LLaMA3 to write code and save it to output/<target>."""
    _ensure_output_dir()

    target  = intent_data.get("target", "output.py")   # filename to save to
    details = intent_data.get("details", "")            # what the code should do

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

    code = response.choices[0].message.content.strip()   # raw generated code

    path = _safe_path(target)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)   # save to disk

    preview = "\n".join(code.splitlines()[:3])   # first 3 lines for the bubble preview
    return f"✅ Wrote code to {path}:\n\n{preview}\n..."


# ── Tool: summarize text ──────────────────────────────────────────────────────
def summarize(intent_data: dict) -> str:
    """Summarize the text in intent_data['details'] into 2-4 sentences."""
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

def streaming_chat(text):
    """
    Streaming version of general chat (prints live in terminal)
    """

    client = _get_client()  # get Groq client

    response = client.chat.completions.create(
        model=MODEL_NAME,  # use updated model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text}
        ],
        stream=True  # 🔥 enables streaming
    )

    full_response = ""  # store final response

    for chunk in response:
        delta = chunk.choices[0].delta.content or ""  # extract token

        print(delta, end="", flush=True)  # print live in terminal

        full_response += delta  # build full response

    print()  # newline after completion

    return full_response.strip()  # return complete response
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
