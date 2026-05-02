"""Intent detection for the voice agent.

The app supports a few concrete tools, so common commands are parsed locally
first. Groq is used as a second pass for phrasing that is harder to classify.
"""

import json
import re

from config import MODEL_NAME
from utils.client import _get_client


SUPPORTED_INTENTS = {"create_file", "write_code", "summarize", "general_chat"}


def _first_filename(text: str) -> str:
    """Return the first filename-looking token from text, if present."""
    match = re.search(r"([\w.-]+\.[A-Za-z0-9]{1,8})", text)
    return match.group(1) if match else ""


def _target_after_keyword(text: str) -> str:
    patterns = [
        r"\b(?:called|named|as)\s+['\"]?([^'\"\n]+?)['\"]?(?:\s+with\b|\s+that\b|$)",
        r"\b(?:file|folder|directory)\s+['\"]?([^'\"\n]+?)['\"]?(?:\s+with\b|\s+that\b|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip(" .")
    return ""


def _rule_based_intent(text: str) -> dict | None:
    cleaned = text.strip()
    lowered = cleaned.lower()

    if not cleaned:
        return {"intent": "general_chat", "details": ""}

    if re.search(r"\b(summarize|summary|tl;dr|tldr)\b", lowered):
        details = re.sub(
            r"^\s*(please\s+)?(summarize|summary|tl;dr|tldr)(\s+this)?\s*[:.-]?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        return {"intent": "summarize", "target": "", "details": details or cleaned}

    if re.search(r"\b(folder|directory)\b", lowered) and re.search(
        r"\b(create|make|new|add)\b", lowered
    ):
        target = _target_after_keyword(cleaned) or cleaned.split()[-1]
        return {"intent": "create_file", "target": target, "details": "folder"}

    filename = _first_filename(cleaned) or _target_after_keyword(cleaned)
    asks_for_code = re.search(
        r"\b(write|generate|build|create|make|code|script|program|function|class)\b",
        lowered,
    )
    has_extension = bool(_first_filename(cleaned))

    if filename and has_extension and asks_for_code:
        blank_file = re.search(r"\b(empty|blank|touch)\b", lowered)
        if blank_file:
            return {"intent": "create_file", "target": filename, "details": "file"}
        return {"intent": "write_code", "target": filename, "details": cleaned}

    if filename and re.search(r"\b(create|make|new|add|touch)\b", lowered):
        return {"intent": "create_file", "target": filename, "details": "file"}

    return None


def _normalise_llm_payload(payload: dict, original_text: str) -> dict:
    intent = payload.get("intent", "general_chat")
    if intent not in SUPPORTED_INTENTS:
        intent = "general_chat"

    entities = payload.get("entities") or {}
    target = payload.get("target") or entities.get("target") or entities.get("filename") or ""
    details = payload.get("details") or entities.get("details") or original_text

    return {
        "intent": intent,
        "target": target,
        "details": details,
        "entities": entities,
    }


def detect_intent(text: str) -> dict:
    """Detect user intent and return a dispatcher-friendly dictionary."""
    local = _rule_based_intent(text)
    if local:
        local.setdefault("target", "")
        local.setdefault("details", text)
        local.setdefault("entities", {})
        return local

    try:
        response = _get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify commands for a Flask voice assistant. "
                        "Return ONLY valid JSON with keys intent, target, details, entities. "
                        "Supported intents are create_file, write_code, summarize, general_chat. "
                        "Use create_file only for blank files or folders. "
                        "Use write_code when code should be generated and saved."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=180,
        )
        content = response.choices[0].message.content or "{}"
        return _normalise_llm_payload(json.loads(content), text)

    except Exception as e:
        return {
            "intent": "general_chat",
            "target": "",
            "details": text,
            "entities": {},
            "error": str(e),
        }
