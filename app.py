# app.py — Voice Agent web server.
# Browser UI (templates/index.html) → Flask routes → STT → intent → tool → history.

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

try:
    from dotenv import load_dotenv

    load_dotenv()  # reads GROQ_API_KEY from .env before any client is created
except ImportError:  # pragma: no cover
    pass

from utils.client import has_api_key
from utils.documents import UnsupportedDocument, extract_text
from utils.history import HistoryStore
from utils.intent import detect_intent
from utils.memory import add_to_memory, clear_memory
from utils.stt import transcribe_audio
from utils.tools import execute_tool

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", 25)) * 1024 * 1024

MAX_FILE_CHARS = 12_000  # keeps the LLM request within the model's context window
MAX_TEXT_CHARS = 12_000  # folder uploads send up to 12k chars of combined text
HISTORY_FILE = os.environ.get("HISTORY_FILE", "chat_history.json")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".mp4", ".mpeg", ".mpga"}


def _history() -> HistoryStore:
    return HistoryStore(HISTORY_FILE)


def _error(message: str, status: int):
    return jsonify({"error": message}), status


@app.errorhandler(413)
def too_large(_):
    return _error(f"Upload too large (limit {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB).", 413)


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "llm_configured": has_api_key()})


@app.post("/run_audio")
def run_audio():
    """Audio (recorded blob or uploaded file) → Whisper transcript → text pipeline."""
    audio = request.files.get("audio")
    if not audio:
        return _error("No audio file received.", 400)
    if not has_api_key():
        return _error("Transcription needs GROQ_API_KEY. Type your request instead.", 503)

    ext = Path(audio.filename or "upload.wav").suffix.lower() or ".wav"
    if ext not in AUDIO_EXTENSIONS:
        return _error(f"Unsupported audio format: {ext}", 415)

    note = request.form.get("note", "").strip()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            audio.save(tmp.name)
            tmp_path = tmp.name
        transcript = transcribe_audio(tmp_path)
    except Exception as exc:  # provider/network error
        app.logger.exception("transcription failed")
        return _error(f"Transcription failed: {exc}", 502)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not transcript.strip():
        return _error("No speech detected. Speak more clearly and try again.", 400)

    request_text = f"{note}\n\nSpoken: {transcript}" if note else transcript
    result = _process_text(request_text)
    result["transcript"] = transcript
    return jsonify(result)


@app.post("/run_file")
def run_file():
    """Document upload (PDF, DOCX, text/code) → extracted text → text pipeline."""
    uploaded = request.files.get("file")
    if not uploaded:
        return _error("No file received.", 400)

    filename = Path(uploaded.filename or "file.txt").name
    data = uploaded.read()
    if not data:
        return _error("The uploaded file is empty.", 400)

    try:
        contents = extract_text(filename, data)
    except UnsupportedDocument as exc:
        return _error(str(exc), 415)
    except Exception as exc:
        return _error(f"Could not read file: {exc}", 400)

    if not contents:
        return _error("File contained no readable text.", 400)

    truncated = len(contents) > MAX_FILE_CHARS
    if truncated:
        contents = contents[:MAX_FILE_CHARS] + "\n\n[Truncated — only the first portion was sent to the model.]"

    note = request.form.get("note", "").strip()
    body = f"File: {filename}\n\n{contents}"
    result = _process_text(f"{note}\n\n{body}" if note else body)
    result["file_truncated"] = truncated
    return jsonify(result)


@app.post("/run_text")
def run_text():
    """``{"text": "…"}`` → intent → tool."""
    body = request.get_json(silent=True)
    text = body.get("text") if isinstance(body, dict) else None
    if not isinstance(text, str) or not text.strip():
        return _error("No text provided.", 400)
    if len(text) > MAX_TEXT_CHARS:
        return _error(f"Text too long (max {MAX_TEXT_CHARS} characters).", 413)
    return jsonify(_process_text(text.strip()))


@app.get("/history")
def history():
    return jsonify(_history().load())


@app.post("/clear_history")
def clear_history_route():
    clear_memory()
    _history().clear()
    return jsonify({"status": "cleared"})


# ── Shared pipeline ──────────────────────────────────────────────────────────


def _process_text(text: str) -> dict:
    """text → intent → tool → memory + disk → response."""
    intent_data = detect_intent(text)
    result_text = execute_tool(intent_data)

    entry = {
        "text": text,
        "intent": intent_data.get("intent", "general_chat"),
        "result": result_text,
        "timestamp": datetime.now().isoformat(),
    }
    add_to_memory(entry)
    _history().append(entry)
    return {"intent": entry["intent"], "result": result_text}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    print(f"\n✅ Voice Agent running → http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
