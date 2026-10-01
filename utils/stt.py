# utils/stt.py — Speech-to-text via Groq's hosted Whisper.
# The client is created lazily, so importing this module never requires an
# API key (the app can still start, serve the UI and run typed commands).

from __future__ import annotations

import os

from utils.client import _get_client

STT_MODEL = os.environ.get("STT_MODEL", "whisper-large-v3")


def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        transcription = _get_client().audio.transcriptions.create(model=STT_MODEL, file=f)
    return (transcription.text or "").strip()
