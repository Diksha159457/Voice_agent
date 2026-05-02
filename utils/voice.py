"""Compatibility helpers for older imports.

Browser recording is handled in ``app.py`` and transcription is implemented by
``utils.stt`` with faster-whisper. This module intentionally avoids desktop
audio packages so cloud deployment does not fail at import time.
"""

from utils.stt import transcribe_audio


def speech_to_text(audio_file_path: str) -> str:
    """Transcribe a local audio file to text."""
    return transcribe_audio(audio_file_path)


def record_audio(*_args, **_kwargs):
    """Desktop mic capture is not available in the deployed web app."""
    raise RuntimeError("Use the browser Record button or Upload Audio instead.")


def speak(text: str) -> str:
    """Return text unchanged; server-side TTS is intentionally not used."""
    return text
