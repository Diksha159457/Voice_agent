# utils/stt.py — Speech-To-Text module
# Converts an audio file on disk → plain text string.
# Uses faster-whisper (CTranslate2 backend) instead of openai-whisper
# so we avoid pulling in the full PyTorch stack (~1.5 GB).

from faster_whisper import WhisperModel
# WhisperModel wraps CTranslate2 inference around Whisper weights.

_model = None   # module-level cache — avoids reloading on every request


def get_model(model_size: str = "tiny") -> WhisperModel:
    """
    Load the Whisper model once and reuse it on every subsequent call.

    model_size options (speed vs accuracy trade-off):
        'tiny'  — 39 MB,  fastest,  good for clear speech
        'base'  — 74 MB,  balanced
        'small' — 244 MB, best accuracy, slower
    """
    global _model
    if _model is None:               # only load on the first call
        _model = WhisperModel(
            model_size,
            device="cpu",            # no GPU required
            compute_type="int8",     # 8-bit quantisation: ~4× less RAM than float32
        )
    return _model


def transcribe_audio(audio_path: str, model_size: str = "tiny") -> str:
    """
    Transcribe a local audio file to text.

    Args:
        audio_path:  path to the audio file (wav, mp3, m4a, ogg, etc.)
                     ffmpeg must be installed on the OS for non-wav formats.
        model_size:  passed to get_model(); default 'tiny'

    Returns:
        All spoken words joined into one string. Empty string if silent.
    """
    model = get_model(model_size)

    segments, _info = model.transcribe(
        audio_path,   # file to process
        beam_size=1,  # greedy decoding — fastest; raise to 5 for better accuracy
    )
    # segments is a lazy generator — we must iterate it to get the text
    return " ".join(seg.text for seg in segments).strip()