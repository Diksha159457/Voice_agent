# utils/stt.py — Speech-To-Text module
# Converts an audio file on disk → plain text string.
# Uses faster-whisper (CTranslate2 backend) instead of openai-whisper
# so we avoid pulling in the full PyTorch stack (~1.5 GB).

import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
        )
    return transcription.text
_model = None   # module-level cache — avoids reloading on every request


# The rest of the code in this file is the old openai-whisper implementation, left here for reference.
# The new Groq-based code above is much simpler and faster, but this is here in   case you want to see how it was done with the old library or if you want to switch back for any reason.