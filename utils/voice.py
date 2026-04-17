import sounddevice as sd              # For recording audio
import scipy.io.wavfile as wav        # To save audio file
import pyttsx3                        # Text-to-speech (offline)
from faster_whisper import WhisperModel  # Speech-to-text

_model = None


def get_model(model_size="base"):
    """Load the Whisper model only when voice transcription is used."""
    global _model
    if _model is None:
        _model = WhisperModel(model_size)
    return _model


def record_audio(filename="input.wav", duration=5, fs=16000):
    """
    Records audio from microphone
    """
    print("🎙️ Listening...")

    # Record audio
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished

    # Save as WAV file
    wav.write(filename, fs, recording)

    print("✅ Recording complete")

    return filename


def speech_to_text(audio_file):
    """
    Converts speech → text using Whisper
    """
    model = get_model()
    segments, _ = model.transcribe(audio_file)

    text = ""
    for segment in segments:
        text += segment.text

    print("🧠 Recognized:", text)

    return text


def speak(text):
    """
    Converts text → speech (speaker output)
    """
    engine = pyttsx3.init()

    engine.setProperty("rate", 180)  # Speed of speech
    engine.setProperty("volume", 1)  # Volume (0 to 1)

    engine.say(text)
    engine.runAndWait()
