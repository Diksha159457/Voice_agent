# 🎙️ Voice Agent

A lightweight, browser-based voice assistant with a Claude-style UI.
You can type a prompt, record from your microphone, upload an audio clip, or attach a text/document file for analysis.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)
![Whisper](https://img.shields.io/badge/Whisper-faster--whisper-green)
![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA3-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🗣️ Speech-to-Text | `faster-whisper` (tiny model, int8) — no PyTorch, ~200 MB RAM |
| 🧠 Intent detection | Groq LLaMA3-8B — classifies commands in < 1 second |
| 🛠️ Actions | Create files/folders, generate code, summarize text, answer questions |
| 💬 Claude-style UI | Dark theme, message bubbles, sidebar history, suggestion chips |
| 🎙️ Mic recording | Record directly in the browser with replay before sending |
| 🎵 Audio upload | Supports `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.webm` |
| 📄 File upload | Supports text/code files plus `.pdf` and `.docx` |
| ⚡ Lightweight | ~300 MB total vs 6 GB+ for the PyTorch + Ollama stack |

---

## 📁 Project Structure

```
voice_agent/
├── app.py               ← Flask server + Claude-style HTML UI
├── requirements.txt     ← All Python dependencies
├── .env.example         ← Copy to .env and add your Groq key
├── .gitignore
└── utils/
    ├── stt.py           ← Audio → text  (faster-whisper)
    ├── intent.py        ← Text → intent dict  (Groq LLaMA3)
    ├── client.py        ← Shared Groq client helper
    ├── chat.py          ← Conversational reply + streaming chat
    ├── voice.py         ← Local mic/speech helper functions
    ├── tools.py         ← Intent → action  (create file, write code, etc.)
    └── memory.py        ← In-session conversation history
```

---

## 🚀 Quick Start

### 1 — Clone the repo

```bash
git clone https://github.com/Diksha159457/Voice_agent.git
cd Voice_agent
```

### 2 — Create and activate a virtual environment

```bash
# Mac / Linux
python3 -m venv venv
source venv/bin/activate

# Windows CMD
python -m venv venv
venv\Scripts\activate
```

### 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4 — Install ffmpeg (required for audio decoding)

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# Windows (run as Administrator)
winget install ffmpeg
# OR
choco install ffmpeg
```

### 5 — Get a free Groq API key

1. Go to **https://console.groq.com** and sign up (free)
2. Click **API Keys → Create API Key**
3. Copy the key (starts with `gsk_`)

### 6 — Set your API key

**Option A — `.env` file (recommended, persists across sessions):**

```bash
cp .env.example .env
# open .env in any editor and replace the placeholder:
# GROQ_API_KEY=gsk_your_actual_key_here
```

**Option B — export in your terminal (one session only):**

```bash
# Mac / Linux
export GROQ_API_KEY=gsk_your_actual_key_here

# Windows CMD
set GROQ_API_KEY=gsk_your_actual_key_here

# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_actual_key_here"
```

### 7 — Run the app

```bash
python app.py
```

Open **http://localhost:8501** in your browser.

Note: the app now starts quietly. `python app.py` may show no banner text, which is expected.

---

## 🎯 How to Use

### Type a command
Click the input box at the bottom and type naturally:

| What you type | What happens |
|---|---|
| `Create a Python file called calculator.py` | Generates and saves `output/calculator.py` |
| `Make a folder called my_project` | Creates `output/my_project/` directory |
| `Summarize this: <your text here>` | Returns a 2–4 sentence summary |
| `What is recursion?` | Answers conversationally |

Press **Enter** to send (or **Shift + Enter** for a newline).

### Record audio
1. Click **Record audio**
2. Allow microphone access in your browser
3. Click the button again to stop recording
4. Use the built-in audio player to listen back
5. Press **Send** to transcribe and process the recording

### Upload audio
1. Click **Upload audio**
2. Pick any audio file (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.webm`)
3. Press **Send** — the app transcribes it and runs the command

The status bar shows **"Heard: …"** so you can verify what Whisper transcribed.

### Upload a file
1. Click **Upload file**
2. Choose a supported file:
   `.txt`, `.md`, `.py`, `.js`, `.ts`, `.tsx`, `.json`, `.csv`, `.html`, `.css`, `.java`, `.c`, `.cpp`, `.sql`, `.xml`, `.yaml`, `.yml`, `.pdf`, `.docx`
3. Optionally add a typed note in the text box
4. Press **Send**

Large files are trimmed before being sent to the LLM so the request stays within model limits.

### Sidebar
- The **Recent** panel on the left shows your conversation history
- Click **Clear history** to wipe it and start fresh

---

## ⚙️ Configuration

All generated files are saved to the `output/` folder (created automatically).

To use a more accurate (but slower) Whisper model, change `model_size` in `utils/stt.py`:

```python
# utils/stt.py  line ~32
def get_model(model_size: str = "tiny"):   # change "tiny" to "base" or "small"
```

| Model | Disk | RAM | Speed |
|---|---|---|---|
| `tiny`  | 39 MB  | ~200 MB | Fastest |
| `base`  | 74 MB  | ~290 MB | Balanced |
| `small` | 244 MB | ~500 MB | Most accurate |

---

## 🐛 Troubleshooting

### `GroqError: The api_key client option must be set`
Your `GROQ_API_KEY` environment variable is not set. Follow **Step 6** above.  
Using a `.env` file is the easiest fix — the app loads it automatically.

### `Transcription failed` / audio upload not working
- Make sure **ffmpeg is installed** (`ffmpeg -version` in your terminal should print a version)
- The audio file must be a supported format: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.webm`
- File size limit is 50 MB

### PDF / DOCX upload issues
- PDF extraction works best for text-based PDFs; scanned-image PDFs may need OCR
- DOCX extraction reads document text, but highly custom layouts may lose some formatting
- Very large files are intentionally truncated before they are sent to the model

### App seems to “hang” after `python app.py`
- The startup banner was intentionally removed, so no console output is expected
- Open **http://localhost:8501** manually in your browser
- If needed, confirm the server is listening with `lsof -i :8501`

### Port already in use
```bash
# Find and kill whatever is using port 8501
lsof -i :8501          # Mac / Linux
netstat -ano | findstr 8501   # Windows
```
Or change the port in the last line of `app.py`: `app.run(debug=False, port=8502)`

### `ModuleNotFoundError`
Make sure your virtual environment is activated (`source venv/bin/activate`) and you ran `pip install -r requirements.txt`.

---

## 📦 Dependencies

| Package | Purpose | Size |
|---|---|---|
| `flask` | Web server | ~5 MB |
| `faster-whisper` | Speech-to-text | ~80 MB + model |
| `groq` | Groq API client | ~2 MB |
| `numpy` | Used by faster-whisper | ~20 MB |
| `python-dotenv` | Loads `.env` file | <1 MB |
| **ffmpeg** (OS package) | Audio format decoding | ~80 MB |

**Total: ~300 MB** vs the original ~6 GB stack.

---

## 🔒 Security Notes

- The `.env` file containing your API key is in `.gitignore` — it will never be committed
- All files the agent creates are written to the `output/` folder only (path-traversal protected)
- The app is intended for local / single-user use; do not expose it to the public internet without adding authentication

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
