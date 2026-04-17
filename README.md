# 🎙️ Voice Agent

A lightweight, Claude-style voice assistant that runs in your browser.  
Type a command **or** upload an audio file — the agent transcribes it, classifies your intent, and takes action.

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
| 🎵 Audio upload | Supports `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.webm` |
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

### Upload audio
1. Click **Upload audio** (microphone icon)
2. Pick any audio file (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.webm`)
3. Press **Send** — the app transcribes it and runs the command

The status bar shows **"Heard: …"** so you can verify what Whisper transcribed.

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