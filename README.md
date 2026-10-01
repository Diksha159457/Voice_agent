# 🎙️ Voice Agent

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=flat&logo=render)](https://voice-agent-8085.onrender.com)
[![CI](https://github.com/Diksha159457/Voice_agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Diksha159457/Voice_agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3-orange?logo=groq)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A voice-controlled AI agent that transcribes speech, detects intent, and executes actions — all from a clean browser UI. Built with Flask, Groq Whisper, and LLaMA3.

> 🚀 **[Try the live demo →](https://voice-agent-8085.onrender.com)**

---

## Demo

<!-- TODO: add assets/demo.gif (record with QuickTime → convert at ezgif.com) and uncomment:
![Voice Agent Demo](assets/demo.gif) -->
> 🎥 Try it live: **[voice-agent-8085.onrender.com](https://voice-agent-8085.onrender.com)**

---

## What it does

Speak or type a command, and the agent figures out what you want and does it:

| Command | Intent detected | What happens |
|---|---|---|
| "Create a Python file called calculator.py" | `write_code` | Generates code, saves to `output/` |
| "Make a new folder called my_project" | `create_file` | Creates `output/my_project/` |
| "Summarize this: Python is a high-level language..." | `summarize` | Returns a concise summary |
| "What is machine learning?" | `general_chat` | LLM answers conversationally |

---

## How it works

```
Audio / Text input
       │
       ▼
  [STT] Groq Whisper API → transcript
       │
       ▼
  [Intent] LLaMA3 via Groq → { intent, target, details }
       │
       ▼
  [Tools] create_file / write_code / summarize / general_chat
       │
       ▼
  [Memory] session history → Flask backend → browser UI
```

1. **Input** — user types text or uploads an audio file (.wav, .mp3, .m4a)
2. **Transcription** — Groq Whisper API converts speech to text
3. **Intent detection** — LLaMA3 classifies the request into one of four actions
4. **Execution** — the right tool runs and returns a result
5. **Memory** — each interaction is saved to session history and shown in the sidebar

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, Flask, Gunicorn |
| STT | Groq Whisper API (`whisper-large-v3`) |
| LLM | Groq LLaMA3 |
| Frontend | Vanilla HTML / CSS / JS (dark UI) |
| Deployment | Render |

---

## Project structure

```
Voice_agent/
├── app.py                # Flask routes + shared text pipeline
├── templates/index.html  # Browser UI (vanilla HTML/CSS/JS)
├── config.py             # Model name
├── utils/
│   ├── stt.py            # Speech-to-text via Groq Whisper (lazy client)
│   ├── intent.py         # Rule-based intent first, LLM fallback
│   ├── tools.py          # Tool executor (file ops, code gen, chat)
│   ├── sandbox.py        # Filesystem sandbox for agent-written files
│   ├── documents.py      # PDF / DOCX / text extraction (zip-bomb safe)
│   ├── history.py        # Atomic, locked JSON chat history
│   └── memory.py         # In-process session history
├── tests/                # 55 offline tests (fake Groq client)
└── output/               # All generated files go here (auto-created)
```

---

## Safety: the agent can only write inside `output/`

File names come from speech or an LLM, so they're treated as hostile. `utils/sandbox.py` accepts a name only if it is a single path component made of safe characters, isn't `.`, `..` or hidden, is at most 100 characters, and still resolves inside `output/` after symlinks are followed. Anything else is rejected with a message rather than silently "fixed".

The test suite tries `../escape.txt`, absolute paths, Windows separators, null bytes, shell metacharacters, over-long names, non-string LLM output and a symlink pointing outside the sandbox.

Other hardening:

- The app **boots and serves typed commands without an API key**. Before, importing `utils/stt.py` created a Groq client at import time and crashed the whole app.
- Uploads: 25 MB default cap (`MAX_UPLOAD_MB`), audio format allow-list, binary-file detection, DOCX parsed in memory with an uncompressed-size cap.
- Input validation on `/run_text` (type and size); provider errors map to `502`, a missing key to `503`, unsupported types to `415`.
- Generated code has stray Markdown fences stripped before it's saved.
- Chat history is written atomically under a lock, so it can't be corrupted by a crash or concurrent requests.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest --cov=.        # 55 tests, no API key or network needed
ruff check .
```

CI runs lint and tests on Python 3.11 and 3.12, then boots the app under Gunicorn **without** an API key and checks that `/health` and the UI respond.

---

## Run locally

**Prerequisites:** Python 3.10+, a [Groq API key](https://console.groq.com) (free)

```bash
# 1. Clone the repo
git clone https://github.com/Diksha159457/Voice_agent.git
cd Voice_agent

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
echo "GROQ_API_KEY=gsk_your_key_here" > .env

# 5. Run
python app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Deployment (Render)

1. Push this repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect your GitHub repo
4. Set the following in Render's dashboard:

| Setting | Value |
|---|---|
| Start command | `gunicorn app:app --timeout 120 --workers 1` |
| Environment variable | `GROQ_API_KEY` = your key |

Render auto-deploys on every push to `main`.

---

## Limitations

- Chat history is a JSON file: fine for a single-instance demo, not for multiple users
- Uploads are limited to 25 MB by default (`MAX_UPLOAD_MB`)
- Single Gunicorn worker — not designed for high concurrency
- Generated files are ephemeral on Render's free tier

---

## Future improvements

- [ ] Persistent memory with a database (SQLite / PostgreSQL)
- [ ] User authentication
- [ ] Support for more intents (web search, calendar, email)
- [ ] Streaming LLM responses
- [x] Unit tests for intent detection and tool execution

---

## License

MIT — see [LICENSE](LICENSE)