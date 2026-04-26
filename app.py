# app.py — Main entry point for the Voice Agent web application.
# Starts a Flask web server, handles all HTTP routes, and serves
# the Claude-style single-page UI to the browser.

# ── Standard library imports ──────────────────────────────────────────────────
import os                                  # file deletion, path ops, reading env vars
import logging                             # lets us silence noisy Flask/Werkzeug startup logs
import sys                                 # lets us append extra paths so bundled packages are found
import tempfile                            # creates safe auto-named temp files for audio uploads
import zipfile                             # DOCX files are ZIP containers — we open them with this
import xml.etree.ElementTree as ET         # parses the XML inside DOCX without needing python-docx
from pathlib import Path                   # safe cross-platform filename/extension handling

# ── Third-party imports ───────────────────────────────────────────────────────
from flask import Flask, request, jsonify, render_template_string
# Flask                → creates the WSGI web application object
# request              → reads incoming HTTP data (uploaded files, JSON body, form fields)
# jsonify              → converts a Python dict into a proper JSON HTTP response
# render_template_string → renders an HTML string as a full browser page response

# ── Our own utility modules (all live inside utils/) ─────────────────────────
from utils.stt    import transcribe_audio       # audio file path → transcribed text string
from utils.intent import detect_intent          # text → intent dict  e.g. {"intent":"create_file","target":"calc.py"}
from utils.chat   import general_chat, streaming_chat  # text → conversational LLM reply string
from utils.tools  import execute_tool           # intent dict → runs the action → returns result string
from utils.memory import (
    add_to_memory,   # save one interaction dict to in-memory session history
    get_memory,      # return the full history list (used by the sidebar)
    clear_memory,    # wipe all history (called by the Clear History button)
)

# ── App creation ──────────────────────────────────────────────────────────────
app = Flask(__name__)
# Flask(__name__) creates the WSGI application.
# __name__ tells Flask where to find templates and static assets
# (same directory as this file).

# ── Upload size limit ─────────────────────────────────────────────────────────
MAX_FILE_CHARS = 12000
# Large uploaded documents can exceed the Groq model's context/TPM budget.
# We keep only the first MAX_FILE_CHARS characters so the request stays fast.

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
# Allow uploads up to 50 MB (50 × 1024 × 1024 bytes).
# Without this Flask silently returns HTTP 413 for large audio files.
# 50 MB covers roughly 50 minutes of compressed audio (mp3/m4a).

# ── Optional: auto-load .env file ─────────────────────────────────────────────
# If python-dotenv is installed, reads GROQ_API_KEY (and others) from a .env
# file in the project root so you don't have to export them every session.
# Falls back silently if dotenv isn't installed.
try:
    from dotenv import load_dotenv       # pip install python-dotenv  (optional)
    load_dotenv()                        # searches for .env in the current working directory
except ImportError:
    pass                                 # fine — just run: export GROQ_API_KEY=gsk_...

# ── LLM backend: Groq (cloud) or Ollama (local) ──────────────────────────────
USE_GROQ = os.environ.get("GROQ_API_KEY") is not None
# Checks whether GROQ_API_KEY exists in environment variables.
# True  → use Groq cloud API  (works on Streamlit Cloud / any server)
# False → fall back to local Ollama  (only works on your own machine)

if USE_GROQ:
    from groq import Groq                                    # import Groq SDK
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])  # authenticates with your API key

    def get_llm_response(text):
        """Send text to Groq's LLaMA3 model and return the reply string."""
        chat = groq_client.chat.completions.create(
            model="llama3-8b-8192",                          # model name on Groq's platform
            messages=[{"role": "user", "content": text}]    # single-turn message format
        )
        return chat.choices[0].message.content               # extract the reply text from response

else:
    import requests as http_requests                         # renamed to avoid shadowing Flask's `request`

    def get_llm_response(text):
        """Send text to a locally running Ollama server and return the reply string."""
        r = http_requests.post(
            "http://localhost:11434/api/generate",           # Ollama's local REST endpoint
            json={
                "model": "llama3",                           # model name loaded in Ollama
                "prompt": text,                              # the user's message
                "stream": False                              # get the full reply at once, not streamed
            }
        )
        return r.json()["response"]                          # Ollama wraps the reply in a "response" key

# ── Optional bundled document/PDF parsers ─────────────────────────────────────
# The project venv may not have pypdf installed, but the desktop runtime does.
# Adding that path makes PDF uploads work without forcing a pip install.
bundle_python = os.environ.get(
    "CODEX_BUNDLED_PYTHON_SITE",
    "/Users/dikshashahi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages",
)
if os.path.isdir(bundle_python) and bundle_python not in sys.path:
    sys.path.append(bundle_python)    # add bundled packages to Python's search path

try:
    from pypdf import PdfReader       # PDF text extraction — optional dependency
except ImportError:
    PdfReader = None                  # set to None so we can check availability later

# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# Written as a Python string — no separate templates/ folder needed.
# The browser renders this when it opens http://localhost:8501/
# ─────────────────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Voice Agent</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet" />
  <style>
    /* ── Reset ─────────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    /* Removes default browser spacing and makes padding count inside element width */

    /* ── Design tokens — change these to retheme the whole app ── */
    :root {
      --bg-primary:    #1a1a1a;   /* darkest layer — main page background */
      --bg-secondary:  #262626;   /* slightly lighter — sidebar background */
      --bg-input:      #2f2f2f;   /* textarea background */
      --bg-message:    #1e1e1e;   /* agent chat bubble background */
      --bg-user:       #2563eb;   /* user chat bubble — bright blue */
      --border:        #3a3a3a;   /* color of all dividing lines */
      --text-primary:  #ececec;   /* main readable text — off-white */
      --text-secondary:#a0a0a0;   /* muted labels and metadata */
      --text-hint:     #6b6b6b;   /* placeholder text inside inputs */
      --accent:        #d4a574;   /* warm amber — brand highlight color */
      --accent-hover:  #e8b98a;   /* slightly lighter amber for hover state */
      --danger:        #f87171;   /* red — destructive actions like Clear History */
      --r-sm: 6px; --r-md: 10px; --r-lg: 16px; --r-xl: 24px; /* border-radius scale */
    }

    html, body {
      height: 100%;                      /* fills the full viewport height */
      font-family: 'Inter', sans-serif;  /* Inter font loaded from Google Fonts */
      background: var(--bg-primary);     /* dark page background */
      color: var(--text-primary);        /* off-white default text */
      font-size: 15px;                   /* base font size — rem values relate to this */
      line-height: 1.6;                  /* comfortable reading line height */
      overflow: hidden;                  /* page body does NOT scroll — inner panels do */
    }

    /* ── App shell — sidebar + main side by side ─────────────── */
    .app-shell { display: flex; height: 100vh; }
    /* display:flex makes sidebar and main-area sit side by side */
    /* height:100vh fills the full screen height */

    /* ── Sidebar ─────────────────────────────────────────────── */
    .sidebar {
      width: 260px; min-width: 260px;          /* fixed width, won't shrink */
      background: var(--bg-secondary);          /* slightly lighter background */
      border-right: 1px solid var(--border);   /* thin line separating it from chat */
      display: flex; flex-direction: column;   /* stacks logo, history, footer vertically */
      padding: 20px 0;                         /* vertical breathing room, no horizontal */
      overflow-y: auto;                        /* scrolls if history is long */
    }

    .sidebar-logo {
      display: flex; align-items: center; gap: 10px; /* icon + text side by side, vertically centered */
      padding: 0 20px 24px;                    /* bottom gap before the divider line */
      border-bottom: 1px solid var(--border);  /* separates logo from history list */
    }

    .logo-icon {
      width: 32px; height: 32px;               /* fixed square */
      border-radius: 50%;                      /* makes it a circle */
      background: var(--accent);               /* amber fill */
      display: flex; align-items: center; justify-content: center; /* centers "VA" text */
      font-size: 13px; font-weight: 700;       /* bold initials */
      color: #1a1a1a;                          /* dark text on amber for contrast */
      flex-shrink: 0;                          /* won't shrink if parent is narrow */
    }

    .logo-text { font-size: 15px; font-weight: 600; letter-spacing: -.02em; }
    /* slightly tight letter-spacing gives a modern look */

    .sidebar-label {
      font-size: 11px; font-weight: 500; color: var(--text-hint);
      text-transform: uppercase; letter-spacing: .08em; /* spaced-out caps like "RECENT" */
      padding: 20px 20px 8px;               /* space above and below the label */
    }

    .history-item {
      padding: 8px 20px; font-size: 13px; color: var(--text-secondary);
      border-left: 2px solid transparent;   /* invisible left border — can highlight active item */
      cursor: default;
      transition: background .15s;          /* smooth hover fade */
    }
    .history-item:hover { background: rgba(255,255,255,.04); } /* subtle hover highlight */

    .hi-intent { font-size: 11px; color: var(--accent); font-weight: 500; margin-bottom: 2px; }
    /* small amber label showing the detected intent above each history item */

    .hi-text {
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px;
      /* single line with "…" if text is too long for the sidebar */
    }

    .sidebar-footer {
      margin-top: auto;                     /* pushes footer to the very bottom of sidebar */
      padding: 16px 20px;
      border-top: 1px solid var(--border); /* line above the Clear History button */
    }

    /* ── Main chat area ─────────────────────────────────────── */
    .main-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
    /* flex:1 means it takes ALL remaining space after the sidebar */

    .topbar {
      height: 56px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; padding: 0 24px; gap: 12px;
      flex-shrink: 0;    /* topbar never shrinks even if content below is large */
    }

    .topbar-title { font-size: 15px; font-weight: 500; }

    .model-badge {
      font-size: 12px; padding: 3px 10px; border-radius: 20px;
      background: rgba(212,165,116,.12); color: var(--accent);
      border: 1px solid rgba(212,165,116,.25); font-weight: 500;
      /* pill-shaped badge showing which model is active */
    }

    .messages-area { flex: 1; overflow-y: auto; padding: 32px 0; }
    /* flex:1 fills remaining vertical space; overflow-y:auto makes THIS scroll */

    .messages-inner { max-width: 700px; margin: 0 auto; padding: 0 24px; }
    /* centers chat content and limits its width for readability */

    /* ── Welcome / empty state ─────────────────────────────── */
    .welcome-state { text-align: center; padding: 60px 20px 40px; }
    .welcome-icon {
      width: 56px; height: 56px; border-radius: 50%;
      background: rgba(212,165,116,.15);               /* faint amber circle */
      display: flex; align-items: center; justify-content: center;
      margin: 0 auto 20px; font-size: 24px;            /* centered mic emoji */
    }
    .welcome-title { font-size: 22px; font-weight: 600; letter-spacing: -.02em; margin-bottom: 10px; }
    .welcome-sub { font-size: 14px; color: var(--text-secondary); max-width: 400px; margin: 0 auto 32px; }
    .suggestions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; max-width: 480px; margin: 0 auto; }
    /* 2-column grid of suggestion chips */

    .suggestion-chip {
      background: var(--bg-secondary); border: 1px solid var(--border);
      border-radius: var(--r-md); padding: 12px 14px;
      text-align: left; font-size: 13px; color: var(--text-secondary);
      cursor: pointer; transition: border-color .15s, background .15s; line-height: 1.4;
    }
    .suggestion-chip:hover { border-color: var(--accent); background: rgba(212,165,116,.06); color: var(--text-primary); }
    /* hovering a chip glows with amber border */

    .chip-icon { font-size: 16px; margin-bottom: 6px; display: block; }

    /* ── Message bubbles ────────────────────────────────────── */
    .message { display: flex; gap: 12px; margin-bottom: 24px; animation: fadeIn .2s ease; }
    /* each message row is: avatar + bubble, side by side */

    @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
    /* new messages slide up and fade in smoothly */

    .message.user { flex-direction: row-reverse; }
    /* user messages flip the row so avatar is on the right */

    .msg-avatar {
      width: 32px; height: 32px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 11px; font-weight: 700; flex-shrink: 0; margin-top: 2px;
    }
    .msg-avatar.agent { background: var(--accent); color: #1a1a1a; } /* amber "VA" avatar */
    .msg-avatar.user  { background: var(--bg-user); color: #fff; }   /* blue "U" avatar */

    .msg-content { max-width: 80%; display: flex; flex-direction: column; gap: 4px; }
    /* bubble + metadata stacked vertically, capped at 80% width */

    .message.user .msg-content { align-items: flex-end; }
    /* user content aligns to the right side */

    .msg-bubble { padding: 12px 16px; border-radius: var(--r-lg); font-size: 14px; line-height: 1.65; }
    .message.agent .msg-bubble {
      background: var(--bg-message); border: 1px solid var(--border);
      border-top-left-radius: 4px;   /* flattens top-left corner → points to avatar */
    }
    .message.user .msg-bubble { background: var(--bg-user); border-top-right-radius: 4px; color: #fff; }
    /* user bubble is blue; top-right corner flat → points to avatar on the right */

    .msg-meta { font-size: 11px; color: var(--text-hint); display: flex; align-items: center; gap: 6px; }
    .intent-tag {
      background: rgba(212,165,116,.12); color: var(--accent);
      padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500;
      /* small amber pill showing the detected intent below an agent bubble */
    }

    /* ── Input area ─────────────────────────────────────────── */
    .input-area { padding: 16px 24px 20px; border-top: 1px solid var(--border); flex-shrink: 0; }
    /* flex-shrink:0 keeps the input bar at a fixed height — only messages scroll */

    .input-inner { max-width: 700px; margin: 0 auto; }
    /* aligns input with the message column above */

    .input-box {
      background: var(--bg-input); border: 1px solid var(--border);
      border-radius: var(--r-xl); padding: 12px 16px; transition: border-color .15s;
    }
    .input-box:focus-within { border-color: rgba(212,165,116,.5); }
    /* the whole box glows amber when the textarea is focused */

    .text-input {
      width: 100%; background: transparent; border: none; outline: none;
      color: var(--text-primary); font-family: 'Inter', sans-serif;
      font-size: 14px; line-height: 1.6; resize: none;
      min-height: 24px; max-height: 200px; overflow-y: auto; display: block;
      /* transparent background so the box style shows through */
      /* resize:none hides the default browser resize handle */
    }
    .text-input::placeholder { color: var(--text-hint); }

    .input-controls {
      display: flex; align-items: center; justify-content: space-between;
      margin-top: 10px; gap: 10px;
      /* row of buttons below the textarea: left group + send button */
    }
    .input-left { display: flex; align-items: center; gap: 8px; }

    .upload-btn {
      display: flex; align-items: center; gap: 6px;
      padding: 6px 12px; border-radius: var(--r-sm);
      background: rgba(255,255,255,.05); border: 1px solid var(--border);
      color: var(--text-secondary); font-size: 12px; font-weight: 500;
      cursor: pointer; transition: background .15s, border-color .15s;
      user-select: none;   /* prevents text being selected when clicking quickly */
    }
    .upload-btn:hover { background: rgba(255,255,255,.09); border-color: rgba(255,255,255,.2); color: var(--text-primary); }
    .upload-btn.has-file { border-color: var(--accent); color: var(--accent); background: rgba(212,165,116,.08); }
    /* has-file class turns the button amber to confirm a file is loaded */

    .record-btn.recording {
      background: rgba(248,113,113,.14);
      border-color: rgba(248,113,113,.45);
      color: #fecaca;
      /* recording state turns the mic button red */
    }

    .file-name {
      font-size: 12px; color: var(--accent);
      max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      /* shows the selected filename truncated with "…" if too long */
    }

    .audio-preview {
      margin-top: 10px; width: 100%;
      display: none;                   /* hidden until a file is selected/recorded */
      filter: sepia(.15) saturate(.85); /* subtle warm tint on the native audio player */
    }

    .send-btn {
      width: 36px; height: 36px; border-radius: 50%;  /* circular button */
      background: var(--accent); border: none; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      transition: background .15s, transform .1s; flex-shrink: 0;
    }
    .send-btn:hover  { background: var(--accent-hover); }
    .send-btn:active { transform: scale(.94); }       /* slight press-down effect */
    .send-btn:disabled { background: #3a3a3a; cursor: not-allowed; opacity: .5; }
    /* disabled when input is empty or a request is in flight */

    .status-bar { font-size: 12px; color: var(--text-hint); margin-top: 8px; text-align: center; min-height: 16px; }
    /* shows "Transcribing…" / "Thinking…" below the input */

    .btn {
      padding: 7px 14px; border-radius: var(--r-sm); border: 1px solid var(--border);
      background: transparent; color: var(--text-secondary); font-size: 12px;
      font-weight: 500; cursor: pointer; font-family: 'Inter', sans-serif;
      transition: background .15s, color .15s;
    }
    .btn:hover { background: rgba(255,255,255,.06); color: var(--text-primary); }
    .btn.danger { color: var(--danger); border-color: rgba(248,113,113,.3); }
    .btn.danger:hover { background: rgba(248,113,113,.08); }
    /* red styling for destructive buttons like Clear History */

    .typing-dots { display: flex; align-items: center; gap: 4px; padding: 4px 0; }
    .typing-dots span {
      width: 6px; height: 6px; border-radius: 50%; background: var(--text-hint);
      animation: bounce 1.2s infinite;   /* each dot bounces on a loop */
    }
    .typing-dots span:nth-child(2) { animation-delay: .2s; }  /* stagger dot 2 */
    .typing-dots span:nth-child(3) { animation-delay: .4s; }  /* stagger dot 3 */
    @keyframes bounce { 0%,80%,100% { transform:translateY(0); opacity:.4; } 40% { transform:translateY(-5px); opacity:1; } }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    /* custom thin scrollbar for webkit browsers (Chrome, Safari, Edge) */

    @media (max-width: 640px) { .sidebar { display: none; } }
    /* hide sidebar on mobile — not enough space */
  </style>
</head>
<body>
<div class="app-shell">

  <!-- ══════════════ SIDEBAR ══════════════ -->
  <aside class="sidebar">
    <div class="sidebar-logo">
      <div class="logo-icon">VA</div>          <!-- "VA" = Voice Agent initials -->
      <span class="logo-text">Voice Agent</span>
    </div>
    <div class="sidebar-label">Recent</div>    <!-- section label above history list -->
    <div id="historyList"></div>               <!-- JS fills this with history items -->
    <div class="sidebar-footer">
      <!-- danger class makes it red; onclick calls clearHistory() defined in JS -->
      <button class="btn danger" onclick="clearHistory()" style="width:100%">Clear history</button>
    </div>
  </aside>

  <!-- ══════════════ MAIN CHAT ══════════════ -->
  <main class="main-area">

    <!-- Top bar with title and model badge -->
    <div class="topbar">
      <span class="topbar-title">Voice Agent</span>
      <span class="model-badge">Whisper tiny &middot; Groq LLaMA3</span>
      <!-- &middot; is the HTML entity for the · dot separator -->
    </div>

    <!-- Scrollable message list -->
    <div class="messages-area" id="messagesArea">
      <div class="messages-inner">

        <!-- Empty state — JS hides this div once the first message arrives -->
        <div class="welcome-state" id="welcomeState">
          <div class="welcome-icon">🎙</div>
          <h1 class="welcome-title">How can I help you?</h1>
          <p class="welcome-sub">Type a command, record from your mic, upload audio, or attach a text/code file.</p>
          <div class="suggestions">
            <!-- Each chip calls fillInput() to pre-fill the textarea with a sample prompt -->
            <div class="suggestion-chip" onclick="fillInput('Create a Python file called calculator.py with add and subtract functions')">
              <span class="chip-icon">📄</span>Create a calculator.py file
            </div>
            <div class="suggestion-chip" onclick="fillInput('What is machine learning?')">
              <span class="chip-icon">💬</span>What is machine learning?
            </div>
            <div class="suggestion-chip" onclick="fillInput('Make a new folder called my_project')">
              <span class="chip-icon">📁</span>Create a project folder
            </div>
            <div class="suggestion-chip" onclick="fillInput('Summarize this: Python is a high-level programming language known for its simplicity.')">
              <span class="chip-icon">✂️</span>Summarize some text
            </div>
          </div>
        </div>

        <div id="messagesList"></div>  <!-- JS appends bubble divs here -->
      </div>
    </div>

    <!-- ── Input panel ── -->
    <div class="input-area">
      <div class="input-inner">
        <div class="input-box">

          <!-- Textarea: Enter = send, Shift+Enter = newline -->
          <textarea
            class="text-input" id="textInput" rows="1"
            placeholder="Message Voice Agent..."
            onkeydown="handleKey(event)"     <!-- catches Enter key to submit -->
            oninput="autoResize(this)"       <!-- grows textarea as user types -->
          ></textarea>

          <div class="input-controls">
            <div class="input-left">

              <!-- Hidden real audio file input — triggered by the <label> below -->
              <input
                type="file" id="audioFile"
                accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm"
                style="display:none"              <!-- invisible; label acts as the button -->
                onchange="onFileSelected(this)"   <!-- called when user picks a file -->
              />

              <!-- Hidden generic file picker for text/code/PDF/DOCX attachments -->
              <input
                type="file" id="attachmentFile"
                accept=".txt,.md,.py,.js,.ts,.tsx,.json,.csv,.html,.css,.java,.c,.cpp,.sql,.xml,.yaml,.yml,.pdf,.docx"
                style="display:none"
                onchange="onAttachmentSelected(this)"
              />

              <!-- Mic record button — toggleRecording() starts/stops browser recording -->
              <button type="button" class="upload-btn record-btn" id="recordBtn" onclick="toggleRecording()">
                <!-- Microphone SVG icon -->
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8" y1="23" x2="16" y2="23"/>
                </svg>
                <span id="recordLabelText">Record audio</span>  <!-- text changes to "Stop recording" while active -->
              </button>

              <!-- Styled label that opens the hidden audio file picker when clicked -->
              <label for="audioFile" class="upload-btn" id="uploadLabel">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8"  y1="23" x2="16" y2="23"/>
                </svg>
                <span id="uploadLabelText">Upload audio</span>
              </label>

              <!-- Styled label that opens the hidden generic file picker -->
              <label for="attachmentFile" class="upload-btn" id="attachmentLabel">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                  <path d="M14 3h7v7"/>
                  <path d="M10 14L21 3"/>
                </svg>
                <span id="attachmentLabelText">Upload file</span>
              </label>

              <!-- Filename chips — shown after a file is selected/recorded -->
              <span class="file-name" id="audioName"      style="display:none"></span>
              <span class="file-name" id="attachmentName" style="display:none"></span>

            </div>

            <!-- Circular send button — disabled until there is content to send -->
            <button class="send-btn" id="sendBtn" onclick="handleSend()" disabled title="Send (Enter)">
              <!-- Paper-plane send icon -->
              <svg width="15" height="15" viewBox="0 0 24 24" fill="#1a1a1a">
                <path d="M2 21l21-9L2 3v7l15 2-15 2z"/>
              </svg>
            </button>
          </div>
        </div>

        <!-- Native audio player — shown after recording or file selection so user can review -->
        <audio class="audio-preview" id="audioPreview" controls></audio>

        <!-- Status text area — shows "Transcribing…", "Thinking…", "Heard: …" -->
        <div class="status-bar" id="statusBar"></div>
      </div>
    </div>
  </main>
</div>

<script>
// ─────────────────────────────────────────────────────────────────────────────
// DOM REFERENCES — grabbed once at startup and reused everywhere
// ─────────────────────────────────────────────────────────────────────────────
const textInput      = document.getElementById('textInput');       // the textarea element
const sendBtn        = document.getElementById('sendBtn');         // circular amber send button
const audioFile      = document.getElementById('audioFile');       // hidden audio <input type=file>
const attachmentFile = document.getElementById('attachmentFile');  // hidden generic <input type=file>
const uploadLabel    = document.getElementById('uploadLabel');     // styled label for audio picker
const attachmentLabel= document.getElementById('attachmentLabel'); // styled label for generic picker
const audioName      = document.getElementById('audioName');       // audio filename chip span
const attachmentName = document.getElementById('attachmentName'); // generic filename chip span
const recordBtn      = document.getElementById('recordBtn');       // microphone record button
const audioPreview   = document.getElementById('audioPreview');   // native <audio> player
const statusBar      = document.getElementById('statusBar');      // "Transcribing…" text line
const msgList        = document.getElementById('messagesList');   // container for chat bubbles
const histList       = document.getElementById('historyList');    // sidebar history container
const welcomeEl      = document.getElementById('welcomeState');   // empty-state block

// ── State variables ────────────────────────────────────────────────────────
let busy           = false;   // true while a server request is in flight — prevents double-submit
let mediaRecorder  = null;    // active MediaRecorder instance (null when not recording)
let recordedChunks = [];      // audio data chunks collected during recording
let recordedBlob   = null;    // final Blob assembled from chunks after recording stops
let recordingStream= null;    // microphone MediaStream — stored so we can release the mic cleanly
let audioPreviewUrl= null;    // object URL for the audio preview player (freed after use)

// ─────────────────────────────────────────────────────────────────────────────
// TEXTAREA AUTO-RESIZE — grows as the user types, up to 200px then scrolls
// ─────────────────────────────────────────────────────────────────────────────
function autoResize(el) {
  el.style.height = 'auto';                                    // reset height so scrollHeight is accurate
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';    // grow up to 200px, then scroll inside
  refreshSendBtn();                                            // re-check if send button should enable
}

// ─────────────────────────────────────────────────────────────────────────────
// SEND BUTTON STATE — enable only when there is something to send and not busy
// ─────────────────────────────────────────────────────────────────────────────
function refreshSendBtn() {
  const hasText       = textInput.value.trim().length > 0;      // user typed something
  const hasAudio      = audioFile.files.length > 0 || recordedBlob !== null; // file or recording
  const hasAttachment = attachmentFile.files.length > 0;        // generic file attached
  sendBtn.disabled = busy || (!hasText && !hasAudio && !hasAttachment);
  // disabled if busy OR if there is nothing at all to send
}

// ─────────────────────────────────────────────────────────────────────────────
// AUDIO FILE SELECTED — called by onchange on the hidden audio <input>
// ─────────────────────────────────────────────────────────────────────────────
function onFileSelected(input) {
  if (input.files.length === 0) return;          // user hit Cancel in the picker — nothing to do
  const selectedFile = input.files[0];           // the File object chosen by the user
  const name = selectedFile.name;                // e.g. "voice_note.m4a"

  recordedBlob = null;                           // discard any previous browser recording
  setRecordVisualState(false, 'Record audio');   // reset mic button to its default look

  uploadLabel.classList.add('has-file');                              // turn upload button amber
  document.getElementById('uploadLabelText').textContent = 'Audio ready'; // update label text

  audioName.textContent    = name;               // show filename next to the button
  audioName.style.display  = 'inline';

  setAudioPreview(selectedFile);                 // load file into the native audio player
  refreshSendBtn();                              // file present → enable send button
}

// ─────────────────────────────────────────────────────────────────────────────
// GENERIC FILE SELECTED — called by onchange on the attachment <input>
// ─────────────────────────────────────────────────────────────────────────────
function onAttachmentSelected(input) {
  if (input.files.length === 0) return;          // user cancelled — nothing to do

  attachmentLabel.classList.add('has-file');                              // turn button amber
  document.getElementById('attachmentLabelText').textContent = 'File ready';
  attachmentName.textContent   = input.files[0].name;  // show the filename
  attachmentName.style.display = 'inline';

  refreshSendBtn();  // file present → enable send button
}

// ─────────────────────────────────────────────────────────────────────────────
// BROWSER RECORDING — click once to start, click again to stop
// ─────────────────────────────────────────────────────────────────────────────
async function toggleRecording() {
  if (busy) return;   // don't allow recording while a server request is running

  // If already recording, stop it
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();          // triggers the onstop handler below
    setStatus('Finishing recording…');
    return;
  }

  try {
    // Ask the browser for microphone permission and get the audio stream
    recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });

    // Create a MediaRecorder attached to the mic stream
    mediaRecorder = new MediaRecorder(recordingStream);
    recordedChunks = [];   // clear previous chunks before a new recording

    // Each time the recorder has a chunk of audio ready, push it to the array
    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordedChunks.push(event.data);  // accumulate audio data
      }
    };

    // When recording stops: assemble the final blob and update the UI
    mediaRecorder.onstop = () => {
      // Merge all chunks into one Blob using the recorder's detected MIME type
      recordedBlob = new Blob(recordedChunks, {
        type: mediaRecorder.mimeType || 'audio/webm',  // webm is the most common browser format
      });

      // Release the microphone immediately — stops the red indicator in the browser tab
      if (recordingStream) {
        recordingStream.getTracks().forEach(track => track.stop());
        recordingStream = null;
      }

      // Update UI to show recording is ready
      uploadLabel.classList.add('has-file');
      document.getElementById('uploadLabelText').textContent = 'Recorded audio';
      audioName.textContent   = 'recording.webm';
      audioName.style.display = 'inline';

      setAudioPreview(recordedBlob);                   // let user play back the recording
      setRecordVisualState(false, 'Re-record');        // reset mic button
      setStatus('Recorded audio is ready to send.');
      refreshSendBtn();                                // recording exists → enable send
    };

    mediaRecorder.start();   // begin capturing audio from the mic

    // Clear any previously selected file so only one audio source is active
    recordedBlob  = null;
    audioFile.value = '';
    audioName.textContent   = '';
    audioName.style.display = 'none';
    uploadLabel.classList.remove('has-file');
    document.getElementById('uploadLabelText').textContent = 'Upload audio';

    setRecordVisualState(true, 'Stop recording');      // turn mic button red
    setStatus('Recording from microphone… click again to stop.');
    refreshSendBtn();

  } catch (err) {
    setRecordVisualState(false, 'Record audio');       // restore button on failure
    setStatus('Microphone access failed: ' + err.message);
  }
}

// Updates the record button's visual state (red when recording, normal otherwise)
function setRecordVisualState(isRecording, label) {
  recordBtn.classList.toggle('recording', isRecording); // adds/removes the red "recording" class
  document.getElementById('recordLabelText').textContent = label;
}

// Loads audio into the native <audio> player so the user can preview before sending
function setAudioPreview(source) {
  if (audioPreviewUrl) {
    URL.revokeObjectURL(audioPreviewUrl);  // free the previous object URL to avoid memory leaks
    audioPreviewUrl = null;
  }
  audioPreviewUrl    = URL.createObjectURL(source); // create a temporary browser URL for the blob/file
  audioPreview.src   = audioPreviewUrl;
  audioPreview.style.display = 'block';             // make the player visible
  audioPreview.load();                              // reload the player with the new source
}

// ─────────────────────────────────────────────────────────────────────────────
// KEYBOARD HANDLER — Enter sends, Shift+Enter adds a newline
// ─────────────────────────────────────────────────────────────────────────────
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {    // Enter without Shift
    e.preventDefault();                       // stop the browser inserting a real newline
    if (!sendBtn.disabled) handleSend();      // only send if button is currently enabled
  }
  // Shift+Enter falls through and the browser inserts a newline normally
}

// ─────────────────────────────────────────────────────────────────────────────
// FILL INPUT — suggestion chips call this to pre-populate the textarea
// ─────────────────────────────────────────────────────────────────────────────
function fillInput(text) {
  textInput.value = text;      // set the textarea content programmatically
  autoResize(textInput);       // recalculate height since value was changed by JS not user
  textInput.focus();           // move keyboard focus to the textarea
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN SEND HANDLER — decides which backend route to call
// ─────────────────────────────────────────────────────────────────────────────
async function handleSend() {
  if (busy) return;                                             // guard: already waiting for a response

  const text       = textInput.value.trim();                   // typed text (may be empty)
  const audio      = audioFile.files[0] || null;               // selected audio file (may be null)
  const attachment = attachmentFile.files[0] || null;          // selected text/code file (may be null)

  if (!text && !audio && !attachment && !recordedBlob) return; // nothing to send — exit early

  if (attachment) {
    await sendAttachment(attachment, text);   // file uploads go to /run_file
  } else if (audio || recordedBlob) {
    // Convert recordedBlob to a File object so FormData can send it with a filename
    const audioToSend = audio || new File(
      [recordedBlob],
      'recording.webm',
      { type: recordedBlob.type || 'audio/webm' }
    );
    await sendAudio(audioToSend, text);       // audio goes to /run_audio
  } else {
    await sendText(text);                     // plain text goes to /run_text
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEND AUDIO — POST multipart/form-data to /run_audio
// ─────────────────────────────────────────────────────────────────────────────
async function sendAudio(file, extraText) {
  appendMessage('user', extraText || '🎵 ' + file.name);  // show what the user sent
  showTyping();                                            // show bouncing dots
  setStatus('Transcribing audio…');
  setBusy(true);

  const form = new FormData();            // multipart body — only way to send a File via fetch
  form.append('audio', file, file.name); // field name must match request.files.get("audio") in Flask
  if (extraText) form.append('note', extraText);  // optional typed context

  try {
    const res  = await fetch('/run_audio', { method: 'POST', body: form });
    // Note: do NOT set Content-Type header — browser sets it automatically with the correct boundary
    const data = await res.json();        // parse the JSON response from Flask
    handleResponse(data);
  } catch (err) {
    removeTyping();
    appendMessage('agent', '⚠️ Network error: ' + err.message);
  } finally {
    setBusy(false);
    clearInputs();   // clear AFTER fetch so the File reference isn't lost mid-request
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEND FILE — POST text/code/PDF/DOCX attachment to /run_file
// ─────────────────────────────────────────────────────────────────────────────
async function sendAttachment(file, extraText) {
  appendMessage('user', extraText || '📎 ' + file.name);
  showTyping();
  setStatus('Reading file…');
  setBusy(true);

  const form = new FormData();
  form.append('file', file, file.name);           // field name matches request.files.get("file")
  if (extraText) form.append('note', extraText);

  try {
    const res  = await fetch('/run_file', { method: 'POST', body: form });
    const data = await res.json();
    handleResponse(data);
  } catch (err) {
    removeTyping();
    appendMessage('agent', '⚠️ Network error: ' + err.message);
  } finally {
    setBusy(false);
    clearInputs();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEND TEXT — POST JSON body to /run_text
// ─────────────────────────────────────────────────────────────────────────────
async function sendText(text) {
  appendMessage('user', text);
  showTyping();
  setStatus('Thinking…');
  setBusy(true);

  try {
    const res = await fetch('/run_text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },  // tell Flask to expect JSON
      body: JSON.stringify({ text }),                    // serialize the text field to JSON
    });
    const data = await res.json();
    handleResponse(data);
  } catch (err) {
    removeTyping();
    appendMessage('agent', '⚠️ Network error: ' + err.message);
  } finally {
    setBusy(false);
    clearInputs();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// HANDLE SERVER RESPONSE — updates chat after any /run_* route replies
// ─────────────────────────────────────────────────────────────────────────────
function handleResponse(data) {
  removeTyping();   // remove the bouncing dots placeholder
  setStatus('');    // clear the "Thinking…" status text

  if (data.error) {
    appendMessage('agent', '⚠️ ' + data.error);   // show server error in chat
  } else {
    appendMessage('agent', data.result, data.intent);  // show reply + optional intent pill
    if (data.transcript) {
      setStatus('Heard: "' + data.transcript + '"');   // show what Whisper transcribed
    }
  }
  loadHistory();   // refresh the sidebar Recent list with the new entry
}

// ─────────────────────────────────────────────────────────────────────────────
// APPEND MESSAGE BUBBLE — creates and inserts a chat message div
// ─────────────────────────────────────────────────────────────────────────────
function appendMessage(role, text, intent) {
  welcomeEl.style.display = 'none';   // hide the empty state on the first message

  const wrap = document.createElement('div');
  wrap.className = 'message ' + role;  // "message agent" or "message user"

  const pill = intent ? `<span class="intent-tag">${intent}</span>` : '';  // optional amber pill

  wrap.innerHTML = `
    <div class="msg-avatar ${role}">${role === 'agent' ? 'VA' : 'U'}</div>
    <div class="msg-content">
      <div class="msg-bubble">${escHtml(text)}</div>
      <div class="msg-meta">${pill}</div>
    </div>`;

  msgList.appendChild(wrap);   // add bubble to the chat

  const area = document.getElementById('messagesArea');
  area.scrollTop = area.scrollHeight;   // auto-scroll to the newest message
}

// ─────────────────────────────────────────────────────────────────────────────
// TYPING INDICATOR — three animated dots shown while waiting for the server
// ─────────────────────────────────────────────────────────────────────────────
function showTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'message agent';
  wrap.id = 'typingMsg';   // ID so we can find and remove it later
  wrap.innerHTML = `
    <div class="msg-avatar agent">VA</div>
    <div class="msg-content">
      <div class="msg-bubble" style="padding:14px 16px">
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </div>
    </div>`;
  msgList.appendChild(wrap);
  document.getElementById('messagesArea').scrollTop = 99999;  // scroll to show the dots
}

function removeTyping() {
  const el = document.getElementById('typingMsg');
  if (el) el.remove();   // remove the dots bubble when the real response arrives
}

// ─────────────────────────────────────────────────────────────────────────────
// SIDEBAR HISTORY — fetch /history and render items newest-first
// ─────────────────────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const data = await (await fetch('/history')).json();  // GET /history → array of objects
    histList.innerHTML = '';   // clear existing items before re-rendering

    if (!data.length) {
      histList.innerHTML = '<div style="padding:12px 20px;font-size:13px;color:var(--text-hint)">No history yet</div>';
      return;
    }

    [...data].reverse().forEach(item => {   // spread+reverse to show newest first without mutating original
      const div = document.createElement('div');
      div.className = 'history-item';
      div.innerHTML = `
        <div class="hi-intent">${item.intent || 'general_chat'}</div>
        <div class="hi-text" title="${escHtml(item.text)}">${escHtml(item.text)}</div>`;
      histList.appendChild(div);
    });
  } catch (_) {}   // silently fail — history is non-critical to core functionality
}

// ─────────────────────────────────────────────────────────────────────────────
// CLEAR HISTORY — wipes server memory and resets the UI
// ─────────────────────────────────────────────────────────────────────────────
async function clearHistory() {
  await fetch('/clear_history', { method: 'POST' });  // POST to Flask → clears server-side memory
  msgList.innerHTML   = '';                           // remove all chat bubbles from the DOM
  welcomeEl.style.display = '';                       // show the empty state again
  histList.innerHTML  = '';                           // clear the sidebar list
  setStatus('');
}

// ─────────────────────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────────────────────
function setStatus(msg) { statusBar.textContent = msg; }   // update the status text line

function setBusy(val) {
  busy = val;         // update the global busy flag
  refreshSendBtn();   // re-evaluate whether send button should be enabled
}

function clearInputs() {
  textInput.value        = '';           // clear the textarea
  textInput.style.height = 'auto';      // reset textarea height

  audioFile.value        = '';          // reset audio file picker
  attachmentFile.value   = '';          // reset generic file picker
  recordedBlob           = null;        // discard the in-browser recording blob

  audioName.style.display      = 'none';  // hide audio filename chip
  audioName.textContent        = '';
  attachmentName.style.display = 'none';  // hide attachment filename chip
  attachmentName.textContent   = '';

  if (audioPreviewUrl) {
    URL.revokeObjectURL(audioPreviewUrl);  // free the object URL to release browser memory
    audioPreviewUrl = null;
  }

  audioPreview.pause();                    // stop playback
  audioPreview.removeAttribute('src');     // detach the audio source
  audioPreview.style.display = 'none';     // hide the player

  uploadLabel.classList.remove('has-file');     // reset audio button to default style
  attachmentLabel.classList.remove('has-file'); // reset file button to default style

  document.getElementById('uploadLabelText').textContent     = 'Upload audio';
  document.getElementById('attachmentLabelText').textContent = 'Upload file';

  setRecordVisualState(false, 'Record audio');  // reset mic button
  refreshSendBtn();                             // disable send button (nothing to send)
}

// XSS prevention — converts special HTML characters to safe entities before inserting into DOM
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')    // & → &amp;  (must be first to avoid double-escaping)
    .replace(/</g, '&lt;')     // < → &lt;
    .replace(/>/g, '&gt;')     // > → &gt;
    .replace(/"/g, '&quot;');  // " → &quot;
}

// Load sidebar history when the page first opens
loadHistory();
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main page — just renders the HTML string above."""
    return render_template_string(HTML)


@app.route("/run_audio", methods=["POST"])
def run_audio():
    """
    Receives an audio file upload, transcribes it with Whisper,
    then passes the transcript through the normal text pipeline.

    Flow:
      1. Validate the upload exists
      2. Preserve the original file extension (ffmpeg detects format from extension)
      3. Write to a temp file and transcribe
      4. Delete temp file even if transcription fails (via finally)
      5. Pass transcript to _process_text()
    """
    audio = request.files.get("audio")   # None if the "audio" field is missing from the form
    if not audio:
        return jsonify({"error": "No audio file received. Make sure the field name is 'audio'."}), 400

    note = request.form.get("note", "").strip()  # optional typed context sent alongside the audio

    # ── Preserve original file extension ─────────────────────────────────────
    # faster-whisper delegates audio decoding to ffmpeg.
    # ffmpeg infers the codec from the FILE EXTENSION, not the Content-Type header.
    # If we always save as ".wav" but the user sends ".m4a", ffmpeg will fail.
    original_name = audio.filename or "upload.wav"      # filename sent by the browser
    _, ext        = os.path.splitext(original_name)     # split into ("upload", ".m4a")
    ext           = ext.lower() if ext else ".wav"       # default to .wav if no extension

    # ── Write to disk and transcribe ─────────────────────────────────────────
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            # delete=False: we control deletion ourselves
            # (required on Windows where open files can't be deleted by another process)
            audio.save(tmp.name)    # stream the uploaded bytes directly to the temp file
            tmp_path = tmp.name     # save the path for use after the 'with' block closes

        transcript = transcribe_audio(tmp_path)   # Whisper → plain text string

    except Exception as e:
        return jsonify({
            "error": (
                f"Transcription failed: {e}. "
                "Make sure ffmpeg is installed: brew install ffmpeg (Mac) "
                "/ sudo apt install ffmpeg (Linux)"
            )
        }), 500

    finally:
        # Always delete the temp file — even if an exception was raised above
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not transcript.strip():
        return jsonify({"error": "No speech detected. Try speaking louder or closer to the mic."}), 400

    # If the user also typed a note, combine it with the transcript
    request_text = transcript if not note else f"{note}\n\nSpoken request:\n{transcript}"

    # Process exactly like a typed text message
    result = _process_text(request_text)
    result["transcript"] = transcript   # include what Whisper heard so the UI can display it
    return jsonify(result)


@app.route("/run_file", methods=["POST"])
def run_file():
    """
    Receives a text/code/PDF/DOCX file upload, extracts its text content,
    and passes it through the normal text pipeline.
    """
    uploaded = request.files.get("file")   # field name matches form.append('file', ...) in JS
    if not uploaded:
        return jsonify({"error": "No file received. Choose a file and try again."}), 400

    # Use only the base filename — never trust the full path sent by the browser
    filename = Path(uploaded.filename or "uploaded_file.txt").name
    suffix   = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""  # e.g. "pdf", "py"

    note      = request.form.get("note", "").strip()  # optional typed instruction from the textarea
    raw_bytes = uploaded.read()   # read file bytes into memory

    if not raw_bytes:
        return jsonify({"error": "The selected file is empty."}), 400

    try:
        if suffix == "pdf":
            # ── PDF extraction ────────────────────────────────────────────────
            if PdfReader is None:
                return jsonify({"error": "PDF support is not available in this environment."}), 500

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(raw_bytes)    # write PDF bytes to disk so PdfReader can open it
                pdf_path = tmp.name

            try:
                reader   = PdfReader(pdf_path)
                contents = "\n".join(
                    (page.extract_text() or "") for page in reader.pages
                ).strip()              # extract text from every page and join them
            finally:
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)   # always clean up the temp PDF

        elif suffix == "docx":
            # ── DOCX extraction — reads XML directly, no python-docx needed ──
            # A .docx file is just a ZIP archive containing XML files.
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(raw_bytes)
                docx_path = tmp.name

            try:
                with zipfile.ZipFile(docx_path) as docx_zip:
                    xml_bytes = docx_zip.read("word/document.xml")   # main content XML
                    root      = ET.fromstring(xml_bytes)              # parse the XML tree

                    namespace  = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    paragraphs = []

                    for paragraph in root.findall(".//w:p", namespace):   # find all <w:p> paragraph elements
                        text_runs = [
                            node.text or ""
                            for node in paragraph.findall(".//w:t", namespace)  # <w:t> = text run
                        ]
                        joined = "".join(text_runs).strip()   # join all runs in the paragraph
                        if joined:
                            paragraphs.append(joined)

                    contents = "\n".join(paragraphs).strip()   # combine all paragraphs
            finally:
                if os.path.exists(docx_path):
                    os.unlink(docx_path)   # clean up temp DOCX

        else:
            # ── Plain text / code / markdown / CSV etc. ───────────────────────
            contents = raw_bytes.decode("utf-8", errors="replace").strip()
            # errors="replace" substitutes unrecognized bytes with "?" so the app never crashes

    except Exception as e:
        return jsonify({"error": f"Could not read the uploaded file: {e}"}), 400

    if not contents:
        return jsonify({"error": "The uploaded file did not contain readable text."}), 400

    # ── Truncate oversized files ───────────────────────────────────────────────
    was_truncated    = len(contents) > MAX_FILE_CHARS
    trimmed_contents = contents[:MAX_FILE_CHARS].strip()

    if was_truncated:
        trimmed_contents += (
            "\n\n[File truncated — only the beginning was sent to the model "
            "because the upload was too large.]"
        )

    # Build a single prompt combining the optional note and the file content
    request_text = (
        f"{note}\n\nAttached file: {filename}\n\n{trimmed_contents}"
        if note else
        f"Attached file: {filename}\n\n{trimmed_contents}"
    )

    result = _process_text(request_text)
    result["uploaded_file"] = filename         # echo filename back to the UI
    result["file_truncated"] = was_truncated   # let the UI warn the user if truncated
    return jsonify(result)


@app.route("/run_text", methods=["POST"])
def run_text():
    """
    Receives a JSON body { "text": "..." } and runs the intent→tool pipeline.
    This is the simplest route — no file handling needed.
    """
    body = request.get_json(silent=True) or {}   # silent=True returns None (not an error) on bad JSON
    text = body.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400

    return jsonify(_process_text(text))


@app.route("/history")
def history():
    """Returns the full session history as a JSON array for the sidebar Recent list."""
    return jsonify(get_memory())


@app.route("/clear_history", methods=["POST"])
def clear_history_route():
    """Wipes session memory. Called by the Clear History button in the sidebar."""
    clear_memory()
    return jsonify({"status": "cleared"})


# ─────────────────────────────────────────────────────────────────────────────
# SHARED TEXT PIPELINE — called by all three /run_* routes
# ─────────────────────────────────────────────────────────────────────────────
def _process_text(text: str) -> dict:
    """
    Core brain of the app:
      text → detect intent → run tool or chat → save to memory → return result

    Returns a dict with "intent" and "result" keys (both routes call this).
    """
    print("USER INPUT:", text)         # debug log — visible in the terminal

    # Step 1: Ask the intent module what the user wants to do
    intent_data = detect_intent(text)
    print("INTENT:", intent_data)      # debug log

    intent = intent_data.get("intent", "general_chat")  # fall back to chat if intent is unclear

    # Step 2: Run the appropriate handler
    if intent == "general_chat":
        result_text = streaming_chat(text)   # conversational LLM reply
    else:
        result_text = execute_tool(intent_data)  # file creation, folder, summarize, etc.

    # Step 3: Save this interaction to in-memory session history
    add_to_memory({
        "text":   text,
        "intent": intent,
        "result": result_text
    })

    # Step 4: Return structured dict — Flask routes convert this to JSON
    return {
        "intent": intent,
        "result": result_text
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — only runs when you execute: python app.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # A WSGI server (gunicorn, waitress) imports this module directly
    # and does NOT run this block — it's skipped in production deployment.

    try:
        import flask.cli
        flask.cli.show_server_banner = lambda *args, **kwargs: None
        # Suppress the Flask "Serving on..." startup banner to keep the console clean
    except Exception:
        pass

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    # Silence Werkzeug's per-request logs (e.g. "GET / HTTP/1.1 200") in development

    print("Server running at http://localhost:8501")
    # ✅ New (works everywhere)
app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8501)))
    # app.run(debug=True)  # 🔥 Old (debug mode only, not for production)