# app.py — Enhanced Voice Agent with mic recording, audio preview,
# folder upload, file upload, stop recording, and persistent chat history.

# ── Standard library imports ──────────────────────────────────────────────────
import os          # file/folder ops, env vars, path handling
import json        # serialize chat history to/from JSON for persistent storage
import tempfile    # safe temp files for audio uploads
import zipfile     # open DOCX files (they are ZIP archives internally)
import xml.etree.ElementTree as ET  # parse DOCX XML without needing python-docx
from pathlib import Path             # safe cross-platform filename handling
from datetime import datetime        # timestamp each chat message for display

# ── Third-party imports ───────────────────────────────────────────────────────
from flask import Flask, request, jsonify, render_template_string
# Flask                → creates the WSGI web app
# request              → reads uploaded files, JSON body, form fields from HTTP
# jsonify              → wraps a Python dict into a JSON HTTP response
# render_template_string → renders our HTML string as a full browser page

# ── Our utility modules (all inside utils/) ───────────────────────────────────
from utils.stt    import transcribe_audio   # audio file path → transcribed text
from utils.intent import detect_intent      # text → {"intent", "target", "details"}
from utils.tools  import execute_tool       # intent dict → performs action → result string
from utils.memory import add_to_memory, get_memory, clear_memory
# add_to_memory  → saves one interaction to in-memory session history
# get_memory     → returns the full history list
# clear_memory   → wipes all history

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)                        # create the Flask application
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # allow uploads up to 100 MB

MAX_FILE_CHARS = 12000   # max characters we read from uploaded text/code files
                         # keeps the LLM request within its context/token limit

# ── Persistent chat history file ─────────────────────────────────────────────
HISTORY_FILE = "chat_history.json"
# We store chat history in a local JSON file so it survives server restarts.
# On Render free tier this resets on redeploy — for true persistence use a DB.

def load_persistent_history():
    """Load chat history from the JSON file on disk. Returns [] if file missing."""
    if os.path.exists(HISTORY_FILE):          # only try to read if the file exists
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)           # parse JSON → Python list
        except Exception:
            return []                         # corrupted file — start fresh
    return []                                 # file doesn't exist yet — empty history

def save_persistent_history(history):
    """Write the current history list to the JSON file on disk."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            # ensure_ascii=False → keeps non-ASCII chars (Hindi, emoji) as-is
            # indent=2 → pretty-prints so the file is readable if you open it
    except Exception:
        pass   # if saving fails (e.g. read-only filesystem on cloud), silently continue

# ── Load .env for local development ──────────────────────────────────────────
try:
    from dotenv import load_dotenv   # pip install python-dotenv
    load_dotenv()                    # reads GROQ_API_KEY from .env file in project root
except ImportError:
    pass                             # not installed — use: export GROQ_API_KEY=gsk_...

# ── Optional PDF support ──────────────────────────────────────────────────────
try:
    from pypdf import PdfReader      # pip install pypdf — extracts text from PDFs
except ImportError:
    PdfReader = None                 # None means PDF uploads will show an error message

# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# Everything the browser sees is in this one Python string.
# Flask serves it at http://localhost:8501/
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
    /* ── CSS Reset ───────────────────────────────────────────── */
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    /* ── Design tokens ──────────────────────────────────────── */
    :root {
      --bg-primary:    #1a1a1a;
      --bg-secondary:  #262626;
      --bg-input:      #2f2f2f;
      --bg-message:    #1e1e1e;
      --bg-user:       #2563eb;
      --border:        #3a3a3a;
      --text-primary:  #ececec;
      --text-secondary:#a0a0a0;
      --text-hint:     #6b6b6b;
      --accent:        #d4a574;
      --accent-hover:  #e8b98a;
      --danger:        #f87171;
      --success:       #4ade80;
      --r-sm: 6px; --r-md: 10px; --r-lg: 16px; --r-xl: 24px;
    }

    html, body {
      height: 100%;
      font-family: 'Inter', sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      font-size: 15px;
      line-height: 1.6;
      overflow: hidden;
    }

    .app-shell { display: flex; height: 100vh; }

    /* ── Sidebar ─────────────────────────────────────────────── */
    .sidebar {
      width: 260px; min-width: 260px;
      background: var(--bg-secondary);
      border-right: 1px solid var(--border);
      display: flex; flex-direction: column;
      padding: 20px 0; overflow-y: auto;
    }
    .sidebar-logo {
      display: flex; align-items: center; gap: 10px;
      padding: 0 20px 24px;
      border-bottom: 1px solid var(--border);
    }
    .logo-icon {
      width: 32px; height: 32px; border-radius: 50%;
      background: var(--accent);
      display: flex; align-items: center; justify-content: center;
      font-size: 13px; font-weight: 700; color: #1a1a1a; flex-shrink: 0;
    }
    .logo-text { font-size: 15px; font-weight: 600; letter-spacing: -.02em; }
    .sidebar-label {
      font-size: 11px; font-weight: 500; color: var(--text-hint);
      text-transform: uppercase; letter-spacing: .08em;
      padding: 20px 20px 8px;
    }
    .history-item {
      padding: 8px 20px; font-size: 13px; color: var(--text-secondary);
      border-left: 2px solid transparent; cursor: default;
      transition: background .15s;
    }
    .history-item:hover { background: rgba(255,255,255,.04); }
    .hi-time   { font-size: 10px; color: var(--text-hint); margin-bottom: 2px; }
    .hi-intent { font-size: 11px; color: var(--accent); font-weight: 500; margin-bottom: 2px; }
    .hi-text   { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px; }
    .sidebar-footer {
      margin-top: auto; padding: 12px 20px;
      border-top: 1px solid var(--border);
      display: flex; flex-direction: column; gap: 8px;
    }

    /* ── Main chat area ─────────────────────────────────────── */
    .main-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
    .topbar {
      height: 56px; border-bottom: 1px solid var(--border);
      display: flex; align-items: center; padding: 0 24px; gap: 12px; flex-shrink: 0;
    }
    .topbar-title { font-size: 15px; font-weight: 500; }
    .model-badge {
      font-size: 12px; padding: 3px 10px; border-radius: 20px;
      background: rgba(212,165,116,.12); color: var(--accent);
      border: 1px solid rgba(212,165,116,.25); font-weight: 500;
    }

    /* Blinking "Recording…" pill shown in topbar during mic recording */
    .recording-indicator {
      display: none; align-items: center; gap: 6px;
      font-size: 12px; color: var(--success);
      background: rgba(74,222,128,.1); border: 1px solid rgba(74,222,128,.3);
      padding: 3px 10px; border-radius: 20px; margin-left: auto;
    }
    .recording-indicator.active { display: flex; }
    .rec-dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: var(--success); animation: pulse 1s infinite;
    }
    @keyframes pulse { 0%,100%{opacity:1;}50%{opacity:.3;} }

    .messages-area { flex: 1; overflow-y: auto; padding: 32px 0; }
    .messages-inner { max-width: 700px; margin: 0 auto; padding: 0 24px; }

    /* ── Welcome empty state ─────────────────────────────────── */
    .welcome-state { text-align: center; padding: 60px 20px 40px; }
    .welcome-icon {
      width: 56px; height: 56px; border-radius: 50%;
      background: rgba(212,165,116,.15);
      display: flex; align-items: center; justify-content: center;
      margin: 0 auto 20px; font-size: 24px;
    }
    .welcome-title { font-size: 22px; font-weight: 600; margin-bottom: 10px; }
    .welcome-sub { font-size: 14px; color: var(--text-secondary); max-width: 420px; margin: 0 auto 32px; }
    .suggestions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; max-width: 480px; margin: 0 auto; }
    .suggestion-chip {
      background: var(--bg-secondary); border: 1px solid var(--border);
      border-radius: var(--r-md); padding: 12px 14px;
      text-align: left; font-size: 13px; color: var(--text-secondary);
      cursor: pointer; transition: border-color .15s, background .15s; line-height: 1.4;
    }
    .suggestion-chip:hover { border-color: var(--accent); background: rgba(212,165,116,.06); color: var(--text-primary); }
    .chip-icon { font-size: 16px; margin-bottom: 6px; display: block; }

    /* ── Message bubbles ─────────────────────────────────────── */
    .message { display: flex; gap: 12px; margin-bottom: 24px; animation: fadeIn .2s ease; }
    @keyframes fadeIn { from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:none;} }
    .message.user { flex-direction: row-reverse; }
    .msg-avatar {
      width: 32px; height: 32px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 11px; font-weight: 700; flex-shrink: 0; margin-top: 2px;
    }
    .msg-avatar.agent { background: var(--accent); color: #1a1a1a; }
    .msg-avatar.user  { background: var(--bg-user); color: #fff; }
    .msg-content { max-width: 80%; display: flex; flex-direction: column; gap: 4px; }
    .message.user .msg-content { align-items: flex-end; }
    .msg-bubble { padding: 12px 16px; border-radius: var(--r-lg); font-size: 14px; line-height: 1.65; white-space: pre-wrap; }
    .message.agent .msg-bubble { background: var(--bg-message); border: 1px solid var(--border); border-top-left-radius: 4px; }
    .message.user  .msg-bubble { background: var(--bg-user); border-top-right-radius: 4px; color: #fff; }
    .msg-meta { font-size: 11px; color: var(--text-hint); display: flex; align-items: center; gap: 6px; }
    .intent-tag {
      background: rgba(212,165,116,.12); color: var(--accent);
      padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500;
    }

    /* ── Input area ─────────────────────────────────────────── */
    .input-area { padding: 16px 24px 20px; border-top: 1px solid var(--border); flex-shrink: 0; }
    .input-inner { max-width: 700px; margin: 0 auto; }
    .input-box {
      background: var(--bg-input); border: 1px solid var(--border);
      border-radius: var(--r-xl); padding: 12px 16px; transition: border-color .15s;
    }
    .input-box:focus-within { border-color: rgba(212,165,116,.5); }
    .text-input {
      width: 100%; background: transparent; border: none; outline: none;
      color: var(--text-primary); font-family: 'Inter', sans-serif;
      font-size: 14px; line-height: 1.6; resize: none;
      min-height: 24px; max-height: 200px; overflow-y: auto; display: block;
    }
    .text-input::placeholder { color: var(--text-hint); }
    .input-controls { display: flex; align-items: center; justify-content: space-between; margin-top: 10px; gap: 8px; }
    .input-left { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }

    /* Small action buttons in the input row */
    .action-btn {
      display: flex; align-items: center; gap: 5px;
      padding: 5px 11px; border-radius: var(--r-sm);
      background: rgba(255,255,255,.05); border: 1px solid var(--border);
      color: var(--text-secondary); font-size: 12px; font-weight: 500;
      cursor: pointer; transition: background .15s, border-color .15s, color .15s;
      user-select: none; font-family: 'Inter', sans-serif; white-space: nowrap;
    }
    .action-btn:hover { background: rgba(255,255,255,.09); border-color: rgba(255,255,255,.2); color: var(--text-primary); }
    .action-btn.has-file  { border-color: var(--accent); color: var(--accent); background: rgba(212,165,116,.08); }
    .action-btn.recording { background: rgba(248,113,113,.14); border-color: rgba(248,113,113,.5); color: #fecaca; }

    /* Filename chip shown after a file is selected */
    .file-chip {
      font-size: 11px; color: var(--accent); background: rgba(212,165,116,.1);
      border: 1px solid rgba(212,165,116,.25); border-radius: 12px;
      padding: 2px 8px; max-width: 120px;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      display: none;
    }

    /* Native audio player — shown after file selected or recording complete */
    .audio-preview {
      margin-top: 10px; width: 100%; display: none;
      filter: sepia(.1) saturate(.8);
      border-radius: var(--r-sm);
    }

    /* Circular send button */
    .send-btn {
      width: 36px; height: 36px; border-radius: 50%;
      background: var(--accent); border: none; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      transition: background .15s, transform .1s; flex-shrink: 0;
    }
    .send-btn:hover  { background: var(--accent-hover); }
    .send-btn:active { transform: scale(.94); }
    .send-btn:disabled { background: #3a3a3a; cursor: not-allowed; opacity: .5; }

    .status-bar { font-size: 12px; color: var(--text-hint); margin-top: 8px; text-align: center; min-height: 16px; }

    .btn {
      padding: 7px 14px; border-radius: var(--r-sm); border: 1px solid var(--border);
      background: transparent; color: var(--text-secondary); font-size: 12px;
      font-weight: 500; cursor: pointer; font-family: 'Inter', sans-serif;
      transition: background .15s, color .15s; text-align: center;
    }
    .btn:hover  { background: rgba(255,255,255,.06); color: var(--text-primary); }
    .btn.danger { color: var(--danger); border-color: rgba(248,113,113,.3); }
    .btn.danger:hover { background: rgba(248,113,113,.08); }

    .typing-dots { display: flex; align-items: center; gap: 4px; padding: 4px 0; }
    .typing-dots span {
      width: 6px; height: 6px; border-radius: 50%; background: var(--text-hint);
      animation: bounce 1.2s infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay: .2s; }
    .typing-dots span:nth-child(3) { animation-delay: .4s; }
    @keyframes bounce { 0%,80%,100%{transform:translateY(0);opacity:.4;}40%{transform:translateY(-5px);opacity:1;} }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    @media (max-width: 640px) { .sidebar { display: none; } }
  </style>
</head>
<body>
<div class="app-shell">

  <!-- ══════════════════ SIDEBAR ══════════════════ -->
  <aside class="sidebar">
    <div class="sidebar-logo">
      <div class="logo-icon">VA</div>
      <span class="logo-text">Voice Agent</span>
    </div>
    <div class="sidebar-label">Chat History</div>
    <div id="historyList"></div>  <!-- JS fills this with history items -->
    <div class="sidebar-footer">
      <!-- Downloads history as a .json file -->
      <button class="btn" onclick="exportHistory()" style="width:100%">⬇ Export History</button>
      <!-- Wipes all history from memory and disk -->
      <button class="btn danger" onclick="clearHistory()" style="width:100%">🗑 Clear History</button>
    </div>
  </aside>

  <!-- ══════════════════ MAIN CHAT ══════════════════ -->
  <main class="main-area">
    <div class="topbar">
      <span class="topbar-title">Voice Agent</span>
      <span class="model-badge">Whisper · Groq LLaMA3</span>
      <!-- Shown only while mic recording is active -->
      <div class="recording-indicator" id="recIndicator">
        <div class="rec-dot"></div> Recording…
      </div>
    </div>

    <!-- Scrollable message list -->
    <div class="messages-area" id="messagesArea">
      <div class="messages-inner">

        <!-- Empty state — hidden once first message is sent -->
        <div class="welcome-state" id="welcomeState">
          <div class="welcome-icon">🎙</div>
          <h1 class="welcome-title">How can I help you?</h1>
          <p class="welcome-sub">
            Type, record from your mic, upload audio, attach documents or a whole folder.
            I can create files, write code, summarize, and answer questions.
          </p>
          <div class="suggestions">
            <div class="suggestion-chip" onclick="fillInput('Create a Python file called calculator.py with add and subtract functions')">
              <span class="chip-icon">📄</span>Create a calculator.py
            </div>
            <div class="suggestion-chip" onclick="fillInput('What is machine learning?')">
              <span class="chip-icon">💬</span>What is machine learning?
            </div>
            <div class="suggestion-chip" onclick="fillInput('Make a new folder called my_project')">
              <span class="chip-icon">📁</span>Create a project folder
            </div>
            <div class="suggestion-chip" onclick="fillInput('Summarize this: Python is a high-level language known for simplicity.')">
              <span class="chip-icon">✂️</span>Summarize some text
            </div>
          </div>
        </div>

        <div id="messagesList"></div>  <!-- JS appends bubbles here -->
      </div>
    </div>

    <!-- ── Input panel ─────────────────────────────── -->
    <div class="input-area">
      <div class="input-inner">
        <div class="input-box">

          <!-- Textarea: Enter = send, Shift+Enter = newline -->
          <textarea
            class="text-input" id="textInput" rows="1"
            placeholder="Message Voice Agent… (Enter to send)"
            onkeydown="handleKey(event)"
            oninput="autoResize(this)"
          ></textarea>

          <div class="input-controls">
            <div class="input-left">

              <!-- 🎙 RECORD BUTTON — click to start, click again to stop -->
              <button type="button" class="action-btn" id="recordBtn" onclick="toggleRecording()">
                🎙 <span id="recordLabel">Record</span>
              </button>

              <!-- Hidden audio file input — opened by the label below -->
              <input type="file" id="audioFileInput"
                accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm"
                style="display:none" onchange="onAudioFileSelected(this)" />
              <!-- 🎵 UPLOAD AUDIO BUTTON — opens Finder audio file picker -->
              <label for="audioFileInput" class="action-btn" id="audioUploadBtn">
                🎵 Upload Audio
              </label>

              <!-- Hidden document file input — opened by the label below -->
              <input type="file" id="docFileInput"
                accept=".txt,.md,.py,.js,.ts,.json,.csv,.html,.css,.java,.c,.cpp,.sql,.xml,.yaml,.yml,.pdf,.docx"
                style="display:none" onchange="onDocFileSelected(this)" />
              <!-- 📎 UPLOAD FILE BUTTON — opens Finder file picker -->
              <label for="docFileInput" class="action-btn" id="docUploadBtn">
                📎 Upload File
              </label>

              <!-- Hidden folder input — webkitdirectory lets user pick a whole folder -->
              <input type="file" id="folderInput"
                webkitdirectory style="display:none" onchange="onFolderSelected(this)" />
              <!-- 📂 UPLOAD FOLDER BUTTON — opens Finder folder picker -->
              <label for="folderInput" class="action-btn" id="folderBtn">
                📂 Upload Folder
              </label>

              <!-- Filename chips — appear after selection, hidden by default -->
              <span class="file-chip" id="audioChip"></span>
              <span class="file-chip" id="docChip"></span>
              <span class="file-chip" id="folderChip"></span>

            </div>

            <!-- Circular send button -->
            <button class="send-btn" id="sendBtn" onclick="handleSend()" disabled title="Send (Enter)">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="#1a1a1a">
                <path d="M2 21l21-9L2 3v7l15 2-15 2z"/>
              </svg>
            </button>
          </div>
        </div>

        <!-- Audio preview player — user can LISTEN before sending -->
        <audio class="audio-preview" id="audioPreview" controls></audio>

        <!-- Status text: "Transcribing…", "Thinking…", "Heard: …" -->
        <div class="status-bar" id="statusBar"></div>
      </div>
    </div>
  </main>
</div>

<script>
// ═════════════════════════════════════════════════════════════════════════════
// DOM REFERENCES — grabbed once at page load, reused everywhere
// ═════════════════════════════════════════════════════════════════════════════
const textInput      = document.getElementById('textInput');
const sendBtn        = document.getElementById('sendBtn');
const audioFileInput = document.getElementById('audioFileInput');
const docFileInput   = document.getElementById('docFileInput');
const folderInput    = document.getElementById('folderInput');
const recordBtn      = document.getElementById('recordBtn');
const recordLabel    = document.getElementById('recordLabel');
const audioPreview   = document.getElementById('audioPreview');
const statusBar      = document.getElementById('statusBar');
const msgList        = document.getElementById('messagesList');
const histList       = document.getElementById('historyList');
const welcomeEl      = document.getElementById('welcomeState');
const recIndicator   = document.getElementById('recIndicator');
const audioChip      = document.getElementById('audioChip');
const docChip        = document.getElementById('docChip');
const folderChip     = document.getElementById('folderChip');

// ── App state ─────────────────────────────────────────────────────────────
let busy            = false;   // true while a server request is running
let mediaRecorder   = null;    // active MediaRecorder (null when not recording)
let recordedChunks  = [];      // audio chunks collected during recording
let recordedBlob    = null;    // final Blob assembled after recording stops
let recordingStream = null;    // mic MediaStream — stored so we can release it cleanly
let audioObjectUrl  = null;    // object URL for the preview player
let selectedDocFile = null;    // currently selected document File object
let folderFiles     = [];      // array of File objects from folder selection

// ═════════════════════════════════════════════════════════════════════════════
// TEXTAREA — grows as user types, up to 200px then scrolls
// ═════════════════════════════════════════════════════════════════════════════
function autoResize(el) {
  el.style.height = 'auto';                                  // reset to get real scrollHeight
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';  // cap at 200px
  refreshSendBtn();
}

// ═════════════════════════════════════════════════════════════════════════════
// SEND BUTTON — enabled only when there is content and app is not busy
// ═════════════════════════════════════════════════════════════════════════════
function refreshSendBtn() {
  const hasText   = textInput.value.trim().length > 0;
  const hasAudio  = audioFileInput.files.length > 0 || recordedBlob !== null;
  const hasDoc    = selectedDocFile !== null;
  const hasFolder = folderFiles.length > 0;
  sendBtn.disabled = busy || (!hasText && !hasAudio && !hasDoc && !hasFolder);
}

// ═════════════════════════════════════════════════════════════════════════════
// MIC RECORDING — toggle start/stop with one button
// Fixed for: HTTP localhost, browser compatibility, permission errors
// ═════════════════════════════════════════════════════════════════════════════
async function toggleRecording() {
  if (busy) return;  // don't start recording during a server request

  // ── Stop if already recording ────────────────────────────────────────────
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();   // triggers the onstop handler below
    setStatus('Finishing recording…');
    return;
  }

  // ── CHECK 1: Browser support ──────────────────────────────────────────────
  // Some older browsers or non-secure contexts don't support MediaDevices API
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus('Your browser does not support mic recording. Use Chrome or Firefox.');
    alert('Mic recording not available. Use Chrome/Firefox on https:// or localhost. Or use Upload Audio button.');
    return;
  }

  // ── CHECK 2: HTTPS / localhost requirement ────────────────────────────────
  const isSecure = (
    location.protocol === 'https:' ||
    location.hostname === 'localhost' ||
    location.hostname === '127.0.0.1'
  );

  if (!isSecure) {
    setStatus('Mic requires HTTPS. Use Upload Audio instead, or open the app on https://');
    alert('Mic blocked: not on HTTPS. Open the app at https:// (Render URL) or http://localhost:8501. Or use Upload Audio.');
    return;
  }

  // ── CHECK 3: Pick a supported MIME type ──────────────────────────────────
  // Different browsers support different audio formats for MediaRecorder.
  // We try common ones in order and use the first one that works.
  const mimeTypes = [
    'audio/webm;codecs=opus',   // Chrome, Edge — best quality
    'audio/webm',               // Chrome, Edge — fallback
    'audio/ogg;codecs=opus',    // Firefox
    'audio/ogg',                // Firefox fallback
    'audio/mp4',                // Safari (iOS/macOS)
    '',                         // empty string = browser default (last resort)
  ];
  const supportedMime = mimeTypes.find(m => m === '' || MediaRecorder.isTypeSupported(m));
  // MediaRecorder.isTypeSupported() returns true if the browser can record in that format

  // ── START RECORDING ───────────────────────────────────────────────────────
  try {
    // Ask the browser for microphone access
    // This shows the browser's permission popup the first time
    recordingStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,   // reduces echo from speakers
        noiseSuppression: true,   // filters background noise
        sampleRate: 16000,        // 16kHz is optimal for Whisper transcription
      }
    });

    // Create the MediaRecorder with the best supported format
    const options = supportedMime ? { mimeType: supportedMime } : {};
    mediaRecorder  = new MediaRecorder(recordingStream, options);
    recordedChunks = [];  // clear any leftover chunks from a previous recording

    // Called repeatedly during recording with chunks of audio data
    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        recordedChunks.push(e.data);  // accumulate chunks
      }
    };

    // Called once when mediaRecorder.stop() completes
    mediaRecorder.onstop = () => {
      // Merge all chunks into a single Blob
      const mimeType = mediaRecorder.mimeType || supportedMime || 'audio/webm';
      recordedBlob   = new Blob(recordedChunks, { type: mimeType });

      // Determine file extension from MIME type for the filename chip
      const ext = mimeType.includes('ogg') ? '.ogg'
                : mimeType.includes('mp4') ? '.mp4'
                : '.webm';

      // Release the microphone — browser stops showing the recording indicator
      if (recordingStream) {
        recordingStream.getTracks().forEach(t => t.stop());
        recordingStream = null;
      }

      // Load into preview player so user can listen before sending
      setAudioPreview(recordedBlob);

      // Update UI back to idle state
      recordBtn.classList.remove('recording');
      recordLabel.textContent  = 'Re-record';         // offer to re-record
      audioChip.textContent    = 'recording' + ext;   // show filename chip
      audioChip.style.display  = 'inline';
      recIndicator.classList.remove('active');         // hide topbar pill
      setStatus('✅ Recording ready — press ▶ above to listen, then Send.');
      refreshSendBtn();
    };

    // Request data every 250ms so we get chunks regularly
    // (without this, some browsers only fire ondataavailable once at the end)
    mediaRecorder.start(250);

    // Clear any previously selected audio file — only one audio source at a time
    audioFileInput.value    = '';
    recordedBlob            = null;
    audioChip.style.display = 'none';
    audioChip.textContent   = '';
    document.getElementById('audioUploadBtn').classList.remove('has-file');

    // Hide the old preview player while new recording is in progress
    audioPreview.pause();
    audioPreview.removeAttribute('src');
    audioPreview.style.display = 'none';

    // Switch button to red "Stop" state
    recordBtn.classList.add('recording');
    recordLabel.textContent = '⏹ Stop';
    recIndicator.classList.add('active');  // show topbar "Recording…" pill
    setStatus('🔴 Recording… speak now, then click ⏹ Stop when done.');
    refreshSendBtn();

  } catch (err) {
    // Handle specific known errors with helpful messages
    recordBtn.classList.remove('recording');
    recordLabel.textContent = 'Record';
    recIndicator.classList.remove('active');

    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      setStatus('Mic blocked. Click the lock icon in address bar, set Mic to Allow, then refresh.');
      alert('Mic blocked. Fix: click the lock icon in your address bar, set Microphone to Allow, refresh the page.');
    } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      // No microphone hardware detected
      setStatus('❌ No microphone found. Plug one in or use Upload Audio instead.');
    } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
      // Mic is being used by another app (Zoom, Teams, etc.)
      setStatus('❌ Mic is in use by another app. Close it and try again.');
    } else {
      // Unknown error — show the raw message for debugging
      setStatus('❌ Recording failed: ' + err.name + ' — ' + err.message);
    }

    // Release stream if we managed to get it before the error
    if (recordingStream) {
      recordingStream.getTracks().forEach(t => t.stop());
      recordingStream = null;
    }
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// AUDIO FILE SELECTED — from Finder via Upload Audio button
// ═════════════════════════════════════════════════════════════════════════════
function onAudioFileSelected(input) {
  if (!input.files.length) return;  // user hit Cancel

  const file = input.files[0];

  // Discard any existing browser recording — file takes priority
  recordedBlob = null;
  recordBtn.classList.remove('recording');
  recordLabel.textContent = 'Record';
  recIndicator.classList.remove('active');

  document.getElementById('audioUploadBtn').classList.add('has-file'); // turn button amber
  audioChip.textContent  = file.name;   // show filename chip
  audioChip.style.display = 'inline';

  setAudioPreview(file);   // load into preview player so user can LISTEN before sending
  setStatus('Audio file ready — listen above, then press Send.');
  refreshSendBtn();
}

// ═════════════════════════════════════════════════════════════════════════════
// DOCUMENT FILE SELECTED — from Finder via Upload File button
// ═════════════════════════════════════════════════════════════════════════════
function onDocFileSelected(input) {
  if (!input.files.length) return;

  selectedDocFile = input.files[0];  // store for sendDoc()

  document.getElementById('docUploadBtn').classList.add('has-file');
  docChip.textContent  = selectedDocFile.name;
  docChip.style.display = 'inline';

  setStatus('File attached: ' + selectedDocFile.name);
  refreshSendBtn();
}

// ═════════════════════════════════════════════════════════════════════════════
// FOLDER SELECTED — from Finder via Upload Folder button
// webkitdirectory gives all files inside the chosen folder as a FileList
// ═════════════════════════════════════════════════════════════════════════════
function onFolderSelected(input) {
  if (!input.files.length) return;

  folderFiles = Array.from(input.files);  // convert FileList to array

  const folderName = folderFiles[0].webkitRelativePath.split('/')[0];  // extract folder name
  document.getElementById('folderBtn').classList.add('has-file');
  folderChip.textContent  = folderName + '/ (' + folderFiles.length + ' files)';
  folderChip.style.display = 'inline';

  setStatus('Folder attached: ' + folderFiles.length + ' files from ' + folderName + '/');
  refreshSendBtn();
}

// ═════════════════════════════════════════════════════════════════════════════
// AUDIO PREVIEW — loads a Blob or File into the <audio> player
// ═════════════════════════════════════════════════════════════════════════════
function setAudioPreview(source) {
  if (audioObjectUrl) {
    URL.revokeObjectURL(audioObjectUrl);  // free the old URL to avoid memory leaks
    audioObjectUrl = null;
  }
  audioObjectUrl           = URL.createObjectURL(source);  // create temp browser URL
  audioPreview.src         = audioObjectUrl;
  audioPreview.style.display = 'block';  // make player visible
  audioPreview.load();                   // reload with new source
}

// ═════════════════════════════════════════════════════════════════════════════
// KEYBOARD — Enter sends, Shift+Enter newline
// ═════════════════════════════════════════════════════════════════════════════
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();                      // prevent real newline
    if (!sendBtn.disabled) handleSend();
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// FILL INPUT — suggestion chips pre-populate the textarea
// ═════════════════════════════════════════════════════════════════════════════
function fillInput(text) {
  textInput.value = text;
  autoResize(textInput);
  textInput.focus();
}

// ═════════════════════════════════════════════════════════════════════════════
// MAIN SEND HANDLER — routes to the right backend endpoint
// ═════════════════════════════════════════════════════════════════════════════
async function handleSend() {
  if (busy) return;  // guard against double-submit

  const text   = textInput.value.trim();
  const audio  = audioFileInput.files[0] || null;
  const useRec = recordedBlob !== null;

  if (!text && !audio && !useRec && !selectedDocFile && !folderFiles.length) return;

  // Priority order: folder > document > audio > text
  if (folderFiles.length) {
    await sendFolder(folderFiles, text);
  } else if (selectedDocFile) {
    await sendDoc(selectedDocFile, text);
  } else if (audio || useRec) {
    const audioToSend = audio || new File(
      [recordedBlob], 'recording.webm', { type: recordedBlob.type || 'audio/webm' }
    );  // convert Blob to File so FormData can attach it with a filename
    await sendAudio(audioToSend, text);
  } else {
    await sendText(text);
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// SEND AUDIO — POST multipart/form-data to /run_audio
// ═════════════════════════════════════════════════════════════════════════════
async function sendAudio(file, extraText) {
  appendMessage('user', extraText ? '🎵 ' + file.name + '\n' + extraText : '🎵 ' + file.name);
  showTyping();
  setStatus('Transcribing audio with Whisper…');
  setBusy(true);

  const form = new FormData();
  form.append('audio', file, file.name);          // key must match request.files.get("audio")
  if (extraText) form.append('note', extraText);  // optional typed context

  try {
    const res  = await fetch('/run_audio', { method: 'POST', body: form });
    // Don't set Content-Type — browser sets it with the correct multipart boundary
    const data = await res.json();
    handleResponse(data);
  } catch (err) {
    removeTyping();
    appendMessage('agent', '⚠️ Network error: ' + err.message);
  } finally {
    setBusy(false);
    clearInputs();  // clear AFTER fetch so file reference isn't lost mid-request
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// SEND DOCUMENT — POST multipart/form-data to /run_file
// ═════════════════════════════════════════════════════════════════════════════
async function sendDoc(file, extraText) {
  appendMessage('user', extraText ? '📎 ' + file.name + '\n' + extraText : '📎 ' + file.name);
  showTyping();
  setStatus('Reading file…');
  setBusy(true);

  const form = new FormData();
  form.append('file', file, file.name);           // key must match request.files.get("file")
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

// ═════════════════════════════════════════════════════════════════════════════
// SEND FOLDER — reads text files client-side then sends to /run_text
// Browser cannot send a real folder — we read each file using FileReader API
// ═════════════════════════════════════════════════════════════════════════════
async function sendFolder(files, extraText) {
  const folderName = files[0].webkitRelativePath.split('/')[0];
  appendMessage('user', '📂 ' + folderName + '/ (' + files.length + ' files)' + (extraText ? '\n' + extraText : ''));
  showTyping();
  setStatus('Reading folder files…');
  setBusy(true);

  // Only read files with text-readable extensions
  const textExts = ['txt','md','py','js','ts','jsx','tsx','json','csv','html','css','java','c','cpp','sql','xml','yaml','yml'];
  const readable = files.filter(f => textExts.includes(f.name.split('.').pop().toLowerCase()));

  if (!readable.length) {
    removeTyping();
    appendMessage('agent', '⚠️ No readable text/code files found in the selected folder.');
    setBusy(false);
    clearInputs();
    return;
  }

  // Read each text file and combine into one big prompt
  let combined = extraText ? extraText + '\n\n' : '';
  combined += 'Folder: ' + folderName + '/\n';   // no backticks — string concat instead

  for (const file of readable.slice(0, 10)) {  // limit to 10 files to avoid huge payloads
    const content = await readFileAsText(file);
    combined += '\n--- ' + file.webkitRelativePath + ' ---\n' + content.slice(0, 2000) + '\n';
    // Each file capped at 2000 chars to stay within LLM context budget
  }

  if (readable.length > 10) {
    combined += '\n[' + (readable.length - 10) + ' more files not shown — only first 10 included]';
  }

  try {
    const res  = await fetch('/run_text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: combined.slice(0, 12000) })  // hard cap at 12000 chars
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

// Helper: wraps FileReader in a Promise so we can use await
function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = e => resolve(e.target.result);  // resolve with file text content
    reader.onerror = () => reject(new Error('Could not read ' + file.name));
    reader.readAsText(file, 'utf-8');  // read as UTF-8 encoded text
  });
}

// ═════════════════════════════════════════════════════════════════════════════
// SEND TEXT — POST JSON to /run_text
// ═════════════════════════════════════════════════════════════════════════════
async function sendText(text) {
  appendMessage('user', text);
  showTyping();
  setStatus('Thinking…');
  setBusy(true);

  try {
    const res = await fetch('/run_text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },  // tell Flask to expect JSON
      body: JSON.stringify({ text }),                    // serialize the text field
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

// ═════════════════════════════════════════════════════════════════════════════
// HANDLE SERVER RESPONSE
// ═════════════════════════════════════════════════════════════════════════════
function handleResponse(data) {
  removeTyping();   // remove bouncing dots
  setStatus('');    // clear status text

  if (data.error) {
    appendMessage('agent', '⚠️ ' + data.error);
  } else {
    appendMessage('agent', data.result, data.intent);
    if (data.transcript) {
      setStatus('Heard: "' + data.transcript + '"');  // show Whisper transcription
    }
  }
  loadHistory();   // refresh sidebar
}

// ═════════════════════════════════════════════════════════════════════════════
// APPEND MESSAGE BUBBLE
// ═════════════════════════════════════════════════════════════════════════════
function appendMessage(role, text, intent) {
  welcomeEl.style.display = 'none';  // hide empty state on first message

  const wrap = document.createElement('div');
  wrap.className = 'message ' + role;

  const pill = intent ? '<span class="intent-tag">' + escHtml(intent) + '</span>' : '';
  const now  = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  wrap.innerHTML =
    '<div class="msg-avatar ' + role + '">' + (role === 'agent' ? 'VA' : 'U') + '</div>' +
    '<div class="msg-content">' +
      '<div class="msg-bubble">' + escHtml(text) + '</div>' +
      '<div class="msg-meta">' + now + ' ' + pill + '</div>' +
    '</div>';

  msgList.appendChild(wrap);
  document.getElementById('messagesArea').scrollTop = 99999;  // scroll to newest
}

// ═════════════════════════════════════════════════════════════════════════════
// TYPING INDICATOR
// ═════════════════════════════════════════════════════════════════════════════
function showTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'message agent';
  wrap.id = 'typingMsg';
  wrap.innerHTML =
    '<div class="msg-avatar agent">VA</div>' +
    '<div class="msg-content">' +
      '<div class="msg-bubble" style="padding:14px 16px">' +
        '<div class="typing-dots"><span></span><span></span><span></span></div>' +
      '</div>' +
    '</div>';
  msgList.appendChild(wrap);
  document.getElementById('messagesArea').scrollTop = 99999;
}

function removeTyping() {
  const el = document.getElementById('typingMsg');
  if (el) el.remove();  // remove dots when real response arrives
}

// ═════════════════════════════════════════════════════════════════════════════
// SIDEBAR HISTORY — fetch from /history and render newest-first with timestamps
// ═════════════════════════════════════════════════════════════════════════════
async function loadHistory() {
  try {
    const data = await (await fetch('/history')).json();  // GET /history → array
    histList.innerHTML = '';

    if (!data.length) {
      histList.innerHTML = '<div style="padding:12px 20px;font-size:13px;color:var(--text-hint)">No history yet</div>';
      return;
    }

    [...data].reverse().forEach(item => {  // reverse to show newest first
      const div = document.createElement('div');
      div.className = 'history-item';
      const time = item.timestamp
        ? new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        : '';
      div.innerHTML =
        '<div class="hi-time">'   + time + '</div>' +
        '<div class="hi-intent">' + escHtml(item.intent || 'general_chat') + '</div>' +
        '<div class="hi-text" title="' + escHtml(item.text) + '">' + escHtml(item.text) + '</div>';
      histList.appendChild(div);
    });
  } catch (_) {}  // silently fail — history is non-critical
}

// ═════════════════════════════════════════════════════════════════════════════
// EXPORT HISTORY — download as .json file
// ═════════════════════════════════════════════════════════════════════════════
async function exportHistory() {
  try {
    const data = await (await fetch('/history')).json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);   // create download URL
    const a    = document.createElement('a');
    a.href     = url;
    a.download = 'voice_agent_history.json';  // suggested download filename
    a.click();                                // trigger download
    URL.revokeObjectURL(url);                 // free the object URL
  } catch (err) {
    setStatus('Export failed: ' + err.message);
  }
}

// ═════════════════════════════════════════════════════════════════════════════
// CLEAR HISTORY
// ═════════════════════════════════════════════════════════════════════════════
async function clearHistory() {
  if (!confirm('Clear all chat history? This cannot be undone.')) return;
  await fetch('/clear_history', { method: 'POST' });  // clears memory + disk file
  msgList.innerHTML       = '';
  welcomeEl.style.display = '';  // show empty state again
  histList.innerHTML      = '';
  setStatus('History cleared.');
}

// ═════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═════════════════════════════════════════════════════════════════════════════
function setStatus(msg) { statusBar.textContent = msg; }

function setBusy(val) {
  busy = val;
  refreshSendBtn();
}

function clearInputs() {
  textInput.value = '';
  textInput.style.height = 'auto';

  // Reset audio
  audioFileInput.value = '';
  recordedBlob = null;
  document.getElementById('audioUploadBtn').classList.remove('has-file');
  audioChip.style.display = 'none';
  audioChip.textContent   = '';

  // Reset doc
  docFileInput.value = '';
  selectedDocFile    = null;
  document.getElementById('docUploadBtn').classList.remove('has-file');
  docChip.style.display = 'none';
  docChip.textContent   = '';

  // Reset folder
  folderInput.value = '';
  folderFiles       = [];
  document.getElementById('folderBtn').classList.remove('has-file');
  folderChip.style.display = 'none';
  folderChip.textContent   = '';

  // Release preview player
  if (audioObjectUrl) {
    URL.revokeObjectURL(audioObjectUrl);  // free memory
    audioObjectUrl = null;
  }
  audioPreview.pause();
  audioPreview.removeAttribute('src');
  audioPreview.style.display = 'none';

  // Reset record button
  recordBtn.classList.remove('recording');
  recordLabel.textContent = 'Record';
  recIndicator.classList.remove('active');

  refreshSendBtn();
}

// XSS prevention — escape special chars before inserting text into DOM
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')   // always first — prevents double-encoding
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Load sidebar history on first page open
loadHistory();
</script>
</body>
</html>
"""

# ═════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main page HTML."""
    return render_template_string(HTML)


@app.route("/run_audio", methods=["POST"])
def run_audio():
    """
    Receives audio (recorded blob or uploaded file), transcribes with Whisper,
    then passes transcript through the text pipeline.
    """
    audio = request.files.get("audio")   # None if field missing
    if not audio:
        return jsonify({"error": "No audio file received."}), 400

    note = request.form.get("note", "").strip()  # optional typed context

    # Preserve original file extension — ffmpeg needs it to detect format
    original_name = audio.filename or "upload.wav"
    _, ext        = os.path.splitext(original_name)
    ext           = ext.lower() if ext else ".wav"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            audio.save(tmp.name)   # write upload bytes to disk
            tmp_path = tmp.name

        transcript = transcribe_audio(tmp_path)   # Whisper → text

    except Exception as e:
        return jsonify({"error": f"Transcription failed: {e}. Is ffmpeg installed?"}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)   # always delete temp file

    if not transcript.strip():
        return jsonify({"error": "No speech detected. Speak more clearly and try again."}), 400

    request_text = f"{note}\n\nSpoken: {transcript}" if note else transcript
    result = _process_text(request_text)
    result["transcript"] = transcript   # return transcript so UI shows "Heard: …"
    return jsonify(result)


@app.route("/run_file", methods=["POST"])
def run_file():
    """
    Receives a document upload (PDF, DOCX, or text/code file),
    extracts text, and runs through the text pipeline.
    """
    uploaded = request.files.get("file")
    if not uploaded:
        return jsonify({"error": "No file received."}), 400

    filename  = Path(uploaded.filename or "file.txt").name   # safe base filename
    suffix    = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    note      = request.form.get("note", "").strip()
    raw_bytes = uploaded.read()

    if not raw_bytes:
        return jsonify({"error": "The uploaded file is empty."}), 400

    try:
        if suffix == "pdf":
            if PdfReader is None:
                return jsonify({"error": "PDF support unavailable. Install: pip install pypdf"}), 500
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(raw_bytes)
                pdf_path = tmp.name
            try:
                reader   = PdfReader(pdf_path)
                contents = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
            finally:
                if os.path.exists(pdf_path): os.unlink(pdf_path)

        elif suffix == "docx":
            # DOCX = ZIP file containing XML — no python-docx needed
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(raw_bytes)
                docx_path = tmp.name
            try:
                with zipfile.ZipFile(docx_path) as z:
                    xml  = z.read("word/document.xml")
                    root = ET.fromstring(xml)
                    ns   = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    paras = []
                    for p in root.findall(".//w:p", ns):           # each <w:p> = paragraph
                        runs = "".join(t.text or "" for t in p.findall(".//w:t", ns))
                        if runs.strip(): paras.append(runs.strip())
                    contents = "\n".join(paras)
            finally:
                if os.path.exists(docx_path): os.unlink(docx_path)

        else:
            # Plain text / code / markdown / CSV
            contents = raw_bytes.decode("utf-8", errors="replace").strip()

    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    if not contents:
        return jsonify({"error": "File contained no readable text."}), 400

    # Truncate if too large for the LLM context window
    was_truncated = len(contents) > MAX_FILE_CHARS
    contents      = contents[:MAX_FILE_CHARS]
    if was_truncated:
        contents += "\n\n[Truncated — only the first portion was sent to the model.]"

    request_text = f"{note}\n\nFile: {filename}\n\n{contents}" if note else f"File: {filename}\n\n{contents}"
    result = _process_text(request_text)
    result["file_truncated"] = was_truncated
    return jsonify(result)


@app.route("/run_text", methods=["POST"])
def run_text():
    """Receives { "text": "…" } and runs the full intent → tool pipeline."""
    body = request.get_json(silent=True) or {}   # silent=True: no error on bad JSON
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    return jsonify(_process_text(text))


@app.route("/history")
def history():
    """Returns persistent chat history as JSON for the sidebar."""
    return jsonify(load_persistent_history())   # reads from disk every time


@app.route("/clear_history", methods=["POST"])
def clear_history_route():
    """Wipes in-memory session history and the on-disk JSON file."""
    clear_memory()                  # clear utils/memory.py in-memory list
    save_persistent_history([])     # overwrite file with empty list
    return jsonify({"status": "cleared"})


# ═════════════════════════════════════════════════════════════════════════════
# SHARED TEXT PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def _process_text(text: str) -> dict:
    """
    Core brain: text → intent → tool/chat → save to memory + disk → return result.
    Called by all three /run_* routes.
    """
    intent_data = detect_intent(text)               # LLM classifies the request
    result_text = execute_tool(intent_data)          # runs the appropriate action

    entry = {
        "text":      text,
        "intent":    intent_data.get("intent", "general_chat"),
        "result":    result_text,
        "timestamp": datetime.now().isoformat(),     # ISO timestamp for sidebar display
    }

    add_to_memory(entry)                             # in-memory session history

    # Persist to disk so history survives server restarts
    hist = load_persistent_history()
    hist.append(entry)
    hist = hist[-200:]          # cap at 200 entries to keep file size reasonable
    save_persistent_history(hist)

    return {"intent": entry["intent"], "result": result_text}


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n✅ Voice Agent running → http://localhost:8501\n")
    app.run(
        debug=False,
        host="0.0.0.0",                              # accept connections from any interface
        port=int(os.environ.get("PORT", 8501))       # Render sets PORT; default 8501 locally
    )