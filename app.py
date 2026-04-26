# app.py — Main entry point for the Voice Agent web application.
# Starts a Flask web server, handles all HTTP routes, and serves
# the Claude-style single-page UI to the browser.

# ── Standard library imports ──────────────────────────────────────────────────
import os          # file deletion, path ops, env vars
import tempfile    # creates safe temporary files for audio uploads

# ── Third-party imports ───────────────────────────────────────────────────────
from flask import Flask, request, jsonify, render_template_string
# Flask               → creates the web application object
# request             → reads incoming HTTP data (files, JSON, form fields)
# jsonify             → converts Python dicts → JSON HTTP responses
# render_template_string → renders an HTML string as a full page response

# ── Our modules (all in utils/) ───────────────────────────────────────────────
from utils.stt    import transcribe_audio  # audio file → text string
from utils.intent import detect_intent    # text → intent dict {"intent","target","details"}
from utils.tools  import execute_tool     # intent dict → action → result string
from utils.memory import (
    add_to_memory,   # save one interaction to session history
    get_memory,      # return full history list
    clear_memory,    # wipe history (Clear History button)
)

# ── App creation ──────────────────────────────────────────────────────────────
app = Flask(__name__)
# Flask(__name__) creates the WSGI application.
# __name__ tells Flask where to find templates/static assets
# (same directory as this file).

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
# Allow uploads up to 50 MB.
# Without this Flask returns a silent 413 for large audio files.
# 50 MB covers ~50 minutes of compressed audio (mp3/m4a).

# ── Optional: auto-load .env file ─────────────────────────────────────────────
# If python-dotenv is installed, reads GROQ_API_KEY from a .env file in the
# project root so you don't have to export it every terminal session.
# Falls back silently if dotenv isn't installed.
try:
    from dotenv import load_dotenv   # pip install python-dotenv  (optional)
    load_dotenv()                    # looks for .env in the current working directory
except ImportError:
    pass  # fine — just use: export GROQ_API_KEY=gsk_...


# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE — Claude.ai-inspired dark UI
# Written as a Python string; no separate templates/ folder needed.
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

    /* ── Design tokens — change these to retheme the whole app ── */
    :root {
      --bg-primary:   #1a1a1a;   /* page background */
      --bg-secondary: #262626;   /* sidebar */
      --bg-input:     #2f2f2f;   /* text-input area */
      --bg-message:   #1e1e1e;   /* agent bubble */
      --bg-user:      #2563eb;   /* user bubble (blue) */
      --border:       #3a3a3a;   /* dividing lines */
      --text-primary: #ececec;   /* main text */
      --text-secondary:#a0a0a0;  /* muted labels */
      --text-hint:    #6b6b6b;   /* placeholders */
      --accent:       #d4a574;   /* warm amber — Claude brand colour */
      --accent-hover: #e8b98a;
      --danger:       #f87171;
      --r-sm: 6px; --r-md: 10px; --r-lg: 16px; --r-xl: 24px;
    }

    html, body {
      height: 100%;
      font-family: 'Inter', sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      font-size: 15px;
      line-height: 1.6;
      overflow: hidden;           /* inner panels scroll, not the body */
    }

    /* ── Three-column shell ─────────────────────────────────── */
    .app-shell { display: flex; height: 100vh; }

    /* ── Sidebar ────────────────────────────────────────────── */
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
    .hi-intent { font-size: 11px; color: var(--accent); font-weight: 500; margin-bottom: 2px; }
    .hi-text {
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px;
    }
    .sidebar-footer {
      margin-top: auto; padding: 16px 20px;
      border-top: 1px solid var(--border);
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

    /* Scrollable message list */
    .messages-area { flex: 1; overflow-y: auto; padding: 32px 0; }
    .messages-inner { max-width: 700px; margin: 0 auto; padding: 0 24px; }

    /* ── Welcome / empty state ─────────────────────────────── */
    .welcome-state { text-align: center; padding: 60px 20px 40px; }
    .welcome-icon {
      width: 56px; height: 56px; border-radius: 50%;
      background: rgba(212,165,116,.15);
      display: flex; align-items: center; justify-content: center;
      margin: 0 auto 20px; font-size: 24px;
    }
    .welcome-title { font-size: 22px; font-weight: 600; letter-spacing: -.02em; margin-bottom: 10px; }
    .welcome-sub { font-size: 14px; color: var(--text-secondary); max-width: 400px; margin: 0 auto 32px; }
    .suggestions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; max-width: 480px; margin: 0 auto; }
    .suggestion-chip {
      background: var(--bg-secondary); border: 1px solid var(--border);
      border-radius: var(--r-md); padding: 12px 14px;
      text-align: left; font-size: 13px; color: var(--text-secondary);
      cursor: pointer; transition: border-color .15s, background .15s; line-height: 1.4;
    }
    .suggestion-chip:hover { border-color: var(--accent); background: rgba(212,165,116,.06); color: var(--text-primary); }
    .chip-icon { font-size: 16px; margin-bottom: 6px; display: block; }

    /* ── Message bubbles ────────────────────────────────────── */
    .message { display: flex; gap: 12px; margin-bottom: 24px; animation: fadeIn .2s ease; }
    @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
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

    .msg-bubble { padding: 12px 16px; border-radius: var(--r-lg); font-size: 14px; line-height: 1.65; }
    .message.agent .msg-bubble {
      background: var(--bg-message); border: 1px solid var(--border);
      border-top-left-radius: 4px;
    }
    .message.user .msg-bubble { background: var(--bg-user); border-top-right-radius: 4px; color: #fff; }

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

    .input-controls {
      display: flex; align-items: center; justify-content: space-between;
      margin-top: 10px; gap: 10px;
    }
    .input-left { display: flex; align-items: center; gap: 8px; }

    /* Upload button (styled <label> over hidden <input type=file>) */
    .upload-btn {
      display: flex; align-items: center; gap: 6px;
      padding: 6px 12px; border-radius: var(--r-sm);
      background: rgba(255,255,255,.05); border: 1px solid var(--border);
      color: var(--text-secondary); font-size: 12px; font-weight: 500;
      cursor: pointer; transition: background .15s, border-color .15s;
      user-select: none;
    }
    .upload-btn:hover { background: rgba(255,255,255,.09); border-color: rgba(255,255,255,.2); color: var(--text-primary); }
    .upload-btn.has-file { border-color: var(--accent); color: var(--accent); background: rgba(212,165,116,.08); }

    .file-name {
      font-size: 12px; color: var(--accent);
      max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }

    /* Send button */
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

    /* Buttons */
    .btn {
      padding: 7px 14px; border-radius: var(--r-sm); border: 1px solid var(--border);
      background: transparent; color: var(--text-secondary); font-size: 12px;
      font-weight: 500; cursor: pointer; font-family: 'Inter', sans-serif;
      transition: background .15s, color .15s;
    }
    .btn:hover { background: rgba(255,255,255,.06); color: var(--text-primary); }
    .btn.danger { color: var(--danger); border-color: rgba(248,113,113,.3); }
    .btn.danger:hover { background: rgba(248,113,113,.08); }

    /* Typing indicator */
    .typing-dots { display: flex; align-items: center; gap: 4px; padding: 4px 0; }
    .typing-dots span {
      width: 6px; height: 6px; border-radius: 50%; background: var(--text-hint);
      animation: bounce 1.2s infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay: .2s; }
    .typing-dots span:nth-child(3) { animation-delay: .4s; }
    @keyframes bounce { 0%,80%,100% { transform:translateY(0); opacity:.4; } 40% { transform:translateY(-5px); opacity:1; } }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

    /* Mobile: hide sidebar */
    @media (max-width: 640px) { .sidebar { display: none; } }
  </style>
</head>
<body>

<div class="app-shell">

  <!-- ══════════════ SIDEBAR ══════════════ -->
  <aside class="sidebar">
    <div class="sidebar-logo">
      <div class="logo-icon">VA</div>
      <span class="logo-text">Voice Agent</span>
    </div>
    <div class="sidebar-label">Recent</div>
    <div id="historyList"></div>
    <div class="sidebar-footer">
      <button class="btn danger" onclick="clearHistory()" style="width:100%">Clear history</button>
    </div>
  </aside>

  <!-- ══════════════ MAIN CHAT ══════════════ -->
  <main class="main-area">

    <div class="topbar">
      <span class="topbar-title">Voice Agent</span>
      <span class="model-badge">Whisper tiny &middot; Groq LLaMA3</span>
    </div>

    <div class="messages-area" id="messagesArea">
      <div class="messages-inner">

        <!-- Empty state — hidden once first message arrives -->
        <div class="welcome-state" id="welcomeState">
          <div class="welcome-icon">🎙</div>
          <h1 class="welcome-title">How can I help you?</h1>
          <p class="welcome-sub">Type a command or upload an audio file. I can create files, write code, summarize text, and answer questions.</p>
          <div class="suggestions">
            <div class="suggestion-chip" onclick="fillInput('Create a Python file called calculator.py with add and subtract functions')">
              <span class="chip-icon">📄</span>Create a calculator.py file
            </div>
            <div class="suggestion-chip" onclick="fillInput('What is machine learning?')">
              <span class="chip-icon">💬</span>What is machine learning?
            </div>
            <div class="suggestion-chip" onclick="fillInput('Make a new folder called my_project')">
              <span class="chip-icon">📁</span>Create a project folder
            </div>
            <div class="suggestion-chip" onclick="fillInput('Summarize this: Python is a high-level programming language known for its simplicity and readability.')">
              <span class="chip-icon">✂️</span>Summarize some text
            </div>
          </div>
        </div>

        <div id="messagesList"></div>
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
            onkeydown="handleKey(event)"
            oninput="autoResize(this)"
          ></textarea>

          <div class="input-controls">
            <div class="input-left">

              <!-- Hidden real file input — triggered by the label below -->
              <!-- IMPORTANT: keep this OUTSIDE the <label> to avoid browser quirks -->
              <input
                type="file" id="audioFile"
                accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac,.webm"
                style="display:none"
                onchange="onFileSelected(this)"
              />

              <!-- Visible styled button — clicking opens the file picker via for="audioFile" -->
              <label for="audioFile" class="upload-btn" id="uploadLabel">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="23"/>
                  <line x1="8"  y1="23" x2="16" y2="23"/>
                </svg>
                <span id="uploadLabelText">Upload audio</span>
              </label>

              <!-- Filename shown after picking; hidden by default -->
              <span class="file-name" id="fileName" style="display:none"></span>

            </div>

            <button class="send-btn" id="sendBtn" onclick="handleSend()" disabled title="Send (Enter)">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="#1a1a1a">
                <path d="M2 21l21-9L2 3v7l15 2-15 2z"/>
              </svg>
            </button>
          </div>
        </div>
        <div class="status-bar" id="statusBar"></div>
      </div>
    </div>

  </main>
</div>


<script>
// ─────────────────────────────────────────────────────────────────────────────
// DOM REFERENCES — grabbed once; reused throughout
// ─────────────────────────────────────────────────────────────────────────────
const textInput   = document.getElementById('textInput');    // the textarea
const sendBtn     = document.getElementById('sendBtn');      // circular send button
const audioFile   = document.getElementById('audioFile');    // hidden <input type=file>
const uploadLabel = document.getElementById('uploadLabel');  // styled label for file input
const fileName    = document.getElementById('fileName');     // filename chip
const statusBar   = document.getElementById('statusBar');    // "Transcribing…" text
const msgList     = document.getElementById('messagesList'); // chat bubble container
const histList    = document.getElementById('historyList');  // sidebar list
const welcomeEl   = document.getElementById('welcomeState'); // empty state block

let busy = false;   // prevents double-submits while a request is in flight

// ─────────────────────────────────────────────────────────────────────────────
// TEXTAREA — auto-resize as user types
// ─────────────────────────────────────────────────────────────────────────────
function autoResize(el) {
  el.style.height = 'auto';                           // reset to measure real scrollHeight
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';  // grow up to 200 px then scroll
  refreshSendBtn();                                   // re-check if send should be enabled
}

// ─────────────────────────────────────────────────────────────────────────────
// SEND BUTTON — enabled only when there is content and no request is in flight
// ─────────────────────────────────────────────────────────────────────────────
function refreshSendBtn() {
  const hasText = textInput.value.trim().length > 0;
  const hasFile = audioFile.files.length > 0;
  sendBtn.disabled = busy || (!hasText && !hasFile);
}

// ─────────────────────────────────────────────────────────────────────────────
// FILE SELECTION — called by onchange on the hidden <input type=file>
// ─────────────────────────────────────────────────────────────────────────────
function onFileSelected(input) {
  if (input.files.length === 0) return;   // user cancelled the picker — nothing to do
  const name = input.files[0].name;       // e.g. "voice_note.m4a"

  // Update the upload label to show a success state
  uploadLabel.classList.add('has-file');
  document.getElementById('uploadLabelText').textContent = '🎵';

  // Show the filename next to the button
  fileName.textContent = name;
  fileName.style.display = 'inline';

  refreshSendBtn();   // file present → enable send
}

// ─────────────────────────────────────────────────────────────────────────────
// KEYBOARD — Enter sends, Shift+Enter inserts newline
// ─────────────────────────────────────────────────────────────────────────────
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();                    // stop browser adding a real newline
    if (!sendBtn.disabled) handleSend();   // only send if button is active
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// FILL INPUT — suggestion chips call this to pre-populate the textarea
// ─────────────────────────────────────────────────────────────────────────────
function fillInput(text) {
  textInput.value = text;
  autoResize(textInput);   // recalculate height now that value was set programmatically
  textInput.focus();
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN SEND HANDLER — routes to audio or text path
// ─────────────────────────────────────────────────────────────────────────────
async function handleSend() {
  if (busy) return;                              // guard: ignore if already processing

  const text = textInput.value.trim();
  const file = audioFile.files[0] || null;      // null if no file selected

  if (!text && !file) return;                    // nothing to send (shouldn't happen)

  // Capture both values NOW before clearInputs() wipes the DOM state
  if (file) {
    await sendAudio(file, text);   // audio takes priority when both present
  } else {
    await sendText(text);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEND AUDIO — POST multipart/form-data to /run_audio
// ─────────────────────────────────────────────────────────────────────────────
async function sendAudio(file, extraText) {
  // Show what the user "said" (filename or typed context)
  appendMessage('user', extraText || '🎵 ' + file.name);
  showTyping();
  setStatus('Transcribing audio…');
  setBusy(true);

  // Build multipart body — the ONLY way to send a File object via fetch
  const form = new FormData();
  form.append('audio', file, file.name);   // key='audio', value=File, filename preserved
  if (extraText) form.append('note', extraText);  // optional typed context

  try {
    const res  = await fetch('/run_audio', { method: 'POST', body: form });
    // Do NOT set Content-Type header — browser sets it automatically with
    // the correct multipart boundary when body is FormData.
    const data = await res.json();          // parse server's JSON response
    handleResponse(data);
  } catch (err) {
    removeTyping();
    appendMessage('agent', '⚠️ Network error: ' + err.message);
  } finally {
    setBusy(false);
    clearInputs();    // clear AFTER fetch so file reference isn't lost mid-request
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SEND TEXT — POST JSON to /run_text
// ─────────────────────────────────────────────────────────────────────────────
async function sendText(text) {
  appendMessage('user', text);
  showTyping();
  setStatus('Thinking…');
  setBusy(true);

  try {
    const res  = await fetch('/run_text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },  // tell Flask we're sending JSON
      body: JSON.stringify({ text }),                    // serialize to JSON string
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
// HANDLE SERVER RESPONSE — update chat after /run_audio or /run_text replies
// ─────────────────────────────────────────────────────────────────────────────
function handleResponse(data) {
  removeTyping();    // remove bouncing dots
  setStatus('');     // clear "Thinking…" text

  if (data.error) {
    appendMessage('agent', '⚠️ ' + data.error);
  } else {
    appendMessage('agent', data.result, data.intent);
    if (data.transcript) {
      // For audio, show what Whisper heard so user can verify
      setStatus('Heard: "' + data.transcript + '"');
    }
  }

  loadHistory();   // refresh sidebar Recent list
}

// ─────────────────────────────────────────────────────────────────────────────
// APPEND MESSAGE BUBBLE
// role:   'user' | 'agent'
// text:   message body
// intent: optional — shown as a coloured pill under the bubble
// ─────────────────────────────────────────────────────────────────────────────
function appendMessage(role, text, intent) {
  welcomeEl.style.display = 'none';   // hide empty state on first message

  const wrap = document.createElement('div');
  wrap.className = 'message ' + role;

  const pill = intent ? `<span class="intent-tag">${intent}</span>` : '';

  wrap.innerHTML = `
    <div class="msg-avatar ${role}">${role === 'agent' ? 'VA' : 'U'}</div>
    <div class="msg-content">
      <div class="msg-bubble">${escHtml(text)}</div>
      <div class="msg-meta">${pill}</div>
    </div>`;

  msgList.appendChild(wrap);

  // Scroll to newest message
  const area = document.getElementById('messagesArea');
  area.scrollTop = area.scrollHeight;
}

// ─────────────────────────────────────────────────────────────────────────────
// TYPING INDICATOR — three bouncing dots while waiting for the server
// ─────────────────────────────────────────────────────────────────────────────
function showTyping() {
  const wrap = document.createElement('div');
  wrap.className = 'message agent';
  wrap.id = 'typingMsg';
  wrap.innerHTML = `
    <div class="msg-avatar agent">VA</div>
    <div class="msg-content">
      <div class="msg-bubble" style="padding:14px 16px">
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </div>
    </div>`;
  msgList.appendChild(wrap);
  document.getElementById('messagesArea').scrollTop = 99999;
}

function removeTyping() {
  const el = document.getElementById('typingMsg');
  if (el) el.remove();
}

// ─────────────────────────────────────────────────────────────────────────────
// SIDEBAR HISTORY — fetch from /history and render
// ─────────────────────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const data = await (await fetch('/history')).json();  // GET /history → array
    histList.innerHTML = '';

    if (!data.length) {
      histList.innerHTML = '<div style="padding:12px 20px;font-size:13px;color:var(--text-hint)">No history yet</div>';
      return;
    }

    // Show newest first
    [...data].reverse().forEach(item => {
      const div = document.createElement('div');
      div.className = 'history-item';
      div.innerHTML = `
        <div class="hi-intent">${item.intent || 'general_chat'}</div>
        <div class="hi-text" title="${escHtml(item.text)}">${escHtml(item.text)}</div>`;
      histList.appendChild(div);
    });
  } catch (_) {}   // silently fail — history is non-critical
}

// ─────────────────────────────────────────────────────────────────────────────
// CLEAR HISTORY
// ─────────────────────────────────────────────────────────────────────────────
async function clearHistory() {
  await fetch('/clear_history', { method: 'POST' });   // wipe server-side memory
  msgList.innerHTML   = '';                            // remove all bubbles
  welcomeEl.style.display = '';                        // show empty state again
  histList.innerHTML  = '';
  setStatus('');
}

// ─────────────────────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────────────────────
function setStatus(msg)  { statusBar.textContent = msg; }

function setBusy(val) {
  busy = val;
  refreshSendBtn();
}

function clearInputs() {
  textInput.value = '';
  textInput.style.height = 'auto';
  audioFile.value = '';                       // clear the file picker
  fileName.style.display   = 'none';
  fileName.textContent     = '';
  uploadLabel.classList.remove('has-file');
  document.getElementById('uploadLabelText').textContent = 'Upload audio';
  refreshSendBtn();
}

// Prevent XSS: convert &, <, >, " to HTML entities before inserting into DOM
function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Load sidebar on page open
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
    """Serve the main page HTML."""
    return render_template_string(HTML)


@app.route("/run_audio", methods=["POST"])
def run_audio():
    """
    Receive an audio file upload, transcribe it, then run it through the
    text pipeline exactly like /run_text does.

    Flow:
      1. Validate the upload exists
      2. Preserve the original file extension (critical — ffmpeg detects
         format from extension, not MIME type)
      3. Write to a temp file and transcribe
      4. Delete temp file (even on error via finally)
      5. Pass transcript to _process_text()
    """
    audio = request.files.get("audio")   # None if key is missing
    if not audio:
        return jsonify({"error": "No audio file received. Make sure the field name is 'audio'."}), 400

    # ── Preserve original extension ───────────────────────────────────────────
    # faster-whisper delegates decoding to ffmpeg.
    # ffmpeg infers codec from the file extension, NOT the Content-Type header.
    # If we always save as ".wav" but the user uploads ".m4a", ffmpeg fails.
    original_name  = audio.filename or "upload.wav"  # browser-sent filename
    _, ext         = os.path.splitext(original_name) # e.g. ".m4a"
    ext            = ext.lower() if ext else ".wav"  # default to .wav if absent

    # ── Write to disk and transcribe ─────────────────────────────────────────
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            # delete=False: we control deletion; required on Windows where open
            # files can't be deleted by another process.
            audio.save(tmp.name)   # stream uploaded bytes to disk
            tmp_path = tmp.name    # save path for use after the 'with' block

        transcript = transcribe_audio(tmp_path)   # Whisper → text string

    except Exception as e:
        return jsonify({
            "error": (
                f"Transcription failed: {e}. "
                "Make sure ffmpeg is installed: brew install ffmpeg  (Mac) "
                "/ sudo apt install ffmpeg  (Linux)"
            )
        }), 500

    finally:
        # Always clean up the temp file — even when an exception was raised
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # ── Process transcript the same way as typed text ─────────────────────────
    result = _process_text(transcript)
    result["transcript"] = transcript   # return what Whisper heard for UI display
    return jsonify(result)


@app.route("/run_text", methods=["POST"])
def run_text():
    """
    Receive a JSON body { "text": "..." } and run the intent→tool pipeline.
    """
    body = request.get_json(silent=True) or {}  # silent=True: returns None on bad JSON
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400
    return jsonify(_process_text(text))


@app.route("/history")
def history():
    """Return session history as a JSON array for the sidebar."""
    return jsonify(get_memory())


@app.route("/clear_history", methods=["POST"])
def clear_history_route():
    """Wipe session memory. Called by the Clear History button."""
    clear_memory()
    return jsonify({"status": "cleared"})


# ─────────────────────────────────────────────────────────────────────────────
# SHARED TEXT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _process_text(text: str) -> dict:
    """
    Core pipeline used by both /run_audio and /run_text.
    text  →  detect_intent()  →  execute_tool()  →  add_to_memory()  →  dict

    Returns dict with keys 'intent' and 'result'.
    """
    intent_data = detect_intent(text)        # ask LLM: what does the user want?
    result_text = execute_tool(intent_data)  # perform the action
    add_to_memory({                          # save to session history
        "text":   text,
        "intent": intent_data.get("intent", ""),
        "result": result_text,
    })
    return {
        "intent": intent_data.get("intent", "general_chat"),
        "result": result_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Only executed when you run  python app.py  directly.
    # A WSGI server (gunicorn, waitress) imports the module instead and
    # runs it differently — this block is skipped in production.
    print("\n  Voice Agent running → http://localhost:8501\n")
    # ✅ New (works everywhere)
app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8501)))

# For production, use a WSGI server like gunicorn or waitress:
#   pip install gunicorn
#   gunicorn app:app 