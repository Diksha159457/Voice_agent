import io
import json
import os
import zipfile

import pytest

from utils.documents import UnsupportedDocument, extract_text
from utils.history import HistoryStore
from utils.intent import detect_intent
from utils.sandbox import SandboxError, safe_path
from utils.tools import execute_tool, strip_code_fences

# ── intent detection (rule-based, no LLM) ───────────────────────────────────


@pytest.mark.parametrize(
    "text,intent,target",
    [
        ("Make a new folder called my_project", "create_file", "my_project"),
        ("Create a Python file called calculator.py with add and subtract functions", "write_code", "calculator.py"),
        ("Create an empty file notes.txt", "create_file", "notes.txt"),
        ("Summarize this: Python is readable and productive.", "summarize", ""),
    ],
)
def test_rule_based_intents(text, intent, target, no_key):
    result = detect_intent(text)
    assert result["intent"] == intent
    assert result["target"] == target


def test_llm_intent_fallback_is_normalised(fake_llm):
    fake_llm.replies = [json.dumps({"intent": "hack_the_planet", "target": "x"})]
    assert detect_intent("tell me a joke about compilers")["intent"] == "general_chat"


def test_intent_survives_llm_failure(no_key):
    result = detect_intent("what is entropy?")
    assert result["intent"] == "general_chat" and "error" in result


# ── sandbox: the agent must never write outside output/ ─────────────────────


@pytest.mark.parametrize(
    "name",
    ["../escape.txt", "../../etc/passwd", "/etc/passwd", "..", ".", "", "   ", ".env",
     "a\\b.txt", "nul\x00byte.py", "x" * 200, "sub/dir.py", 42, None, "$(rm -rf).sh"],
)
def test_sandbox_rejects_hostile_names(tmp_path, name):
    with pytest.raises(SandboxError):
        safe_path(tmp_path, name)


def test_sandbox_blocks_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    os.symlink(outside / "target.txt", root / "link.txt")
    with pytest.raises(SandboxError, match="outside"):
        safe_path(root, "link.txt")


@pytest.mark.parametrize("name", ["notes.txt", "my project", "calc-v2.py", "README.md"])
def test_sandbox_allows_normal_names(tmp_path, name):
    assert safe_path(tmp_path, name).parent == tmp_path.resolve()


def test_create_file_tool_reports_rejection_instead_of_writing(sandbox, tmp_path):
    msg = execute_tool({"intent": "create_file", "target": "../pwned.txt", "details": "file"})
    assert msg.startswith("⚠️")
    assert not (tmp_path / "pwned.txt").exists()


def test_create_file_and_folder(sandbox):
    execute_tool({"intent": "create_file", "target": "notes.txt", "details": "file"})
    execute_tool({"intent": "create_file", "target": "proj", "details": "folder"})
    assert (sandbox / "notes.txt").is_file() and (sandbox / "proj").is_dir()


def test_type_conflict_is_reported(sandbox):
    execute_tool({"intent": "create_file", "target": "thing", "details": "folder"})
    assert "different type" in execute_tool({"intent": "create_file", "target": "thing", "details": "file"})


# ── code generation ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("```python\nprint('hi')\n```", "print('hi')"),
        ("```\nx = 1\n```", "x = 1"),
        ("print('plain')", "print('plain')"),
    ],
)
def test_strip_code_fences(raw, expected):
    assert strip_code_fences(raw) == expected


def test_write_code_saves_unfenced_code(sandbox, fake_llm):
    fake_llm.replies = ["```python\ndef add(a, b):\n    return a + b\n```"]
    msg = execute_tool({"intent": "write_code", "target": "calc.py", "details": "add two numbers"})
    assert "calc.py" in msg
    assert (sandbox / "calc.py").read_text() == "def add(a, b):\n    return a + b"


def test_write_code_without_key_is_graceful(sandbox, no_key):
    assert "API key is missing" in execute_tool({"intent": "write_code", "target": "a.py"})


# ── document extraction ─────────────────────────────────────────────────────


def _docx(paragraphs: list[str]) -> bytes:
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = f'<w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", xml)
    return buf.getvalue()


def test_extract_docx():
    assert extract_text("a.docx", _docx(["Hello", "World"])) == "Hello\nWorld"


def test_extract_text_file():
    assert extract_text("main.py", b"print(1)\n") == "print(1)"


@pytest.mark.parametrize(
    "name,data",
    [("img.png", b"\x89PNG..."), ("bad.docx", b"not a zip"), ("bin.txt", b"\x00\x01\x02binary")],
)
def test_extract_rejects_unsupported(name, data):
    with pytest.raises(UnsupportedDocument):
        extract_text(name, data)


# ── history store ───────────────────────────────────────────────────────────


def test_history_is_capped_and_recovers_from_corruption(tmp_path):
    store = HistoryStore(tmp_path / "h.json")
    for i in range(250):
        store.append({"i": i})
    entries = store.load()
    assert len(entries) == 200 and entries[-1] == {"i": 249}

    (tmp_path / "h.json").write_text("{corrupt")
    assert store.load() == []


# ── HTTP routes ─────────────────────────────────────────────────────────────


def test_health(http, no_key):
    assert http.get("/health").get_json() == {"status": "ok", "llm_configured": False}


def test_index_serves_ui(http):
    res = http.get("/")
    assert res.status_code == 200 and b"Voice Agent" in res.data


def test_run_text_creates_folder_without_api_key(http, sandbox, no_key):
    res = http.post("/run_text", json={"text": "Make a folder called demo_project"})
    assert res.status_code == 200
    assert res.get_json()["intent"] == "create_file"
    assert (sandbox / "demo_project").is_dir()
    assert http.get("/history").get_json()[-1]["intent"] == "create_file"


@pytest.mark.parametrize("body", [{"text": "  "}, {"text": 5}, {}, [], None])
def test_run_text_rejects_bad_input(http, body):
    res = http.post("/run_text", json=body) if body is not None else http.post("/run_text", data="x")
    assert res.status_code == 400


def test_run_text_accepts_folder_sized_payload(http, no_key):
    # the UI sends up to 12k chars when a folder is attached
    res = http.post("/run_text", json={"text": "Summarize this: " + "a " * 5_900})
    assert res.status_code == 200


def test_run_text_rejects_oversized(http):
    assert http.post("/run_text", json={"text": "a" * 20_000}).status_code == 413


def test_run_file_docx(http, no_key):
    data = {"file": (io.BytesIO(_docx(["Summarize this: tests matter."])), "notes.docx")}
    res = http.post("/run_file", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.get_json()["file_truncated"] is False


def test_run_file_rejects_unsupported(http):
    data = {"file": (io.BytesIO(b"\x89PNG"), "photo.png")}
    assert http.post("/run_file", data=data, content_type="multipart/form-data").status_code == 415


def test_run_audio_transcribes_and_runs_pipeline(http, sandbox, fake_llm):
    data = {"audio": (io.BytesIO(b"RIFF....WAVE"), "clip.wav")}
    res = http.post("/run_audio", data=data, content_type="multipart/form-data")
    body = res.get_json()
    assert res.status_code == 200, body
    assert body["transcript"] == "make a folder called voice_notes"
    assert (sandbox / "voice_notes").is_dir()


def test_run_audio_without_key_is_503(http, no_key):
    data = {"audio": (io.BytesIO(b"x"), "clip.wav")}
    assert http.post("/run_audio", data=data, content_type="multipart/form-data").status_code == 503


def test_run_audio_rejects_unknown_format(http, fake_llm):
    data = {"audio": (io.BytesIO(b"x"), "clip.exe")}
    assert http.post("/run_audio", data=data, content_type="multipart/form-data").status_code == 415
