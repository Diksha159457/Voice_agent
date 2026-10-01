import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeGroq:
    """Minimal stand-in for the Groq SDK: queue chat replies, record calls."""

    def __init__(self):
        self.replies: list[str] = []
        self.calls: list[dict] = []
        self.transcript = "make a folder called voice_notes"
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=self._transcribe))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.replies.pop(0) if self.replies else "ok"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def _transcribe(self, **kwargs):
        self.calls.append({"stt": kwargs["model"]})
        return SimpleNamespace(text=self.transcript)


@pytest.fixture
def no_key(monkeypatch):
    import utils.client as client

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(client, "_client", None)


@pytest.fixture
def fake_llm(monkeypatch):
    import utils.client as client

    fake = FakeGroq()
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(client, "_client", fake)
    return fake


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    import app
    import utils.tools as tools

    out = tmp_path / "output"
    monkeypatch.setattr(tools, "OUTPUT_DIR", str(out))
    monkeypatch.setattr(app, "HISTORY_FILE", str(tmp_path / "history.json"))
    return out


@pytest.fixture
def http(sandbox):
    import app

    app.app.config.update(TESTING=True)
    return app.app.test_client()
