import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
from utils.intent import detect_intent


class IntentTests(unittest.TestCase):
    def test_create_folder_intent_is_dispatcher_compatible(self):
        intent = detect_intent("Make a new folder called my_project")

        self.assertEqual(intent["intent"], "create_file")
        self.assertEqual(intent["target"], "my_project")
        self.assertEqual(intent["details"], "folder")

    def test_code_generation_intent_is_not_misclassified(self):
        intent = detect_intent(
            "Create a Python file called calculator.py with add and subtract functions"
        )

        self.assertEqual(intent["intent"], "write_code")
        self.assertEqual(intent["target"], "calculator.py")

    def test_summarize_intent_keeps_text_to_summarize(self):
        intent = detect_intent("Summarize this: Python is readable and productive.")

        self.assertEqual(intent["intent"], "summarize")
        self.assertEqual(intent["details"], "Python is readable and productive.")


class FlaskTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.history_path = Path(self.tmpdir.name) / "history.json"
        self.output_dir = Path(self.tmpdir.name) / "output"

        self.history_patcher = patch.object(app, "HISTORY_FILE", str(self.history_path))
        self.history_patcher.start()

        import utils.tools as tools

        self.output_patcher = patch.object(tools, "OUTPUT_DIR", str(self.output_dir))
        self.output_patcher.start()

        app.app.config.update(TESTING=True)
        self.client = app.app.test_client()

    def tearDown(self):
        self.output_patcher.stop()
        self.history_patcher.stop()
        self.tmpdir.cleanup()

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_run_text_can_create_folder_without_api_key(self):
        old_key = os.environ.pop("GROQ_API_KEY", None)
        try:
            response = self.client.post(
                "/run_text",
                json={"text": "Make a folder called demo_project"},
            )
        finally:
            if old_key is not None:
                os.environ["GROQ_API_KEY"] = old_key

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["intent"], "create_file")
        self.assertTrue((self.output_dir / "demo_project").is_dir())

    def test_run_text_rejects_empty_text(self):
        response = self.client.post("/run_text", json={"text": "  "})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main()
