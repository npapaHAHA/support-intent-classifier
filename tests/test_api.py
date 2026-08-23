import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "example-request.json").read_text(encoding="utf-8"))


class ApiTest(unittest.TestCase):
    def setUp(self):
        model = ROOT / "models/support-intents.joblib"
        self.enterContext(patch.dict(os.environ, {"SUPPORT_MODEL_PATH": str(model)}))
        self.client = self.enterContext(TestClient(app))

    def test_health_after_loading_model(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predict_with_bundled_model(self):
        response = self.client.post("/predict", json=EXAMPLE)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"intent": "card_arrival"})

    def test_predict_normalizes_text(self):
        text = "\t" + EXAMPLE["text"].upper().replace(" ", " \t  ") + "\n"
        response = self.client.post("/predict", json={"text": text})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"intent": "card_arrival"})

    def test_rejects_empty_message(self):
        for text in ("", " \t\n", "\u2003"):
            with self.subTest(text=text):
                response = self.client.post("/predict", json={"text": text})
                self.assertEqual(response.status_code, 422, response.text)

    def test_rejects_invalid_input(self):
        invalid = ({}, {"text": 42}, {"text": "a" * 1001}, {**EXAMPLE, "extra": True})
        for payload in invalid:
            with self.subTest(payload=payload):
                response = self.client.post("/predict", json=payload)
                self.assertEqual(response.status_code, 422, response.text)
