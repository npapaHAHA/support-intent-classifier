import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from train import load_messages, load_splits


class TrainingDataTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path(self.enterContext(TemporaryDirectory()))

    def test_normalization_removes_duplicate_messages(self):
        path = self.data_dir / "banking-train.csv"
        path.write_text(
            "text,category\nCHECK  CARD,card_arrival\ncheck card,card_arrival\n"
            "  New PIN  ,pin_blocked\n",
            encoding="utf-8",
        )
        messages = load_messages(path)
        self.assertEqual(messages["text"].tolist(), ["new pin"])
        self.assertEqual(messages["category"].tolist(), ["pin_blocked"])

    def test_loads_numeric_messages_as_text(self):
        path = self.data_dir / "banking-train.csv"
        path.write_text("text,category\n0012,pin_blocked\n5678,pin_blocked\n", encoding="utf-8")

        self.assertEqual(load_messages(path)["text"].tolist(), ["0012", "5678"])

    def test_test_split_excludes_messages_seen_in_training(self):
        (self.data_dir / "banking-train.csv").write_text(
            "text,category\nMissing card,card_arrival\n", encoding="utf-8",
        )
        (self.data_dir / "banking-test.csv").write_text(
            "text,category\nMISSING   CARD,card_arrival\nCancel transfer,cancel_transfer\n",
            encoding="utf-8",
        )
        train, test, overlap = load_splits(self.data_dir)
        self.assertEqual(train["text"].tolist(), ["missing card"])
        self.assertEqual(test["text"].tolist(), ["cancel transfer"])
        self.assertEqual(overlap, 1)
