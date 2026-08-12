import argparse
from pathlib import Path

import joblib

from text_features import normalize_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--model", type=Path, default=Path("models/support-intents.joblib"))
    args = parser.parse_args()
    text = normalize_text(args.text)
    if not text:
        parser.error("text must contain at least one non-whitespace character")
    model = joblib.load(args.model)
    print(model.predict([text])[0])


if __name__ == "__main__":
    main()
