import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from text_features import normalize_text


def load_messages(path):
    messages = pd.read_csv(path, dtype={"text": "string"}).dropna(subset=["text", "category"])
    messages = messages.assign(text=messages["text"].map(normalize_text))
    return messages[messages["text"].ne("")].drop_duplicates(subset="text", keep=False)


def load_splits(data_dir):
    development = load_messages(data_dir / "banking-train.csv")
    published_test = load_messages(data_dir / "banking-test.csv")
    test = published_test[~published_test["text"].isin(development["text"])]
    return development, test, len(published_test) - len(test)


def metrics(expected, predicted):
    return {"macro_f1": f1_score(expected, predicted, average="macro"),
            "accuracy": accuracy_score(expected, predicted)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    development, test, overlap = load_splits(args.data_dir)
    train, validation = train_test_split(
        development, test_size=0.2, random_state=42, stratify=development["category"],
    )
    models = {
        "most_frequent": DummyClassifier(strategy="most_frequent"),
        "word_tfidf": make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LinearSVC(random_state=42)),
        "word_and_char_tfidf": make_pipeline(
            FeatureUnion([
                ("words", TfidfVectorizer(ngram_range=(1, 2))),
                ("characters", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
            ]),
            LinearSVC(random_state=42),
        ),
    }

    validation_scores = {}
    for name, model in models.items():
        model.fit(train["text"], train["category"])
        validation_scores[name] = metrics(validation["category"], model.predict(validation["text"]))
    selected = max(models, key=lambda name: validation_scores[name]["macro_f1"])
    model = models[selected].fit(development["text"], development["category"])
    predictions = model.predict(test["text"])
    labels = sorted(development["category"].unique())
    confusion = confusion_matrix(test["category"], predictions, labels=labels)
    mistakes = sorted(
        ((int(count), labels[i], labels[j]) for i, row in enumerate(confusion)
         for j, count in enumerate(row) if i != j and count),
        reverse=True,
    )
    report = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "test_rows_removed_as_train_overlap": overlap,
        "classes": len(labels),
        "selected_model": selected,
        "validation": validation_scores,
        "test": metrics(test["category"], predictions),
        "top_confusions": [{"actual": actual, "predicted": predicted, "count": count}
                           for count, actual, predicted in mistakes[:5]],
    }
    joblib.dump(model, args.output_dir / "model.joblib", compress=3)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
