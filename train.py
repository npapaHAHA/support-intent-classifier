import pandas as pd

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
