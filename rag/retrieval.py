from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Hit:
    document: Document
    score: float


def load_documents(path: Path = ROOT / "rag/data/knowledge.json") -> list[Document]:
    docs = [Document(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
    if not docs or len({doc.id for doc in docs}) != len(docs):
        raise ValueError("Knowledge base must have nonempty, unique document IDs")
    if any(not all((doc.id.strip(), doc.title.strip(), doc.text.strip())) for doc in docs):
        raise ValueError("Knowledge base contains an empty field")
    return docs


class Retriever:
    def __init__(self, documents: list[Document], backend: str = "dense", encoder=None):
        if not documents or len({doc.id for doc in documents}) != len(documents):
            raise ValueError("Expected nonempty documents with unique IDs")
        self.documents = documents
        self.backend = backend
        texts = [f"{doc.title}. {doc.text}" for doc in documents]
        if backend == "tfidf":
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self.matrix = self.vectorizer.fit_transform(texts)
        elif backend == "dense":
            if encoder is None:
                from sentence_transformers import SentenceTransformer

                encoder = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device="cpu")
            self.encoder = encoder
            self.matrix = self.encoder.encode(texts, normalize_embeddings=True)
        else:
            raise ValueError("backend must be dense or tfidf")

    def search(self, query: str, k: int = 3) -> list[Hit]:
        if not query.strip() or not 1 <= k <= len(self.documents):
            raise ValueError("Expected a nonempty query and valid k")
        if self.backend == "tfidf":
            vector = self.vectorizer.transform([query])
            scores = (self.matrix @ vector.T).toarray().ravel()
        else:
            vector = self.encoder.encode([query], normalize_embeddings=True)
            scores = np.asarray(self.matrix @ vector[0]).ravel()
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.documents[int(i)], float(scores[i])) for i in order]
