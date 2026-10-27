from dataclasses import asdict

from rag.retrieval import Retriever


ABSTENTION = "This demo knowledge base does not cover the question. Please contact support."


class SupportAssistant:
    def __init__(self, retriever: Retriever, generator, threshold: float):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        self.retriever = retriever
        self.generator = generator
        self.threshold = threshold

    def answer(self, question: str, variant: str = "grounded") -> dict:
        hits = self.retriever.search(question, k=min(3, len(self.retriever.documents)))
        if hits[0].score < self.threshold:
            return {"answer": ABSTENTION, "abstained": True, "sources": []}
        answer = self.generator.generate(question, hits, variant)
        return {
            "answer": answer,
            "abstained": False,
            "sources": [{**asdict(hit.document), "score": hit.score} for hit in hits],
        }
