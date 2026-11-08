from dataclasses import asdict
import re

from rag.retrieval import Retriever


ABSTENTION = "This demo knowledge base does not cover the question. Please contact support."


class SupportAssistant:
    def __init__(self, retriever: Retriever, generator, threshold: float):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        self.retriever = retriever
        self.generator = generator
        self.threshold = threshold

    def answer(self, question: str, variant: str = "grounded", *, include_raw: bool = False) -> dict:
        hits = self.retriever.search(question, k=min(3, len(self.retriever.documents)))
        if hits[0].score < self.threshold:
            return {"answer": ABSTENTION, "abstained": True, "sources": [], "mode": "abstained"}
        # Keep the small model focused on the best document. Return only context it saw.
        context = hits[:1]
        generated = self.generator.generate(question, context, variant).strip()
        source = context[0].document.text
        allowed_quotes = [source, *re.split(r"(?<=[.!?])\s+", source)]
        is_quote = generated in allowed_quotes
        result = {
            "answer": generated if is_quote else source,
            "abstained": False,
            "sources": [{**asdict(hit.document), "score": hit.score} for hit in context],
            "mode": "verbatim_generation" if is_quote else "retrieved_quote",
        }
        if include_raw:
            result["raw_generation"] = generated
        return result
