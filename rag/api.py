import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag.generation import LocalGenerator
from rag.retrieval import ROOT, Retriever, load_documents
from rag.service import SupportAssistant


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must contain text")
        return value.strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = json.loads((ROOT / "rag/config.json").read_text(encoding="utf-8"))
    app.state.assistant = SupportAssistant(
        Retriever(load_documents()), LocalGenerator(), config["threshold"]
    )
    yield


app = FastAPI(title="Fictional Banking RAG Demo", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "knowledge_base": "fictional-demo"}


@app.post("/answer")
def answer(question: Question, request: Request) -> dict:
    return request.app.state.assistant.answer(question.question)
