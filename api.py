import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from text_features import normalize_text


class SupportMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(min_length=1, max_length=1000)

    @field_validator("text")
    @classmethod
    def normalize_message(cls, text: str) -> str:
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError("text must contain a non-whitespace character")
        return normalized


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.environ.get("SUPPORT_MODEL_PATH", "models/support-intents.joblib"))
    app.state.model = joblib.load(model_path)
    yield


app = FastAPI(title="Support Intent Classifier", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(message: SupportMessage, request: Request) -> dict[str, str]:
    intent = request.app.state.model.predict([message.text])[0]
    return {"intent": str(intent)}
