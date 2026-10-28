from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logistic_regression_model.joblib"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.joblib"
ENCODER_PATH = BASE_DIR / "label_encoder.joblib"

app = FastAPI(
    title="TextLens API",
    description="Classify text with the saved AI and human writing model.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, description="Text to classify.")


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


@lru_cache(maxsize=1)
def load_model_artifacts() -> tuple[Any, Any, Any]:
    missing = [
        path.name
        for path in (MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required model file(s) are missing from the backend directory: "
            + ", ".join(missing)
        )

    return (
        joblib.load(MODEL_PATH),
        joblib.load(VECTORIZER_PATH),
        joblib.load(ENCODER_PATH),
    )


def get_display_label(value: Any, encoder: Any) -> str:
    try:
        return str(encoder.inverse_transform([int(value)])[0])
    except (TypeError, ValueError, IndexError):
        return str(value)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "TextLens text-classification API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    load_model_artifacts()
    return {"status": "ok", "model": "loaded"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be blank.")

    model, vectorizer, encoder = load_model_artifacts()
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    label = get_display_label(prediction, encoder)

    if not hasattr(model, "predict_proba"):
        raise HTTPException(
            status_code=500,
            detail="The configured model does not support probability estimates.",
        )

    probabilities = model.predict_proba(features)[0]
    class_probabilities = {
        get_display_label(class_id, encoder): float(probability)
        for class_id, probability in zip(model.classes_, probabilities)
    }

    return PredictionResponse(
        prediction=label,
        confidence=class_probabilities[label],
        probabilities=class_probabilities,
    )