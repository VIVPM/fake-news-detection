"""
FastAPI backend for Fake News Detection using LSTM model (ONNX Runtime).
Models are loaded from Hugging Face Hub.
"""

import json
import os
import re
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from nltk.corpus import stopwords
from pydantic import BaseModel

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── config ──
MAX_SEQUENCE_LENGTH = 100
HF_REPO_ID = os.getenv("HF_REPO_ID", "VPM100/fake-news-detection-model")
HF_TOKEN = os.getenv("HF_TOKEN")

stop = set(stopwords.words("english"))

# ── global holders ──
session = None
word_index = None
num_words = None


def clean_text(txt: str) -> str:
    txt = txt.lower()
    txt = " ".join([word for word in txt.split() if word not in stop])
    txt = re.sub("[^a-z]", " ", txt)
    return txt


def texts_to_sequences(texts: list[str]) -> list[list[int]]:
    """Convert texts to sequences of integers using word_index."""
    sequences = []
    for text in texts:
        seq = []
        for word in text.split():
            idx = word_index.get(word)
            if idx is not None and idx < num_words:
                seq.append(idx)
        sequences.append(seq)
    return sequences


def pad_sequences_manual(sequences: list[list[int]], maxlen: int) -> np.ndarray:
    """Pad sequences to same length."""
    result = np.zeros((len(sequences), maxlen), dtype=np.float32)
    for i, seq in enumerate(sequences):
        trimmed = seq[-maxlen:]
        result[i, maxlen - len(trimmed):] = trimmed
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, word_index, num_words

    print(f"Downloading model artifacts from HF Hub ({HF_REPO_ID})...")
    onnx_path = hf_hub_download(repo_id=HF_REPO_ID, filename="lstm_model.onnx", token=HF_TOKEN)
    tokenizer_path = hf_hub_download(repo_id=HF_REPO_ID, filename="tokenizer.json", token=HF_TOKEN)

    print("Loading ONNX model and tokenizer...")
    session = ort.InferenceSession(onnx_path)

    with open(tokenizer_path, "r") as f:
        tokenizer_data = json.load(f)
    word_index = tokenizer_data["word_index"]
    num_words = tokenizer_data["num_words"]
    print("Model loaded successfully.")

    yield

    session = None
    word_index = None
    num_words = None


app = FastAPI(
    title="Fake News Detection API",
    description="Predict whether a news article is real or fake using a Bidirectional LSTM model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── schemas ──
class NewsInput(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "Breaking: Major policy changes announced by government officials today."}]
        }
    }


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    raw_score: float


class BatchNewsInput(BaseModel):
    articles: list[str]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


# ── helpers ──
def predict(texts: list[str]) -> list[dict]:
    cleaned = [clean_text(t) for t in texts]
    sequences = texts_to_sequences(cleaned)
    padded = pad_sequences_manual(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    input_name = session.get_inputs()[0].name
    scores = session.run(None, {input_name: padded})[0].ravel()

    results = []
    for score in scores:
        label = "Fake" if score >= 0.5 else "Real"
        confidence = float(score) if score >= 0.5 else float(1 - score)
        results.append(
            {"label": label, "confidence": round(confidence, 4), "raw_score": round(float(score), 4)}
        )
    return results


# ── routes ──
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": session is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_single(news: NewsInput):
    if not news.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    return predict([news.text])[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchNewsInput):
    if not batch.articles:
        raise HTTPException(status_code=400, detail="Articles list cannot be empty.")
    return {"predictions": predict(batch.articles)}
