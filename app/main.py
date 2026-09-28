from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.audio import to_mel
from app.model import EmotionCNN, TransferEmotionCNN

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS, UPLOADS = ROOT / "artifacts", ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)
app = FastAPI(title="Voice Emotion CNN")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")

def load_model() -> tuple[EmotionCNN, list[str]]:
    labels_path, weights_path = ARTIFACTS / "labels.json", ARTIFACTS / "model.pt"
    if not labels_path.exists() or not weights_path.exists():
        raise HTTPException(503, "Modelo ainda não treinado. Execute o treinamento descrito no README.")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    metadata = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    model = TransferEmotionCNN(len(labels), pretrained=False) if metadata["architecture"] == "transfer" else EmotionCNN(len(labels))
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True)); model.eval()
    return model, labels

@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")

@app.get("/health")
def health() -> dict[str, bool]:
    return {"trained_model_available": (ARTIFACTS / "model.pt").exists()}

@app.post("/predict")
async def predict(audio: UploadFile = File(...)) -> dict:
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    source = UPLOADS / f"{uuid.uuid4()}{suffix}"
    with source.open("wb") as output: shutil.copyfileobj(audio.file, output)
    try:
        model, labels = load_model(); features = torch.from_numpy(to_mel(source)).unsqueeze(0)
        with torch.inference_mode(): probabilities = torch.softmax(model(features), dim=1)[0].numpy()
        ranked = np.argsort(probabilities)[::-1]
        return {"emotion": labels[int(ranked[0])], "confidence": round(float(probabilities[ranked[0]]), 4), "scores": {labels[int(i)]: round(float(probabilities[i]), 4) for i in ranked}, "spectrogram": features[0, 0].tolist()}
    except RuntimeError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        source.unlink(missing_ok=True)
