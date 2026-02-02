from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import tempfile
import joblib
import numpy as np
import os


# App Initialization
app = FastAPI()

API_KEY = os.getenv("API_KEY", "sk_test_123456789")
SUPPORTED_LANGUAGES = {"english", "hindi"}

# Load trained ML model
try:
    model = joblib.load("voice_detector.pkl")
except:
    model = None
    print("Model file not found")

# Request Schema

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


# API Endpoint
@app.post("/api/voice-detection")
def detect_voice(
    req: VoiceRequest,
    x_api_key: str = Header(...)
):
    # API Key validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    #  Language validation
    if req.language.lower() not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail="English and Hindi language is supported yet."
        )

    # Audio format check
    if req.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 allowed")

    # Decode Base64 audio
    try:
        audio_bytes = base64.b64decode(req.audioBase64)
    except:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    #  Save temporary MP3 file & extract features
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        from audio_features import extract_audio_features
        features = extract_audio_features(tmp.name)

    # Model prediction
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0]

    classification = "AI_GENERATED" if prediction == 1 else "HUMAN"
    confidence = float(probability[prediction])

    if classification == "AI_GENERATED":
        explanation = "Detected stable pitch patterns and reduced natural variability"
    else:
        explanation = "Detected natural pitch fluctuations and human-like energy variation"

    #  Response
    return {
        "status": "success",
        "language": req.language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
