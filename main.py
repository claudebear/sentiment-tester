from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}

class InputText(BaseModel):
    text: str

# --- Load ALBERT SST-2 model ---
model_name = "textattack/albert-base-v2-SST-2"   # <-- KEEP THIS EXACT

# Force slow tokenizer to avoid fast-tokenizer errors
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.post("/analyze")
async def analyze(input: InputText):
    text = input.text

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)

    sentiment = "positive" if probs[0][1] > probs[0][0] else "negative"

    return {
        "sentiment": sentiment,
        "confidence": float(probs.max())
    }
