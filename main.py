from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# Request schema
class InputText(BaseModel):
    text: str

# Small sentiment model
model_name = "textattack/albert-base-v2-SST-2"

# IMPORTANT: use_fast=False to avoid tokenizer conversion errors
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
    confidence = probs.max().item()

    return {
        "sentiment": sentiment,
        "confidence": confidence
    }
