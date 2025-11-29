from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AlbertTokenizer, AutoModelForSequenceClassification

app = FastAPI()

class InputText(BaseModel):
    text: str

model_name = "textattack/albert-base-v2-SST-2"

# Force slow tokenizer
tokenizer = AlbertTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.post("/analyze")
async def analyze(input: InputText):
    text = input.text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    sentiment = "positive" if probs[0][1] > probs[0][0] else "negative"

    return {
        "sentiment": sentiment,
        "confidence": probs.max().item()
    }
