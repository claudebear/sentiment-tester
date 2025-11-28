from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

class InputText(BaseModel):
    text: str

model_name = "sshleifer/tiny-distilbert-sst2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.post("/analyze")
async def predict(input: InputText):
    text = input.text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    sentiment = "positive" if probs[0][1] > probs[0][0] else "negative"
    return {"sentiment": sentiment, "confidence": probs.max().item()}
