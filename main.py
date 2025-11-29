from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load ALBERT for text classification
classifier = pipeline(
    "text-classification",
    model="textattack/albert-base-v2-SST-2",
    device=-1   # CPU
)

class Item(BaseModel):
    text: str

@app.post("/predict")
def predict(item: Item):
    result = classifier(item.text)[0]
    return {"label": result["label"], "score": float(result["score"])}
