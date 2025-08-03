# simple_api.py
from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)
tokenizer = BertTokenizer.from_pretrained("artifacts/bert_intent_model")
model = BertForSequenceClassification.from_pretrained("artifacts/bert_intent_model")
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text", "")
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**enc).logits
    pred = logits.argmax(dim=-1).item()
    return jsonify({"intent": tokenizer.decode([pred])})

if __name__ == "__main__":
    app.run(port=5000)
