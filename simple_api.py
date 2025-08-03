# simple_api.py
from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

app = Flask(__name__)

# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained("artifacts/bert_intent_model")
model = BertForSequenceClassification.from_pretrained("artifacts/bert_intent_model")
model.eval()

# Load the label mapping
label_mapping_df = pd.read_csv("artifacts/label_mapping.csv")
# Create a mapping from ID (0, 1, 2...) back to the label name
id_to_label = {row['id']: row['label'] for index, row in label_mapping_df.iterrows()}

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("text", "")
    
    # Check if text is provided
    if not text:
        return jsonify({"error": "No text provided"}), 400

    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**enc).logits
    
    # Get the predicted ID
    pred_id = logits.argmax(dim=-1).item()
    
    # Use the custom label mapping to get the intent name
    # Get the predicted intent, using a default value if the ID is not found
    pred_intent = id_to_label.get(pred_id, "Unknown intent")
    
    return jsonify({"intent": pred_intent})

if __name__ == "__main__":
    # The debug=True option restarts the server automatically when you change the code.
    # It should be set to False in a production environment.
    app.run(port=5000, debug=True)