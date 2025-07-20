# prepare_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import torch
import os

# Load cleaned data
df = pd.read_csv("data/cleaned_all.csv")

# Use only 'text' and 'label'
texts = df["text"].tolist()
labels = df["label"].tolist()

# Encode labels to numeric form
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save label mappings for later use
os.makedirs("artifacts", exist_ok=True)
pd.DataFrame({
    "label": label_encoder.classes_,
    "id": range(len(label_encoder.classes_))
}).to_csv("artifacts/label_mapping.csv", index=False)

# Train/validation/test split (80/10/10)
X_temp, X_test, y_temp, y_test = train_test_split(
    texts, encoded_labels, test_size=0.1, random_state=42, stratify=encoded_labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp
)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Function to tokenize texts
def tokenize_texts(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128
    )

# Tokenize each split
train_encodings = tokenize_texts(X_train)
val_encodings = tokenize_texts(X_val)
test_encodings = tokenize_texts(X_test)

# Save processed datasets as torch tensors
torch.save({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": torch.tensor(y_train)
}, "artifacts/train_data.pt")

torch.save({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": torch.tensor(y_val)
}, "artifacts/val_data.pt")

torch.save({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": torch.tensor(y_test)
}, "artifacts/test_data.pt")

print("✅ Data preparation complete. Tensors saved to 'artifacts/'")
