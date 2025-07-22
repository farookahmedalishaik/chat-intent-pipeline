# prepare_data.py (MODIFIED)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import torch
import os
from sqlalchemy import create_engine, text # Added for database connection

# --- NEW: Database connection ---
# Replace with your actual MySQL connection string
conn_str = "mysql+pymysql://intent_user:password:intent_db@localhost:3306/intent_db"
engine = create_engine(conn_str)

# 1. Load data from MySQL database
print("Loading data from MySQL database...")
with engine.connect() as conn:
    # Use 'text()' for a raw SQL query
    result = conn.execute(text("SELECT text, label FROM messages"))
    # Fetch all rows and convert to a list of dictionaries
    data = result.fetchall()
    # Convert list of tuples/rows to DataFrame
    df = pd.DataFrame(data, columns=result.keys())
print(f"Loaded {len(df)} rows from MySQL for data preparation.")


# Use only 'text' and 'label' (already selected in SQL)
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