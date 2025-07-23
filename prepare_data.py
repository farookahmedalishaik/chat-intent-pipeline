# prepare_data.py (MODIFIED)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import torch
import os
import shutil # <-- ADD THIS IMPORT for shutil.rmtree
from sqlalchemy import create_engine, text # Added for database connection

# --- NEW: Database connection ---
# Replace with your actual MySQL connection string
conn_str = "mysql+pymysql://intent_user:password:intent_db@localhost:3306/intent_db"
engine = create_engine(conn_str)


# --- NEW: Delete and recreate artifacts folder ---
artifacts_dir = "artifacts"
if os.path.exists(artifacts_dir):
    print(f"Deleting existing '{artifacts_dir}' folder...")
    shutil.rmtree(artifacts_dir) # DANGER: This permanently deletes the folder and its contents!
    print(f"'{artifacts_dir}' folder deleted.")
os.makedirs(artifacts_dir, exist_ok=True) # This will now create a fresh, empty folder
print(f"'{artifacts_dir}' folder ensured (or created).")


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
# Removed os.makedirs(artifacts_dir, exist_ok=True) here as it's done above
pd.DataFrame({
    "label": label_encoder.classes_,
    "id": range(len(label_encoder.classes_))
}).to_csv(os.path.join(artifacts_dir, "label_mapping.csv"), index=False) # Use os.path.join

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
}, os.path.join(artifacts_dir, "train_data.pt")) # Use os.path.join

torch.save({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": torch.tensor(y_val)
}, os.path.join(artifacts_dir, "val_data.pt")) # Use os.path.join

torch.save({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": torch.tensor(y_test)
}, os.path.join(artifacts_dir, "test_data.pt")) # Use os.path.join

print("✅ Data preparation complete. Tensors saved to 'artifacts/'")