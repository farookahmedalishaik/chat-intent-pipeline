# 3) prepare_data.py

"""
- Prepares cleaned data for model training. Reads cleaned human labeled data from MySQL database. 
- Encodes text labels into numbers so the model can use. Splits the data into training, validation, and test sets.
- Calculates class weights to help the model with imbalanced data. Tokenizes the text and saves everything as ready to use PyTorch files.
"""

import os
import shutil
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from utils import get_db_engine
from collections import Counter  # added for safe stratify fallback

# Import all settings from the central config file
from config import (
    ARTIFACTS_DIR,
    CLEANED_DATA_FILE,
    DB_MESSAGES_TABLE,
    TEST_RATIO,
    VALID_RATIO,
    MAX_TEXT_LENGTH,
    SEED,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_DB,
    BASE_MODEL_ID
)

# --- Helper function to load data ---
def load_data_from_source():
    """
    Loads data from the MySQL database (database as the single source of truth)
    """
    try:
        # Use centralized DB engine helper (cloud first, with optional local fallback)
        engine = get_db_engine(
            MYSQL_USER,
            MYSQL_PASSWORD,
            MYSQL_HOST,
            MYSQL_PORT,
            MYSQL_DB,
            sqlite_fallback_path=None,  # strict MySQL behavior for prepare_data.py
            test_connection=True
        )
        df = pd.read_sql(text(f"SELECT text, label FROM {DB_MESSAGES_TABLE}"), con=engine)
        print(f" Successfully loaded {len(df)} rows from DB table '{DB_MESSAGES_TABLE}'.")
        return df
    except Exception as e:
        print(f" CRITICAL: Could not load data from the database. Error: {e}")
        # strict behavior: raise an error and stop (no local fallback for prepare_data.py)
        raise RuntimeError("Failed to connect to the database. The pipeline cannot continue without data.")


# --- Main Script Execution ---

# 1. Prepare Artifacts Directory
# Delete the old artifacts folder to ensure a clean run, then create a new empty one.
print(" Step 1: Preparing artifacts directory")
if os.path.exists(ARTIFACTS_DIR):
    shutil.rmtree(ARTIFACTS_DIR)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
print(f" Clean artifacts directory created at: '{ARTIFACTS_DIR}'")

# 2. Load and Clean the Data
print("\n Step 2: Loading and Cleaning Data")
df = load_data_from_source()

df = df[['text', 'label']].dropna() # Keep only necessary columns and drop rows with any missing values

df['text'] = df['text'].astype(str).str.strip() # Clean up whitespace from text and labels
df['label'] = df['label'].astype(str).str.strip()
print(f" Data loaded & cleaned. Total rows are: {len(df)}")

# 3. Encode Labels
print("\n Step 3: Encoding Labels")

label_encoder = LabelEncoder() # Convert text labels (e.g., "check_balance") into numbers (e.g., 0, 1, 2...)
df['encoded_label'] = label_encoder.fit_transform(df['label'])
num_labels = len(label_encoder.classes_)

# Save the mapping from text to number for later use
label_mapping = pd.DataFrame({
    "label": label_encoder.classes_,
    "id": range(num_labels)
})
label_mapping_path = os.path.join(ARTIFACTS_DIR, "label_mapping.csv")
label_mapping.to_csv(label_mapping_path, index=False)
print(f" Labels encoded into {num_labels} unique classes.")
print(f" Label mapping saved to: '{label_mapping_path}'")

# 4. Split Data into Training, Validation, and Test Sets
print("\n Step 4: Splitting Data")
texts = df['text'].tolist()
labels = df['encoded_label'].tolist()

# Safe stratified split: if any class is too small or only one class exists, fall back to a non stratified random split and print error.
label_counts = Counter(labels)
min_count = min(label_counts.values()) if label_counts else 0

if len(label_counts) <= 1 or min_count < 2:
    print("WARNING: Not enough examples per class to perform a stratified split. "
          "Falling back to a non stratified random split.")
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=TEST_RATIO, random_state=SEED
    )
    val_size_relative_to_temp = VALID_RATIO / (1.0 - TEST_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_relative_to_temp, random_state=SEED
    )
else:
    # First, split off the test set (e.g., TEST_RATIO of the data)
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=TEST_RATIO, random_state=SEED, stratify=labels
    )
    # Then, split the remaining into training and validation sets with stratify
    val_size_relative_to_temp = VALID_RATIO / (1.0 - TEST_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_relative_to_temp, random_state=SEED, stratify=y_temp
    )

print(f" Data split successfully:")
print(f"   - Training examples:   {len(X_train)}")
print(f"   - Validation examples: {len(X_val)}")
print(f"   - Test examples:       {len(X_test)}")

# 5. Compute Class Weights
print("\n Step 5: Computing Class Weights")
# Calculate weights to give more importance to smaller classes during training
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_path = os.path.join(ARTIFACTS_DIR, "class_weights.npy")
np.save(class_weights_path, class_weights)
print(f" Class weights calculated and saved to: '{class_weights_path}'")

# 6. Tokenize and Save Data as Tensors
print("\n Step 6: Tokenizing and Saving Tensors")
tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_ID)

# Helper function to tokenize a list of texts
def tokenize_texts(texts_list):
    return tokenizer(
        texts_list,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=MAX_TEXT_LENGTH
    )

# Tokenize each data split
train_encodings = tokenize_texts(X_train)
val_encodings = tokenize_texts(X_val)
test_encodings = tokenize_texts(X_test)

# Save the tokenized data and labels as PyTorch files
torch.save({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": torch.tensor(y_train, dtype=torch.long)
}, os.path.join(ARTIFACTS_DIR, "train_data.pt"))

torch.save({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": torch.tensor(y_val, dtype=torch.long),
    "texts": X_val  # Include raw text for error analysis
}, os.path.join(ARTIFACTS_DIR, "val_data.pt"))

torch.save({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": torch.tensor(y_test, dtype=torch.long),
    "texts": X_test  # Include raw text for error analysis
}, os.path.join(ARTIFACTS_DIR, "test_data.pt"))

print(" All datasets tokenized and saved as .pt files in the 'artifacts' directory.")
print("\n Data preparation complete!")
