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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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




# 4. Split Data into Training, Validation, and Test Sets (group-aware, dedupe & conflict detection)
print("\n Step 4: Splitting Data (group-aware on transformed text)")

# We will: (A) detect conflicts (same transformed text -> multiple labels),
# (B) optionally deduplicate exact transformed duplicates (keep-first),
# (C) use GroupShuffleSplit to ensure identical transformed texts do not cross splits.

# Use 'text' as the canonical transformed text column (the model sees this).
df['text_trans'] = df['text'].astype(str).str.strip()

# 1) Detect conflicts: same transformed text mapping to >1 label
conflict_counts = df.groupby('text_trans')['label'].nunique()
conflicts = conflict_counts[conflict_counts > 1]
num_conflicts = len(conflicts)
print(f" Transformed-text conflicts (same transformed text -> multiple labels): {num_conflicts}")

if num_conflicts > 0:
    # Save a small CSV for manual inspection to artifacts (audit)
    try:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        conflict_texts = conflicts.index.tolist()
        conflict_rows = df[df['text_trans'].isin(conflict_texts)][['text', 'text_trans', 'label', 'encoded_label']].drop_duplicates()
        conflict_path = os.path.join(ARTIFACTS_DIR, "transformed_text_conflicts_sample.csv")
        conflict_rows.head(200).to_csv(conflict_path, index=False, encoding="utf-8")
        print(f" -> Saved up to 200 conflicting examples to: {conflict_path}")
    except Exception as e:
        print(" -> Could not save conflicts CSV:", e)
    # NOTE: We do NOT automatically delete these; manual review is recommended.

# 2) Optional: If you prefer to drop exact transformed duplicates (fast), uncomment this line:
# df = df.drop_duplicates(subset=['text_trans'], keep='first').reset_index(drop=True)

# 3) Group-aware splitting so identical transformed strings stay in the same split
groups = df['text_trans'].astype(str)

gss = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=SEED)
train_idx, test_idx = next(gss.split(df, groups=groups))

train_df = df.iloc[train_idx].reset_index(drop=True)
test_df  = df.iloc[test_idx].reset_index(drop=True)

# Now split train_df into train/val with GroupShuffleSplit again (relative size)
val_relative = VALID_RATIO / (1.0 - TEST_RATIO) if (1.0 - TEST_RATIO) > 0 else 0.0
if len(train_df) == 0 or val_relative <= 0:
    print("WARNING: Not enough data to create validation split after test split.")
    X_train = train_df['text_trans'].tolist()
    y_train = train_df['encoded_label'].tolist()
    X_val = []
    y_val = []
else:
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=SEED)
    train_idx2, val_idx = next(gss_val.split(train_df, groups=train_df['text_trans']))
    final_train_df = train_df.iloc[train_idx2].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    X_train = final_train_df['text_trans'].tolist()
    y_train = final_train_df['encoded_label'].tolist()
    X_val   = val_df['text_trans'].tolist()
    y_val   = val_df['encoded_label'].tolist()

# Test split
X_test = test_df['text_trans'].tolist()
y_test = test_df['encoded_label'].tolist()

# 4) Sanity checks: report sizes and ensure no overlap
print(f" Data split successfully (group-aware):")
print(f"   - Training examples:   {len(X_train)}")
print(f"   - Validation examples: {len(X_val)}")
print(f"   - Test examples:       {len(X_test)}")

# Overlap check (should be zero)
train_set = set(X_train)
val_overlap = sum([1 for t in X_val if t in train_set])
test_overlap = sum([1 for t in X_test if t in train_set or t in set(X_val)])
print(f" Overlap train<->val (should be 0): {val_overlap}")
print(f" Overlap train<->(val|test) (should be 0): {test_overlap}")

# ===== ADD THIS GUARD AFTER THE SPLIT BLOCK (right after overlap checks) =====
# Ensure we have training examples — otherwise downstream code (class weights, training) will fail.
if len(X_train) == 0:
    raise RuntimeError(
        "No training examples after group-aware split. "
        "This can happen if TEST_RATIO/VALID_RATIO are too large or grouping collapsed most rows. "
        "Check your data and split settings (TEST_RATIO, VALID_RATIO) and the file 'artifacts/train_snapshot.csv'."
    )
# ========================================================================

# 5) Save snapshots for auditing (optional, useful for reproducibility)
try:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(os.path.join(ARTIFACTS_DIR, "train_snapshot.csv"), index=False, encoding="utf-8")
    pd.DataFrame({"text": X_val,   "label": y_val}).to_csv(os.path.join(ARTIFACTS_DIR, "val_snapshot.csv"), index=False, encoding="utf-8")
    pd.DataFrame({"text": X_test,  "label": y_test}).to_csv(os.path.join(ARTIFACTS_DIR, "test_snapshot.csv"), index=False, encoding="utf-8")
    print(f" Snapshots saved to {ARTIFACTS_DIR} (train/val/test snapshots).")
except Exception as e:
    print(" Could not save snapshots for auditing:", e)




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
    # Safe wrapper: Hugging Face tokenizer can behave oddly on empty lists.
    if texts_list is None:
        texts_list = []
    if len(texts_list) == 0:
        # Return empty tensors with shape (0, max_length) so downstream torch.save calls succeed.
        return {
            "input_ids": torch.empty((0, MAX_TEXT_LENGTH), dtype=torch.long),
            "attention_mask": torch.empty((0, MAX_TEXT_LENGTH), dtype=torch.long)
        }
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
