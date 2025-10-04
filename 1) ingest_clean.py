# 1) ingest_clean.py

import os
import pandas as pd
from preprocess import normalize_placeholders

# Import files from config
from config import (
    RAW_DATA_FILE,
    CLEANED_DATA_FILE,
    DATA_DIR
)

# Load raw input file path config
if not os.path.exists(RAW_DATA_FILE):
    raise FileNotFoundError(f"Raw input file not found: {RAW_DATA_FILE}")

df_raw = pd.read_csv(RAW_DATA_FILE)
print("Columns:", df_raw.columns.tolist())

# Standardize column names that allow multiple variants
if "query" in df_raw.columns:
    df_raw.rename(columns={"query": "text"}, inplace=True)
if "intent" in df_raw.columns:
    df_raw.rename(columns={"intent": "label"}, inplace=True)

# Drop missing, deletes any row that has empty value in the text/label column
df_raw.dropna(subset=["text", "label"], inplace=True)
print(f"shape before cleaning: {df_raw.shape}")

# original raw text is preserved as is for auditing
df_raw["text_raw"] = df_raw["text"].astype(str)

# Apply placeholder normalization which both cleans and replaces tokens
normed_texts = []
mappings_list = []
for t in df_raw["text_raw"].tolist():
    normed, mappings = normalize_placeholders(t)
    normed_texts.append(normed)
    mappings_list.append(mappings)

df_raw["text"] = normed_texts
df_raw["mappings"] = mappings_list  # useful to inspect later

# Normalize labels
df_raw["label"] = df_raw["label"].astype(str).str.lower().str.strip()

print(f"After cleaning: {df_raw.shape[0]} rows")

# Data profiling
counts = df_raw["label"].value_counts()
print("Intent distribution:")
print(counts)

# Save cleaned data using paths from config
os.makedirs(DATA_DIR, exist_ok=True)
df_raw.to_csv(CLEANED_DATA_FILE, index=False, encoding="utf-8")
print(f"Cleaned data saved to: {CLEANED_DATA_FILE}")

