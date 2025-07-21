# ingest_clean.py (MODIFIED)

import pandas as pd
import os

# --- NEW: Define the single raw input file ---
raw_input_file = "raw_customer_interactions.csv" # Make sure you manually create this file
data_dir = "data"
raw_file_path = os.path.join(data_dir, raw_input_file)

# 1. Load the single raw input file
if not os.path.exists(raw_file_path):
    raise FileNotFoundError(f"Expected raw input file not found: {raw_file_path}. Please create it by combining your original train/val/test CSVs.")
df_raw = pd.read_csv(raw_file_path)

print(f"Columns in {raw_input_file}:", df_raw.columns.tolist())

# 2. Standardize column names (adjust these mappings if your combined raw CSV uses different names)
# Assuming your combined raw file might still have 'query' and 'intent'
df_raw.rename(columns={"query": "text", "intent": "label"}, inplace=True)


# 3. Drop rows with missing text or label
df_raw.dropna(subset=["text", "label"], inplace=True)
print(f"Combined shape (before cleaning): {df_raw.shape}")


# 4. Strip whitespace & lowercase text
df_raw["text"] = df_raw["text"].astype(str).str.strip().str.lower()
df_raw["label"] = df_raw["label"].astype(str).str.strip().str.lower()

# 5. Final shape
print(f"Combined shape (after cleaning): {df_raw.shape}")

# 6. Save cleaned data
cleaned_path = os.path.join(data_dir, "cleaned_all.csv")
df_raw.to_csv(cleaned_path, index=False)
print(f"Cleaned data saved to: {cleaned_path}")