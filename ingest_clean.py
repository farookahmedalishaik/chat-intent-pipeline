# ingest_clean.py
import os
import pandas as pd

# raw input file
raw_input_file = "raw_customer_interactions.csv"
data_dir = "data"
raw_file_path = os.path.join(data_dir, raw_input_file)

# Load raw input file
if not os.path.exists(raw_file_path):
    raise FileNotFoundError(f"raw input file not found: {raw_file_path}")

df_raw = pd.read_csv(raw_file_path)
print("Columns:", df_raw.columns.tolist())


# Standardize column names & raw file has 'query' and 'intent' columns
df_raw.rename(columns = {"query": "text", "intent": "label"}, inplace =True)


# Clean data by dropping rows with missing text or label
df_raw.dropna(subset = ["text", "label"], inplace = True)
print(f"shape before cleaning: {df_raw.shape}")


# Strip whitespace & lowercase text
df_raw["text"] = df_raw["text"].astype(str).str.strip().str.lower()
df_raw["label"] = df_raw["label"].astype(str).str.strip().str.lower()

# Final shape
print(f"shape after cleaning: {df_raw.shape}")

# Save cleaned data
cleaned_path = os.path.join(data_dir, "cleaned_all.csv")
df_raw.to_csv(cleaned_path, index = False)
print(f"Cleaned data saved to: {cleaned_path}")