# ingest_clean.py

import pandas as pd
import os

# 1. File paths
data_dir = "data"
files = ["train.csv", "validation.csv", "test.csv"]
paths = [os.path.join(data_dir, f) for f in files]

# 2. Load each split into a DataFrame
dfs = []
for path in paths:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected file not found: {path}")
    df = pd.read_csv(path)
    dfs.append(df)

# 3. Inspect columns of the first file
print("Columns in train.csv:", dfs[0].columns.tolist())

# 4. Standardize column names
#    Adjust these mappings if your CSV uses different column names
for df in dfs:
    df.rename(columns={"query": "text", "intent": "label"}, inplace=True)

# 5. Concatenate all splits
df_all = pd.concat(dfs, ignore_index=True)
print(f"Combined shape (before cleaning): {df_all.shape}")

# 6. Drop rows with missing text or label
df_all.dropna(subset=["text", "label"], inplace=True)

# 7. Strip whitespace & lowercase text
df_all["text"] = df_all["text"].astype(str).str.strip().str.lower()
df_all["label"] = df_all["label"].astype(str).str.strip().str.lower()

# 8. Final shape
print(f"Combined shape (after cleaning): {df_all.shape}")

# 9. Save cleaned data
cleaned_path = os.path.join(data_dir, "cleaned_all.csv")
df_all.to_csv(cleaned_path, index=False)
print(f"Cleaned data saved to: {cleaned_path}")
