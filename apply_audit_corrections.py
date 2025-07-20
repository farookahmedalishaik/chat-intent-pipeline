import pandas as pd

# Load full data and your audit edits
df = pd.read_csv("data/cleaned_all.csv")
audit = pd.read_csv("data/audit_sample.csv")

# Wherever audit['audit_label'] is non‑empty, overwrite df['label']
audit = audit.dropna(subset=["audit_label"])
corrections = dict(zip(audit.index, audit["audit_label"]))
for idx, new_label in corrections.items():
    df.at[idx, "label"] = new_label

# Save gold‑standard CSV & JSON
df.to_csv("data/gold_dataset.csv", index=False)
df.to_json("data/gold_dataset.json", orient="records", lines=True)
print("Gold dataset saved to data/gold_dataset.csv/.json")
