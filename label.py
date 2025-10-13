import pandas as pd
df = pd.read_csv("artifacts/val_snapshot.csv", dtype=str).fillna("")
# look for label tokens appearing verbatim in text_model
labels = pd.read_csv("artifacts/label_mapping.csv")['label'].tolist()
for lab in labels:
    count = df['text'].str.contains(lab, case=False, na=False).sum()
    if count>0:
        print(lab, "occurs in validation texts:", count)
# Also quickly search for tokens like "request_invoice" substring
print("examples containing 'invoice' in val:")
print(df[df['text'].str.contains("invoice", case=False, na=False)].head(10))
