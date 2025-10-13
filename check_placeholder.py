import pandas as pd, json
from config import CLEANED_DATA_FILE
from preprocess import normalize_placeholders
df = pd.read_csv(CLEANED_DATA_FILE, dtype=str).fillna("")
df['text_model'] = df['text'].astype(str).apply(lambda t: normalize_placeholders(t)[0])
def has(tok,s): return tok in s.lower()
for label in df['label'].unique():
    sub = df[df['label']==label]
    bill = sub['text_model'].apply(lambda s: has('[bill_present]', s)).sum()
    inv  = sub['text_model'].apply(lambda s: has('[invoice_present]', s)).sum()
    print(label, "count", len(sub), "bill_present", bill, "invoice_present", inv)
