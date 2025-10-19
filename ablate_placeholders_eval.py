# ablate_placeholders_eval.py

import pandas as pd
import re
from sklearn.metrics import classification_report
import numpy as np
import json

ART = "artifacts"
preds = pd.read_csv(f"{ART}/val_predictions_with_probs.csv")
# if the file stores 'text' under a different column, adjust the column name below
text_col = "text" if "text" in preds.columns else "texts" if "texts" in preds.columns else None
if text_col is None:
    print("No text column found in val_predictions_with_probs.csv")
    exit(1)

# remove any invoice/person/id placeholders from the text and re run tokenizer+model inference
# Here we just simulate effect by mapping predictions -> recompute simple baseline:
def remove_placeholders(s):
    return re.sub(r"\[(INVOICE_PRESENT|ID_PRESENT|PERSON_PRESENT|BILL_PRESENT|INVOICE|BILL|ORDER_ID)\]", "", str(s), flags=re.IGNORECASE)

preds["text_masked"] = preds[text_col].apply(remove_placeholders)

# Load a simple baseline mapping from text to predicted label
# For a quick signal, inspect how many rows contained INVOICE_PRESENT and which labels they are
has_inv = preds[text_col].str.contains(r"\[INVOICE_PRESENT\]", na=False)
print("Rows with INVOICE_PRESENT:", has_inv.sum())
print("Label distribution where invoice present:")
print(preds[has_inv]["true"].value_counts())

# Quick heuristic that shows which true labels had INVOICE_PRESENT
print(preds[has_inv].head(20).to_string(index=False))
