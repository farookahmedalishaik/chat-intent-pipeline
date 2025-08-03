# error_analysis.py
import os
import pandas as pd
import torch

# 1) Paths
ARTIFACTS_DIR = "artifacts"
METRICS_CSV    = os.path.join(ARTIFACTS_DIR, "test_metrics_bert.csv")
CONF_CSV       = os.path.join(ARTIFACTS_DIR, "test_confusion_bert.csv")
TEST_DATA_PT   = os.path.join(ARTIFACTS_DIR, "test_data.pt")
LOW_F1_CSV     = os.path.join(ARTIFACTS_DIR, "low_f1_errors.csv")

# 2) Load metrics & confusion matrix
test_metrics      = pd.read_csv(METRICS_CSV)
confusion_matrix  = pd.read_csv(CONF_CSV, index_col=0)

# 3) Identify lowest-F1 intents
#    assume 'metric' column contains intents and 'f1-score' exists
low_f1 = (
    test_metrics[test_metrics.metric != "accuracy"]
    .nsmallest(3, "f1-score")["metric"]
    .tolist()
)
print("Lowest F1 intents:", low_f1)

# 4) Load raw test set and predictions
data = torch.load(TEST_DATA_PT)
input_ids     = data["input_ids"]
attention_mask = data["attention_mask"]
y_true        = data["labels"].numpy()

# 5) Re-run model inference to get y_pred
from transformers import BertForSequenceClassification, BertTokenizer
tokenizer = BertTokenizer.from_pretrained("artifacts/bert_intent_model")
model     = BertForSequenceClassification.from_pretrained("artifacts/bert_intent_model")
model.eval()

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    y_pred = outputs.logits.argmax(dim=-1).numpy()

# 6) Prepare a DataFrame of errors
#    We need the original texts—load from your cleaned CSV if needed
#    Here we assume you saved test_texts in the PT file; if not, reload from CSV.
try:
    test_texts = data["texts"]
except KeyError:
    # Fallback: reload cleaned CSV and split in same order
    df_all = pd.read_csv("data/cleaned_all.csv")
    # assuming you used consistent train/val/test split
    from sklearn.model_selection import train_test_split
    X_temp, X_test, _, _ = train_test_split(
        df_all.text.tolist(), y_true, test_size=0.1,
        random_state=42, stratify=y_true
    )
    test_texts = X_test

df_errors = pd.DataFrame({
    "text": test_texts,
    "true": y_true,
    "pred": y_pred
})

# 7) Extract top-10 misclassifications per low-F1 intent
rows = []
for intent in low_f1:
    subset = df_errors[
        (df_errors["true"] == intent) &
        (df_errors["pred"]  != intent)
    ].head(10)
    for _, row in subset.iterrows():
        rows.append({"intent": intent, "text": row["text"], "pred": row["pred"]})

df_low_f1 = pd.DataFrame(rows)
df_low_f1.to_csv(LOW_F1_CSV, index=False)
print(f"Saved low-F1 misclassifications to {LOW_F1_CSV}")

# 8) (Optional) Placeholder for augmentation routine
# def augment_texts(texts):
#     # e.g. call a paraphrasing API / back-translation here
#     return [t + " (augmented)" for t in texts]
#
# for intent in low_f1:
#     originals = df_low_f1[df_low_f1.intent==intent]["text"].tolist()
#     augmented = augment_texts(originals)
#     # append to your training CSV or database

