# export_bert_metrics.py

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertForSequenceClassification, BertTokenizer

# 1. Load tokenized test set
test_data = torch.load("artifacts/test_data.pt")
input_ids     = test_data["input_ids"]
attention_mask= test_data["attention_mask"]
y_true        = test_data["labels"].numpy()

# 2. Load label mapping
label_map = pd.read_csv("artifacts/label_mapping.csv")
labels    = label_map["label"].tolist()

# 3. Load fine‑tuned BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertForSequenceClassification.from_pretrained("artifacts/bert_intent_model")
model.eval()

# 4. Run inference
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits  = outputs.logits
    y_pred  = logits.argmax(dim=-1).numpy()

# 5. Classification report
report_dict = classification_report(
    y_true, y_pred, target_names=labels, output_dict=True
)
df_metrics = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index":"metric"})
df_metrics.to_csv("artifacts/test_metrics_bert.csv", index=False)

# 6. Confusion matrix
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
df_cm.to_csv("artifacts/test_confusion_bert.csv")

print("✅ Saved BERT test metrics to artifacts/test_metrics_bert.csv")
print("✅ Saved BERT confusion matrix to artifacts/test_confusion_bert.csv")
