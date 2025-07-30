#export_bert_metrics.py
import os
import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Load the preprocessed test data from training pipeline / tokenized test set
if not os.path.exists("artifacts/test_data.pt"):
    raise FileNotFoundError("Test data not found - run preprocessing first")
test_data = torch.load("artifacts/test_data.pt")


# Extract test inputs
input_ids = test_data["input_ids"]
attention_mask = test_data["attention_mask"]
y_true = test_data["labels"].numpy()


# Load label mapping
label_mapping = pd.read_csv("artifacts/label_mapping.csv")
labels = label_mapping["label"].tolist()


# Load fine‑tuned BERT / trained model for evaluation
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("artifacts/bert_intent_model")
bert_model.eval()



# Get predictions on test set
with torch.no_grad():
    outputs = bert_model(input_ids = input_ids, attention_mask = attention_mask)
    logits = outputs.logits
    y_pred = logits.argmax(dim=-1).numpy()

# Classification report
report_dict = classification_report(
    y_true, y_pred, target_names =labels, output_dict = True
)
df_metrics = pd.DataFrame(report_dict).T.reset_index().rename(columns = {"index":"metric"})
df_metrics.to_csv("artifacts/test_metrics_bert.csv", index = False)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, index = labels, columns = labels)
df_cm.to_csv("artifacts/test_confusion_bert.csv")

print("BERT test metrics saved to artifacts/test_metrics_bert.csv")
print("Confusion matrix saved to artifacts/test_confusion_bert.csv")
