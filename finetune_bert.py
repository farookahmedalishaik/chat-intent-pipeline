# finetune_bert.py

import torch
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from transformers import IntervalStrategy # <--- THIS LINE MUST BE PRESENT
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# 1. Load label mapping
label_map = pd.read_csv("artifacts/label_mapping.csv")
num_labels = len(label_map)

# 2. Load tokenized datasets
train_data = torch.load("artifacts/train_data.pt")
val_data    = torch.load("artifacts/val_data.pt")

# 3. Create a PyTorch Dataset class
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels
    def __getitem__(self, idx):
        # Modify this line to return a single dictionary
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx] # Add the label under the 'labels' key
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = IntentDataset(
    {"input_ids": train_data["input_ids"],
     "attention_mask": train_data["attention_mask"]},
    train_data["labels"]
)
val_dataset = IntentDataset(
    {"input_ids": val_data["input_ids"],
     "attention_mask": val_data["attention_mask"]},
    val_data["labels"]
)

# 4. Load pretrained BERT for sequence classification
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels
)

# 5. Define training arguments
training_args = TrainingArguments(
    output_dir="bert_output",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",  # <-- updated argument name!
    save_strategy="epoch",
    logging_dir="bert_logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy"
)

# 6. Define compute_metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 8. Train & evaluate
trainer.train()
trainer.evaluate()

# 9. Save best model & tokenizer
model.save_pretrained("artifacts/bert_intent_model")
print("✅ BERT model saved to artifacts/bert_intent_model/")