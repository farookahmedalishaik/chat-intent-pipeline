# finetune_bert.py
import torch
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from transformers import IntervalStrategy 
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Load label mapping
label_map = pd.read_csv("artifacts/label_mapping.csv")
num_labels = len(label_map)

# Load tokenized datasets
train_data = torch.load("artifacts/train_data.pt")
val_data = torch.load("artifacts/val_data.pt")

# PyTorch Dataset class
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        
        item = {k: v[idx] for k, v in self.encodings.items()}# returns a single dictionary
        item["labels"] = self.labels[idx] 
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

# Load pretrained BERT for sequence classification & tokenizer here to save it later
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = num_labels
)

# Define training arguments
training_args = TrainingArguments(
    output_dir = "bert_output",
    num_train_epochs = 3,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 32,
    eval_strategy = "epoch", 
    save_strategy ="epoch",
    logging_dir = "bert_logs",
    logging_steps = 50,
    load_best_model_at_end = True,
    metric_for_best_model = "eval_accuracy"
)

# Define compute_metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

# Initialize Trainer
trainer = Trainer(
    model= model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    compute_metrics =compute_metrics
)

# Train & evaluate
trainer.train()
trainer.evaluate()

# Save the best model & tokenizer
model.save_pretrained("artifacts/bert_intent_model")
tokenizer.save_pretrained("artifacts/bert_intent_model")

print("BERT model and tokenizer saved to artifacts/bert_intent_model/")