# finetune_bert.py
import torch
import random
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from transformers import IntervalStrategy 
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


# 1) Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# 2) Hyperparameters (we’ll log these)
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 3


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
model     = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

# Log hyperparams to a simple markdown
with open("bert_logs/hyperparameters.md", "w") as f:
    f.write(f"""# Hyperparameter Log

- seed: {SEED}  
- learning_rate: {LEARNING_RATE}  
- batch_size: {BATCH_SIZE}  
- epochs: {EPOCHS}  
""")
    

# Define training arguments
training_args = TrainingArguments(
    output_dir="bert_output",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=2*BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="bert_logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
)

# Define compute_metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)


#trainer = Trainer(
#    model=model,
#    args=training_args,
#   train_dataset=IntentDataset(train_data["encodings"], train_data["labels"]),
#   eval_dataset= IntentDataset(val_data["encodings"], val_data["labels"]),
#    compute_metrics=compute_metrics,
#)

# Train & evaluate
#trainer.train()
#trainer.evaluate()


# Train & evaluate
trainer.train()
eval_metrics = trainer.evaluate()



# Append final eval metrics to our log
with open("bert_logs/training_results.md", "w") as f:
    f.write("## Final Evaluation Metrics\n")
    for k, v in eval_metrics.items():
        f.write(f"- {k}: {v}\n")


# Save the best model & tokenizer
model.save_pretrained("artifacts/bert_intent_model")
tokenizer.save_pretrained("artifacts/bert_intent_model")

print("BERT model and tokenizer saved to artifacts/bert_intent_model/")