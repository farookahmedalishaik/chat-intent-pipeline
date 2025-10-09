# 5) finetune_bert.py

"""
Fine-tune BERT for intent classification.
- Uses utils.load_model_from_hub(..) for all dataset downloads to load model/tokenizer (cloud-first, with local fallback).
"""

import os
import random
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer, default_data_collator
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve

# Import centralized helpers from utils (these handle HF_TOKEN internally)
from utils import load_artifact_from_hub, load_model_from_hub

# Import settings from config
from config import (
    SEED,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    USE_FOCAL_LOSS,
    FOCAL_GAMMA,
    SAMPLING_STRATEGY,
    SAMPLER_REPLACEMENT,
    ARTIFACTS_DIR,
    LOGS_DIR,
    MODEL_OUTPUT_DIR,
    HF_DATASET_REPO_ID,
    HF_REPO_ID,
    BASE_MODEL_ID,
    GRADIENT_ACCUMULATION_STEPS,
)

# Local fallback file paths (used by load_artifact_from_hub as local fallback)
TRAIN_LOCAL = os.path.join(ARTIFACTS_DIR, "train_data.pt")
VAL_LOCAL = os.path.join(ARTIFACTS_DIR, "val_data.pt")
LABEL_MAP_LOCAL = os.path.join(ARTIFACTS_DIR, "label_mapping.csv")
CLASS_WEIGHTS_LOCAL = os.path.join(ARTIFACTS_DIR, "class_weights.npy")


# Helper function to download artifacts using utils.load_artifact_from_hub

def load_artifacts_from_hub(repo_id: str):
    """
    Downloads data artifacts using the centralized helper with consistent token handling & Returns (train_data, val_data, label_map, class_weights_tensor_or_None).
    """
    if not repo_id:
        raise ValueError("HF_DATASET_REPO_ID is not set in config. Cannot download data.")

    print(f"Downloading data artifacts from Hugging Face Hub repo: {repo_id}")

    # train_data and val_data are expected to be PyTorch objects saved by prepare_data.py
    train_data = load_artifact_from_hub(repo_id, "train_data.pt", torch.load, TRAIN_LOCAL)
    val_data   = load_artifact_from_hub(repo_id, "val_data.pt", torch.load, VAL_LOCAL)
    label_map  = load_artifact_from_hub(repo_id, "label_mapping.csv", pd.read_csv, LABEL_MAP_LOCAL)

    # class_weights is optional
    class_weights = load_artifact_from_hub(repo_id, "class_weights.npy", lambda p: np.load(p) if os.path.exists(p) else None, CLASS_WEIGHTS_LOCAL)
    if class_weights is not None:
        class_weights_t = torch.tensor(class_weights, dtype=torch.float)
    else:
        class_weights_t = None

    # Basic validation
    if train_data is None or val_data is None or label_map is None:
        raise RuntimeError("Failed to load one or more required artifacts (train/val/label_map). See messages above.")

    print(" Successfully downloaded and loaded train, validation, and label mapping files.")
    return train_data, val_data, label_map, class_weights_t


# 1) Reproducibility

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# 2) Load all artifacts from the Hub (cloud first via utils)

train_data, val_data, label_map, class_weights_t = load_artifacts_from_hub(repo_id=HF_DATASET_REPO_ID)

labels_list = label_map["label"].tolist()
num_labels = len(labels_list)


# 3) Dataset wrapper

class IntentDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

train_dataset = IntentDataset(train_data["input_ids"], train_data["attention_mask"], train_data["labels"])
val_dataset = IntentDataset(val_data["input_ids"], val_data["attention_mask"], val_data["labels"])


# 4) WeightedRandomSampler (only created if SAMPLING_STRATEGY == "sampler")

sampler = None
if SAMPLING_STRATEGY == "sampler":
    labels_train = train_data["labels"]
    if hasattr(labels_train, "numpy"):
        labels_np = labels_train.numpy()
    else:
        labels_np = np.array(labels_train)

    unique_classes, class_counts = np.unique(labels_np, return_counts=True)
    weight_per_class = {int(c): 1.0 / float(count) for c, count in zip(unique_classes, class_counts)}
    sample_weights = np.array([weight_per_class[int(lbl)] for lbl in labels_np], dtype=np.float64)
    sample_weights = torch.from_numpy(sample_weights)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=SAMPLER_REPLACEMENT)
    print("Sampler created. Class counts:", dict(zip(unique_classes.tolist(), class_counts.tolist())))


# 5) Load model and tokenizer (use load_model_from_hub)

# tries to load the model/tokenizer from a dedicated model repo by using utils.load_model_from_hub if cloud load failes then stop withclear message.
#MODEL_REPO_ID = os.getenv("HF_REPO_ID", HF_REPO_ID) # use env var if set, otherwise use config value
#tokenizer, model = load_model_from_hub(MODEL_REPO_ID)


# The model start fine-tuning from is the base model
print(f"Loading tokenizer and model from {BASE_MODEL_ID}..")
print(f"Configuring model for {num_labels} labels.")

tokenizer, model = load_model_from_hub(BASE_MODEL_ID, num_labels=num_labels)

if tokenizer is None or model is None:
    raise RuntimeError(" Model/tokenizer could not be loaded. See messages above.")

print("Successfully loaded and configured model and tokenizer.")


# 6) Focal loss (simple and clear)

def focal_loss_fn(logits, targets, gamma=2.0, weight=None):
    """
    Simple focal loss:
      loss = ((1 - p_t)^gamma) * cross_entropy
    where p_t is the predicted probability of the true class.
    """
    probs = F.softmax(logits, dim=-1)       # shape (batch, num_labels)


    #targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
   # p_t = (probs * targets_one_hot).sum(dim=-1)  # prob for true class

    p_t = probs.gather(1, targets.unsqueeze(-1)).squeeze() # prob for true class
    ce = F.cross_entropy(logits, targets, weight=weight, reduction="none")
    loss = ((1.0 - p_t) ** gamma) * ce
    return loss.mean()


# 7) Basic metrics used by Trainer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


# 8) Custom Trainer (overrides loss and train loader when needed)

class CustomTrainer(Trainer):
    # ... (other methods like __init__ and compute_loss) ...

    def get_train_dataloader(self) -> DataLoader:
        # If the sampler strategy is chosen, return a DataLoader that uses our custom sampler.
        if SAMPLING_STRATEGY == "sampler" and self.sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=self.sampler,
                collate_fn=self.data_collator,
            )
        # Otherwise, fall back to the parent Trainer's default method.
        return super().get_train_dataloader()


# 9) TrainingArguments & Trainer

training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, 
    per_device_eval_batch_size=2 * BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=LOGS_DIR,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    custom_sampler=sampler
)


# 10) Helper to save artifacts cleanly

def save_artifact_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def save_artifact_model_and_tokenizer(base_dir):
    # ensure directory exists and save model + tokenizer
    save_artifact_dir(base_dir)
    trainer.save_model(base_dir)  # saves model
    tokenizer.save_pretrained(base_dir)  # saves tokenizer files


# 11) Train

train_result = trainer.train()
save_artifact_model_and_tokenizer(os.path.join(ARTIFACTS_DIR, "bert_intent_model"))
print("Saved fine-tuned model to artifacts/bert_intent_model/")


# 12) Evaluate & save metrics


eval_metrics = trainer.evaluate()
print("Eval metrics:", eval_metrics)
with open(os.path.join(LOGS_DIR, "training_results.md"), "w") as f:
    f.write("## Final Evaluation Metrics\n")
    for k, v in eval_metrics.items():
        f.write(f"- {k}: {v}\n")


# 13) Predict on validation set (get logits & probs) and compute thresholds

print("Predicting on validation set to compute thresholds..")
pred_result = trainer.predict(val_dataset)
val_logits = pred_result.predictions
val_labels = pred_result.label_ids
val_probs = torch.softmax(torch.from_numpy(val_logits).float(), dim=1).numpy()

print("Computing per class thresholds..")
class_thresholds = {}
for cls_idx, cls_name in enumerate(labels_list):
    y_true_bin = (val_labels == cls_idx).astype(int)
    y_scores = val_probs[:, cls_idx]
    if y_true_bin.sum() == 0:
        class_thresholds[cls_name] = 0.75
        continue

    precision, recall, thresholds = precision_recall_curve(y_true_bin, y_scores)

    if len(thresholds) == 0:
        class_thresholds[cls_name] = 0.75
        continue

    f1s = []
    for i, th in enumerate(thresholds):
        p = precision[i + 1]
        r = recall[i + 1]
        if (p + r) == 0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * p * r / (p + r))

    best_idx = int(np.argmax(f1s)) if len(f1s) > 0 else 0
    best_thresh = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.75
    best_thresh = max(0.01, min(0.99, best_thresh))
    class_thresholds[cls_name] = best_thresh

th_path = os.path.join(ARTIFACTS_DIR, "class_thresholds.json")
with open(th_path, "w") as f:
    json.dump(class_thresholds, f, indent=2)
print(f"Saved per class thresholds to {th_path}")
print("Example thresholds (first 10):", dict(list(class_thresholds.items())[:10]))

print(" Process Complete. Model and artifacts saved in artifacts/ folder.")
