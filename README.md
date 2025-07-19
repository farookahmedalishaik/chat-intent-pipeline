# chat-intent-pipeline
# Chat‑Log Intent Classification Pipeline

This repository contains an end‑to‑end pipeline for classifying user intents in chat/SMS logs. It includes data ingestion, cleaning, annotation auditing, feature engineering, and both traditional ML and transformer‑based modeling.

---

## 📂 Project Structure



## Intent Classification Model Training

This project focuses on building a robust intent classification system using a fine-tuned BERT model. The `finetune_bert.py` script orchestrates the training process, leveraging the Hugging Face Transformers library.

**Training Details:**
* **Base Model:** `bert-base-uncased` (pre-trained, then fine-tuned)
* **Training Data:** Custom dataset (details on preprocessing and tokenization are handled separately)
* **Training Epochs:** 3
* **Batch Sizes:** Train: 16, Evaluation: 32
* **Framework:** PyTorch with Hugging Face Trainer API

**Key Results (from Validation Set):**
The model was evaluated on a held-out validation set after each epoch. The final performance metrics, based on the best model, are:
* **Validation Loss:** Approximately `0.0078`
* **Validation Accuracy:** Approximately `99.88%`

These results demonstrate the model's high accuracy and effectiveness in identifying user intents. The fine-tuned model checkpoint is committed to this repository using Git Large File Storage (LFS) and is located in the `artifacts/bert_intent_model/` directory.
