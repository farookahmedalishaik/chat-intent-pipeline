# 6) generate_val_metrics.py

import os
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

# Import required settings from the central config file
from config import (
    MODEL_DIR_PATH,
    VAL_DATA_FILE,
    LABEL_MAPPING_FILE,
    VAL_METRICS_FILE,
    VAL_CONFUSION_FILE
)

def main():
    """
    Loads a locally trained model and validation data to generate and save
    detailed validation metrics without re training.
    """
    print("Starting Validation Metrics Generation..")

    # 1. Load Artifacts from Local Disk 
    if not os.path.exists(MODEL_DIR_PATH):
        print(f"ERROR: Trained model not found at '{MODEL_DIR_PATH}'.")
        print("Please make sure finetune_bert.py has run successfully at least once.")
        return

    if not os.path.exists(VAL_DATA_FILE):
        print(f"ERROR: Validation data not found at '{VAL_DATA_FILE}'. Please run prepare_data.py.")
        return
        
    if not os.path.exists(LABEL_MAPPING_FILE):
        print(f"ERROR: Label mapping not found at '{LABEL_MAPPING_FILE}'. Please run prepare_data.py.")
        return

    print("Loading model, validation data, and label mapping...")
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR_PATH)
    val_data = torch.load(VAL_DATA_FILE)
    label_df = pd.read_csv(LABEL_MAPPING_FILE)
    labels_list = label_df["label"].tolist()
    print("Artifacts loaded successfully.")

    # 2. Run Predictions 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_ids = val_data["input_ids"].to(device)
    attention_mask = val_data["attention_mask"].to(device)
    y_true = val_data["labels"].cpu().numpy()

    print("Running predictions on validation data..")
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        y_pred = logits.argmax(dim=-1).cpu().numpy()
    print("Prediction complete.")

    #  3. Calculate and Save Detailed Metrics
    print("Calculating and saving detailed validation metrics..")

    # Classification Report
    report_dict = classification_report(y_true, y_pred, target_names=labels_list, output_dict=True, zero_division=0)
    df_val_metrics = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "metric"})
    df_val_metrics.to_csv(VAL_METRICS_FILE, index=False, encoding="utf-8")
    print(f"-> Saved validation classification report to: {VAL_METRICS_FILE}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=labels_list, columns=labels_list)
    df_cm.to_csv(VAL_CONFUSION_FILE, encoding="utf-8")
    print(f"-> Saved validation confusion matrix to: {VAL_CONFUSION_FILE}")

    print("\n Process Complete! You can now run push_model.py.")

if __name__ == "__main__":
    main()