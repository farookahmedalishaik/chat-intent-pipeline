# 10.1) error_analysis.py
import os
import sys
import pandas as pd
import numpy as np
import torch

# Import required settings and the new helper function
from config import (
    ARTIFACTS_DIR, CLEANED_DATA_FILE, TEST_RATIO, SEED,
    TEST_METRICS_FILE, TEST_CONFUSION_FILE, LABEL_MAPPING_FILE,
    TEST_DATA_FILE, LOW_F1_ERRORS_FILE, TEST_PREDS_FILE,
    HF_DATASET_REPO_ID
)
from utils import load_artifact_from_hub

# Utility helpers 
def read_metrics_and_pick_low_f1(metrics_df, labels_list, k=3):
    metrics_for_labels = metrics_df[metrics_df["metric"].isin(labels_list)].copy()
    if "f1-score" not in metrics_for_labels.columns:
        raise KeyError("Expected 'f1-score' column in test metrics CSV.")
    return metrics_for_labels.nsmallest(k, "f1-score")["metric"].tolist()

def get_texts_from_data(data_dict):
    if "texts" in data_dict:
        return data_dict["texts"]
    raise ValueError("'texts' key not found in test_data.pt. Please re run prepare_data.py.")


# Main script flow
def main():
    print("--- Step 1: Loading artifacts (cloud-first) ---")
    test_metrics_df = load_artifact_from_hub(HF_DATASET_REPO_ID, "test_metrics_bert.csv", pd.read_csv, TEST_METRICS_FILE)
    label_df = load_artifact_from_hub(HF_DATASET_REPO_ID, "label_mapping.csv", pd.read_csv, LABEL_MAPPING_FILE)
    test_data = load_artifact_from_hub(HF_DATASET_REPO_ID, "test_data.pt", torch.load, TEST_DATA_FILE)
    y_pred = load_artifact_from_hub(HF_DATASET_REPO_ID, "test_preds.npy", np.load, TEST_PREDS_FILE)


    if test_metrics_df is None or label_df is None or test_data is None or y_pred is None:
        print("\n Critical artifact missing. Could not load from Hub or local fallback. Exiting.")
        sys.exit(1)
        
    print("\n--- Step 2: Processing artifacts ---")
    labels_list = label_df["label"].tolist()
    label_to_id = {label: idx for idx, label in enumerate(labels_list)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    low_labels = read_metrics_and_pick_low_f1(test_metrics_df, labels_list, k=3)
    low_label_ids = [label_to_id[l] for l in low_labels]
    print("Lowest F1 labels:", low_labels, "-> ids:", low_label_ids)
    
    y_true = test_data["labels"].cpu().numpy()
    test_texts = get_texts_from_data(test_data)

    print("\n--- Step 3: Finding misclassifications for low-F1 intents ---")
    df_errors = pd.DataFrame({"text": test_texts, "true": y_true, "pred": y_pred})

    rows = []
    for intent_id, intent_label in zip(low_label_ids, low_labels):
        subset = df_errors[(df_errors["true"] == intent_id) & (df_errors["pred"] != intent_id)].head(10)
        for _, r in subset.iterrows():
            pred_id = int(r["pred"])
            rows.append({
                "intent_id": int(intent_id),
                "intent_label": intent_label,
                "text": r["text"],
                "pred_id": pred_id,
                "pred_label": id_to_label.get(pred_id, str(pred_id))
            })

    df_low_f1 = pd.DataFrame(rows)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    df_low_f1.to_csv(LOW_F1_ERRORS_FILE, index=False, encoding="utf-8")
    print(f"\n Saved low-F1 misclassifications to {LOW_F1_ERRORS_FILE}")

if __name__ == "__main__":
    main()

    