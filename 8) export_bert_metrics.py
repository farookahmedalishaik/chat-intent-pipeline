# 8) export_bert_metrics.py

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# Import required settings and the new helper functions
from config import (
    ARTIFACTS_DIR, ANALYSIS_DIR, TEST_DATA_FILE, LABEL_MAPPING_FILE,
    MODEL_DIR_PATH, TEST_PREDS_FILE, TEST_METRICS_FILE,
    TEST_CONFUSION_FILE, ANALYSIS_MISCLASSIFIED_SAMPLE_FILE,
    HF_DATASET_REPO_ID, HF_REPO_ID
)
from utils import load_artifact_from_hub, load_model_from_hub


def to_numpy(t: torch.Tensor) -> np.ndarray:
    """Helper — safely convert tensor to numpy on CPU."""
    return t.detach().cpu().numpy()


def save_metrics_and_preds(y_true, y_pred, labels):
    """Save classification report, confusion matrix, and preds to artifacts folder."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    np.save(TEST_PREDS_FILE, y_pred)
    print(f"Saved predictions to {TEST_PREDS_FILE}")
    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    df_metrics = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "metric"})
    df_metrics.to_csv(TEST_METRICS_FILE, index=False, encoding="utf-8")
    print(f"Saved classification metrics to {TEST_METRICS_FILE}")
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(TEST_CONFUSION_FILE, encoding="utf-8")
    print(f"Saved confusion matrix to {TEST_CONFUSION_FILE}")


def export_misclassified_examples(test_data, y_true, y_pred, labels):
    """Export misclassified examples for manual review."""
    texts = test_data.get("texts")
    if texts is None:
        print("No 'texts' field in test_data; skipping export of example rows.")
        return
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    df_err = pd.DataFrame({"text": texts, "true": y_true, "pred": y_pred})
    df_err["true_label"] = df_err["true"].apply(lambda x: labels[int(x)])
    df_err["pred_label"] = df_err["pred"].apply(lambda x: labels[int(x)])
    mis_overall = df_err[df_err.true != df_err.pred].head(200)
    mis_overall.to_csv(ANALYSIS_MISCLASSIFIED_SAMPLE_FILE, index=False, encoding="utf-8")
    print(f"Saved top misclassified examples to {ANALYSIS_MISCLASSIFIED_SAMPLE_FILE}")
    for true_lbl in labels:
        sub = df_err[(df_err.true_label == true_lbl) & (df_err.true_label != df_err.pred_label)]
        if not sub.empty:
            safe_true = "".join(c if c.isalnum() else "_" for c in true_lbl)[:80]
            sub.to_csv(os.path.join(ANALYSIS_DIR, f"test_mis_{safe_true}.csv"), index=False, encoding="utf-8")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Cloud First Loading ---
    print("\n--- Step 1: Loading artifacts (cloud-first) ---")
    test_data = load_artifact_from_hub(HF_DATASET_REPO_ID, "test_data.pt", torch.load, TEST_DATA_FILE)
    label_df = load_artifact_from_hub(HF_DATASET_REPO_ID, "label_mapping.csv", pd.read_csv, LABEL_MAPPING_FILE)
    _tokenizer, model = load_model_from_hub(HF_REPO_ID)

    # --- Validation ---
    if test_data is None or label_df is None or model is None:
        print("\n Critical artifact missing. Could not load from Hub or local fallback. Exiting.")
        sys.exit(1)

    labels = label_df["label"].tolist()
    print(f"Loaded {len(labels)} labels.")

    # --- Main Logic (with batching) ---
    print("\n--- Step 2: Running predictions in batches ---")
    input_ids = test_data["input_ids"]
    attention_mask = test_data["attention_mask"]
    y_true = to_numpy(test_data["labels"])
    print(f"Test examples: {len(y_true)}")

    batch_size = 32  # adjustable based on GPU memory
    y_pred_list = []
    
    # Create a DataLoader to handle batching
    test_dataset = TensorDataset(input_ids, attention_mask)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for batch_input_ids, batch_attention_mask in test_dataloader:
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            y_pred_list.append(to_numpy(preds))

    y_pred = np.concatenate(y_pred_list, axis=0)
    print("Prediction complete.")


    print("\n--- Step 3: Saving metrics and predictions ---")
    save_metrics_and_preds(y_true, y_pred, labels)

    test_data["preds"] = y_pred
    torch.save(test_data, TEST_DATA_FILE)
    print(f"Embedded predictions into local file: {TEST_DATA_FILE}")

    print("\n--- Step 4: Exporting misclassified examples ---")
    try:
        export_misclassified_examples(test_data, y_true, y_pred, labels)
    except Exception as e:
        print(f"Could not export misclassified examples: {e}")

    print("\nDone. Outputs are in 'artifacts/' and 'analysis/'.")

if __name__ == "__main__":
    main()