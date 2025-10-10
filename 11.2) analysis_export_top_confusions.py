# 10.2) analysis/export_top_confusions.py

import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Import required settings and the helper function
from config import (
    TEST_DATA_FILE, TEST_PREDS_FILE, LABEL_MAPPING_FILE, ANALYSIS_DIR,
    ANALYSIS_MAX_CONFUSION_PAIRS_TO_PRINT, HF_DATASET_REPO_ID
)
from utils import load_artifact_from_hub

# --- Helper functions ---
def to_numpy(x):
    if hasattr(x, "cpu"): return x.cpu().numpy()
    return np.asarray(x)

# safety function to prevent IndexError
def get_label_safely(idx, labels_list):
    try:
        return labels_list[int(idx)]
    except IndexError:
        return "INVALID_LABEL"

def build_dataframe(texts, y_true, y_pred, labels):
    df = pd.DataFrame({"text": texts, "true": to_numpy(y_true), "pred": to_numpy(y_pred)})
    df["true"] = df["true"].astype(int)
    df["pred"] = df["pred"].astype(int)
    df["true_label"] = df["true"].apply(lambda x: get_label_safely(x, labels))
    df["pred_label"] = df["pred"].apply(lambda x: get_label_safely(x, labels))
    return df

def confusion_pairs_from_crosstab(cm, labels):
    pairs = []
    for true_label in labels:
        if true_label not in cm.index: continue
        row = cm.loc[true_label]
        non_diag = row[row.index != true_label]
        if non_diag.sum() == 0: continue
        top_pred = non_diag.idxmax()
        count = int(non_diag.max())
        pairs.append((true_label, top_pred, count))
    pairs.sort(key=lambda x: -x[2])
    return pairs

def sanitize_filename(s: str, max_len: int = 80):
    safe = re.sub(r"[^0-9A-Za-z]+", "_", s)
    return safe[:max_len]

def save_misclassified_csvs(df, pairs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for true_label, pred_label, _count in pairs[:ANALYSIS_MAX_CONFUSION_PAIRS_TO_PRINT]:
        sel = df[(df.true_label == true_label) & (df.pred_label == pred_label)]
        if sel.empty: continue
        safe_true = sanitize_filename(true_label)
        safe_pred = sanitize_filename(pred_label)
        out_path = out_dir / f"mis_{safe_true}_as_{safe_pred}.csv"
        sel.to_csv(out_path, index=False, encoding="utf-8")

# --- Main script flow ---
def main():
    print("--- Step 1: Loading artifacts (cloud first) ---")
    test_data = load_artifact_from_hub(HF_DATASET_REPO_ID, "test_data.pt", torch.load, TEST_DATA_FILE)
    y_pred = load_artifact_from_hub(HF_DATASET_REPO_ID, "test_preds.npy", np.load, TEST_PREDS_FILE)
    label_df = load_artifact_from_hub(HF_DATASET_REPO_ID, "label_mapping.csv", pd.read_csv, LABEL_MAPPING_FILE)

    if test_data is None or y_pred is None or label_df is None:
        print("\n Critical artifact missing. Could not load from Hub or local fallback. Exiting.")
        sys.exit(1)

    print("\n--- Step 2: Analyzing confusions ---")
    texts = test_data.get("texts")
    if texts is None:
        print(" 'texts' not saved into test_data.pt. Cannot proceed. Exiting.")
        sys.exit(1)

    # use .get() for safer access and add a check
    labels_tensor = test_data.get("labels")
    if labels_tensor is None:
        print(" 'labels' not saved into test_data.pt. Cannot proceed. Exiting.")
        sys.exit(1)

    y_true = to_numpy(labels_tensor)
    labels = label_df["label"].tolist()

    df = build_dataframe(texts, y_true, y_pred, labels)
    cm = pd.crosstab(df["true_label"], df["pred_label"])
    pairs = confusion_pairs_from_crosstab(cm, labels)

    print("\nTop confusion pairs (true -> pred -> count):")
    for true, pred, cnt in pairs[:ANALYSIS_MAX_CONFUSION_PAIRS_TO_PRINT]:
        print(f"{true} -> {pred} : {cnt}")

    analysis_path = Path(ANALYSIS_DIR)
    save_misclassified_csvs(df, pairs, analysis_path)
    print(f"\n Saved detailed misclassification CSVs to {analysis_path}/")

if __name__ == "__main__":
    main()