# 6) validation_dashboard.py

import os
import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

# Import required settings and the new helper functions
from config import (
    ARTIFACTS_DIR,
    ANALYSIS_DIR,
    HF_DATASET_REPO_ID,
    HF_REPO_ID,
    MODEL_DIR_PATH,
    LABEL_MAPPING_FILE,
    VAL_DATA_FILE
)
# Centralized helpers for loading artifacts
from utils import load_artifact_from_hub, load_model_from_hub

st.set_page_config(page_title="Validation Dashboard")
st.title("Validation Results")

# --- Cloud First Data and Model Loading using utils.py ---
@st.cache_data
def load_validation_artifacts():
    """Loads all necessary validation artifacts using centralized helpers."""
    val_data = load_artifact_from_hub(
        repo_id=HF_DATASET_REPO_ID,
        filename="val_data.pt",
        local_fallback_path=VAL_DATA_FILE,
        load_fn=torch.load
    )
    label_df = load_artifact_from_hub(
        repo_id=HF_DATASET_REPO_ID,
        filename="label_mapping.csv",
        local_fallback_path=LABEL_MAPPING_FILE,
        load_fn=pd.read_csv
    )
    return val_data, label_df

@st.cache_resource
def load_trained_model():
    """Loads the fine-tuned model using the centralized helper."""
    _tokenizer, model = load_model_from_hub(repo_id=HF_REPO_ID)
    return model

val_data, label_df = load_validation_artifacts()
model = load_trained_model()

# --- Main Dashboard Logic ---
# Check if all artifacts were loaded successfully before proceeding
if val_data is None:
    st.error("Validation data (val_data.pt) could not be loaded from Hub or local path. Please run prepare_data.py and push_data_artifacts.py.")
elif label_df is None:
    st.error("Label mapping could not be loaded from Hub or local path. Please run prepare_data.py and push_data_artifacts.py.")
elif model is None:
    st.error("Trained model could not be loaded from Hub or local path. Please run finetune_bert_simpler.py and push_model.py.")
else:
    input_ids = val_data["input_ids"]
    attention_mask = val_data["attention_mask"]
    y_true = val_data["labels"].cpu().numpy()
    labels = label_df["label"].tolist()

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        y_pred = logits.argmax(dim=-1).cpu().numpy()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Classification report & confusion matrix
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    df_metrics = pd.DataFrame(report).T.reset_index().rename(columns={"index": "metric"})
    st.subheader("Validation Metrics")
    st.dataframe(df_metrics.set_index("metric"))

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    st.subheader("Validation Confusion Matrix")
    fig = px.imshow(df_cm, labels=dict(x="Predicted", y="Actual", color="Count"), text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    # Save validation confusion matrix CSV
    try:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        df_cm.to_csv(os.path.join(ARTIFACTS_DIR, "val_confusion_bert.csv"), encoding="utf-8")
        st.write("Saved validation confusion matrix to artifacts/val_confusion_bert.csv")
    except Exception as e:
        st.write("Could not save validation confusion matrix:", e)

    # Compute per class PR AUC (average precision) and save to CSV
    st.subheader("Per-class Precision-Recall AUC (average precision)")
    pr_auc_scores = {}
    try:
        for cls_idx, cls_name in enumerate(labels):
            y_true_bin = (y_true == cls_idx).astype(int)
            y_scores = probs[:, cls_idx]
            if y_true_bin.sum() == 0:
                pr_auc = float("nan")
            else:
                pr_auc = float(average_precision_score(y_true_bin, y_scores))
            pr_auc_scores[cls_name] = pr_auc
        df_pr = pd.DataFrame(list(pr_auc_scores.items()), columns=["label", "pr_auc"]).sort_values("pr_auc", ascending=False)
        st.dataframe(df_pr)
        df_pr.to_csv(os.path.join(ARTIFACTS_DIR, "val_pr_auc.csv"), index=False, encoding="utf-8")
        st.write("Saved per-class PR AUC to artifacts/val_pr_auc.csv")
    except Exception as e:
        st.write("Could not compute/save PR AUCs:", e)

    # Save validation metrics (including macro f1 if present) for record keeping
    try:
        df_metrics.to_csv(os.path.join(ARTIFACTS_DIR, "validation_metrics.csv"), index=False, encoding="utf-8")
        st.write("Saved validation metrics to artifacts/validation_metrics.csv")
    except Exception as e:
        st.write("Could not save validation metrics:", e)

    # Show top low-F1 labels (validation)
    label_metric_rows = df_metrics[df_metrics["metric"].isin(labels)].set_index("metric")
    if not label_metric_rows.empty and "f1-score" in label_metric_rows.columns:
        bad = label_metric_rows[["f1-score"]].nsmallest(3, "f1-score").index.tolist()
        st.subheader("Top low-F1 labels (validation):")
        st.write(bad)

    # Export validation misclassified examples for manual review
    texts = val_data.get("texts", None)
    if texts is not None:
        try:
            df_err = pd.DataFrame({"text": texts, "true": y_true, "pred": y_pred})
            df_err["true_label"] = df_err["true"].apply(lambda x: labels[int(x)])
            df_err["pred_label"] = df_err["pred"].apply(lambda x: labels[int(x)])
            os.makedirs(ANALYSIS_DIR, exist_ok=True)
            df_err[df_err.true != df_err.pred].to_csv(os.path.join(ANALYSIS_DIR, "val_misclassified_all.csv"), index=False, encoding="utf-8")
            st.write("Saved validation misclassified examples to analysis/val_misclassified_all.csv")
            st.subheader("Sample misclassified validation examples (first 20):")
            st.write(df_err[df_err.true != df_err.pred].head(20)[["text", "true_label", "pred_label"]])
        except Exception as e:
            st.write("Could not export validation misclassified examples:", e)
    else:
        st.info("Validation 'texts' not saved in artifacts/val_data.pt; add 'texts' during prepare_data.py to enable drilldown and export.")

        