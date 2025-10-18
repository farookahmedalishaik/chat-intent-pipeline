# 7) validation_dashboard.py

import os
import streamlit as st
import pandas as pd
import plotly.express as px

# Import required settings and the new helper functions
from config import (
    ARTIFACTS_DIR, # for local fallback paths
    HF_REPO_ID,
    VAL_METRICS_FILE,
    VAL_CONFUSION_FILE
)
# Centralized helpers for loading artifacts
from utils import load_artifact_from_hub

st.set_page_config(page_title="Validation Dashboard")
st.title("Validation Results")

# Artifact Loading
@st.cache_data
def load_validation_metrics():
    """Loads pre calculated validation metrics from the MODEL repository."""
    
    # Load the classification report
    df_metrics = load_artifact_from_hub(
        repo_id=HF_REPO_ID, 
        filename="val_classification_report.csv",
        load_fn=lambda p: pd.read_csv(p, index_col=0),
        local_fallback_path=VAL_METRICS_FILE
    )
    
    # Load the confusion matrix
    df_cm = load_artifact_from_hub(
        repo_id=HF_REPO_ID, 
        filename="val_confusion_matrix.csv",
        # index_col=0 tells pandas to use the first column as the row labels
        load_fn=lambda p: pd.read_csv(p, index_col=0),
        local_fallback_path=VAL_CONFUSION_FILE
    )
    
    return df_metrics, df_cm

# Load the artifacts
df_metrics, df_cm = load_validation_metrics()

# Main Dashboard Logic
if df_metrics is None:
    st.error("Validation metrics report could not be loaded. Please run generate_val_metrics.py and push_model.py.")
elif df_cm is None:
    st.error("Validation confusion matrix could not be loaded. Please run generate_val_metrics.py and push_model.py.")
else:
    # Display the validation metrics dataframe
    st.subheader("Validation Metrics")
    st.dataframe(df_metrics)

    # Display the confusion matrix plot
    st.subheader("Validation Confusion Matrix")
    fig = px.imshow(df_cm, labels=dict(x="Predicted", y="Actual", color="Count"), text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    # Show top low-F1 labels (validation)
    labels = df_cm.index.tolist()
    # Filter df_metrics to only include rows corresponding to the labels in the confusion matrix
    label_metric_rows = df_metrics[df_metrics.index.isin(labels)]
    
    if not label_metric_rows.empty and "f1-score" in label_metric_rows.columns:
        bad = label_metric_rows[["f1-score"]].nsmallest(3, "f1-score").index.tolist()
        st.subheader("Top low-F1 labels (validation):")
        st.write(bad)