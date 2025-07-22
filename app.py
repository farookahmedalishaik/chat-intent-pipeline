import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_download # <-- ADD THIS IMPORT

# 1. Page config
st.set_page_config(page_title="BERT Intent Dashboard", layout="wide")

# 2. Load BERT model + tokenizer + CSVs once
@st.cache_resource
def load_bert_and_data():
   
    hf_model_repo = "farookahmedalishaik/intent-bert" 

    # Load tokenizer and model from Hugging Face Hub
    tok = BertTokenizer.from_pretrained(hf_model_repo)
    model = BertForSequenceClassification.from_pretrained(hf_model_repo)
    model.eval()

    # --- IMPORTANT UPDATE FOR CSVS ---
    # Download label mapping CSV from Hugging Face Hub
    label_map_path = hf_hub_download(repo_id=hf_model_repo, filename="label_mapping.csv")
    label_map = pd.read_csv(label_map_path)

    # Download test metrics CSV from Hugging Face Hub
    metrics_csv_path = hf_hub_download(repo_id=hf_model_repo, filename="test_metrics_bert.csv")
    metrics_df = pd.read_csv(metrics_csv_path)

    # Download confusion matrix CSV from Hugging Face Hub
    cm_csv_path = hf_hub_download(repo_id=hf_model_repo, filename="test_confusion_bert.csv")
    cm_df = pd.read_csv(cm_csv_path, index_col=0)
    # --- END IMPORTANT UPDATE FOR CSVS ---

    return tok, model, label_map, metrics_df, cm_df

# Unpack all returned values from the cached function
tokenizer, model, label_map, metrics_df, cm_df = load_bert_and_data()
labels = label_map["label"].tolist()


# 3. Sidebar inputs
st.sidebar.title("Settings")
show_cm = st.sidebar.checkbox("Show Confusion Matrix", value=True)
user_text = st.sidebar.text_area("Enter a message to predict intent:")

# 4. Title
st.title("🔍 BERT Intent Classification Dashboard")

# 5. Display test-set metrics
st.header("Test Set Metrics (BERT)")
st.dataframe(metrics_df.set_index("metric")) # Use the loaded metrics_df

# 6. Confusion matrix
if show_cm:
    st.subheader("Confusion Matrix")
    import plotly.express as px
    fig = px.imshow(
        cm_df, # Use the loaded cm_df
        labels=dict(x="Predicted", y="True", color="Count"),
        x=cm_df.columns, y=cm_df.index,
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)

# 7. Live prediction
st.header("🔮 Predict a New Message’s Intent")
if user_text:
    inputs = tokenizer(
        user_text, padding=True, truncation=True,
        return_tensors="pt", max_length=128
    )
    with torch.no_grad():
        out = model(**inputs)
        pred = out.logits.argmax(dim=-1).item()
    intent = labels[pred]
    st.markdown(f"**Predicted intent:** `{intent}`")

# 8. Footer
st.write("---")
st.caption("Built with Streamlit • Phase 5: BERT-only dashboard")