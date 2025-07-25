# app.py
import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_download # <-- ADD THIS IMPORT

# Page configuration
st.set_page_config(page_title ="BERT Intent Dashboard", layout="wide")

# Load BERT model,tokenizer,CSVs
@st.cache_resource
def load_bert_and_data():
   
    hf_model_repo = "farookahmedalishaik/intent-bert" 

    # Load tokenizer and model from Hugging Face Hub
    bert_tokenizer = BertTokenizer.from_pretrained(hf_model_repo)
    bert_model = BertForSequenceClassification.from_pretrained(hf_model_repo)
    bert_model.eval()

    # Download label mapping CSV from Hugging Face Hub
    label_file = hf_hub_download(repo_id=hf_model_repo, filename="label_mapping.csv")
    label_data = pd.read_csv(label_file)

    return bert_tokenizer, bert_model, label_data

@st.cache_data
def load_evaluation_data():
    hf_model_repo = "farookahmedalishaik/intent-bert"
    
    # Performance metrics
    metrics_file = hf_hub_download(repo_id=hf_model_repo, filename="test_metrics_bert.csv")
    test_metrics = pd.read_csv(metrics_file)
    
    # Confusion matrix data
    confusion_file = hf_hub_download(repo_id=hf_model_repo, filename="test_confusion_bert.csv")
    confusion_matrix = pd.read_csv(confusion_file, index_col=0)
    
    return test_metrics, confusion_matrix


# Unpack/Initialize all returned values from the cached function
tokenizer, model, label_mapping = load_bert_and_data()
available_intents = label_mapping["label"].tolist()
test_metrics, confusion_matrix = load_evaluation_data()


# Sidebar inputs
st.sidebar.title("Dashboard Controls")
display_confusion_matrix = st.sidebar.checkbox("Display Confusion Matrix", value=True)
input_message = st.sidebar.text_area("Test message for intent prediction:")


# Title
st.title("🔍 BERT Intent Classification Dashboard")


# Show model performance
st.header("Model Performance on Test Data")
metrics_display = test_metrics.set_index("metric") # Using the loaded metrics_df
st.dataframe(metrics_display)

# Confusion matrix
if display_confusion_matrix:
    st.subheader("Classification Confusion Matrix")
    import plotly.express as px
    heatmap_fig = px.imshow(
        confusion_matrix, # Using the loaded cm_df
        labels=dict(x = "Predicted Intent", y = "Actual Intent", color = "Frequency"),
        x = confusion_matrix.columns,
        y = confusion_matrix.index,
        text_auto = True
    )
    st.plotly_chart(heatmap_fig, use_container_width =True)


# Live prediction Interface
st.header("🔮 Intent Prediction Tool")
if input_message.strip():
    # Tokenize the input
    model_inputs = tokenizer(
        input_message, 
        padding =True, 
        truncation = True,
        return_tensors = "pt", 
        max_length = 128
    )
    
    # Get prediction
    with torch.no_grad():
        model_output = model(**model_inputs)
        predicted_class = model_output.logits.argmax(dim=-1).item()
    
    predicted_intent = available_intents[predicted_class]
    st.markdown(f"**Predicted Intent:** `{predicted_intent}`")


# Footer
st.write("---")
st.caption("BERT Intent Classification Dashboard - Built with Streamlit")