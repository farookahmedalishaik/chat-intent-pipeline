# app.py
import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_download



# Page Configuration
st.set_page_confit(page_title = "BERT Intent Dashboard")


# Selecting a confidence threshold for 'other' intent ---
CONFIDENCE_THRESHOLD = 0.75



# User guidance text for available intents
INTENT_GUIDANCE_TEXT = """
This tool classifies messages into the following categories:
- **`request_invoice`**: Inquiries about invoices or getting an invoice.
- **`manage_shipping_address`**: Requests to change or set up shipping addresses.
- **`contact_support`**: Direct requests to speak with customer service or a human agent.
- **`general_feedback`**: Reviews, suggestions, or thank you messages.
- **`check_promotions`**: Questions about ongoing promotions, offers, discounts, or sales.
- **`delivery_information`**: Inquiries about delivery options or periods.
- **`queries_related_to_order`**: Questions about placing, tracking, canceling, or changing an order.
- **`queries_related_to_refund`**: Questions about getting or tracking a refund, or refund policy.
- **`account_management`**: Issues or requests related to creating, deleting, editing, switching, or recovering an account, and registration problems.
- **`queries_related_to_payments/fee`**: Questions about payment methods, payment issues, or cancellation fees.
- **`building/store hours`**: Inquiries about operational hours.
- **`newsletter_subscription`**: Requests to subscribe or manage newsletter subscriptions.
- **`complaint`**: Explicit complaints.
- **`other`**: Messages that do not fit into the above categories (assigned if confidence is low).
"""










# Load BERT model,tokenizer,CSVs
@st.cache_resource
def load_bert_and_data():
   
    hf_model_repo = "farookahmedalishaik/intent-bert" 

    # Load tokenizer and model from Hugging Face Hub
    bert_tokenizer = BertTokenizer.from_pretrained(hf_model_repo)
    bert_model = BertForSequenceClassification.from_pretrained(hf_model_repo)
    bert_model.eval()

    # Download label mapping CSV from Hugging Face Hub
    label_file = hf_hub_download(repo_id = hf_model_repo, filename = "label_mapping.csv")
    label_data = pd.read_csv(label_file)

    return bert_tokenizer, bert_model, label_data

@st.cache_data
def load_evaluation_data():
    hf_model_repo = "farookahmedalishaik/intent-bert"
    
    # Performance metrics
    metrics_file = hf_hub_download(repo_id = hf_model_repo, filename = "test_metrics_bert.csv")
    test_metrics = pd.read_csv(metrics_file)
    
    # Confusion matrix data
    confusion_file = hf_hub_download(repo_id = hf_model_repo, filename = "test_confusion_bert.csv")
    confusion_matrix = pd.read_csv(confusion_file, index_col = 0)
    
    return test_metrics, confusion_matrix


# Unpack/Initialize all returned values from the cached function
tokenizer, model, label_mapping = load_bert_and_data()
available_intents = label_mapping["label"].tolist()
test_metrics, confusion_matrix = load_evaluation_data()


# Sidebar (Leftmost Column) ---
st.sidebar.title("📚 Understanding the Categories")

# st.sidebar.header("")

st.sidebar.info("💡 This tool classifies customer messages into specific intent categories. Knowing these helps you craft effective queries:")
st.sidebar.markdown(INTENT_GUIDANCE_TEXT) # To display the detailed list of intents


# --- Main Content (Middle Column) ---
st.title("🔍 BERT Intent Classification Dashboard")

# Input message & prediction
st.header("💬 Enter Your Message")
input_message = st.text_area("Type your message here:", height = 150, label_visibility = "collapsed") # Input in main area

# Live prediction Interface
st.header("🔮 Intent Prediction Result")
if input_message.strip():
    # Tokenizing
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
        # Apply softmax to get probabilities
        probabilities = torch.softmax(model_output.logits, dim = -1)
        
        # Get highest probability and its corresponding class index
        max_probability = probabilities.max().item()
        predicted_class_idx = probabilities.argmax(dim = -1).item()
    
    # Determine the predicted intent based on confidence threshold
    if max_probability >= CONFIDENCE_THRESHOLD:
        predicted_intent = available_intents[predicted_class_idx]
    else:
        predicted_intent = "other" # "other" when confidence is very low
        
    st.markdown(f"**Predicted Intent:** `<span style ='color:green; font-size: 24px;'>{predicted_intent}</span>", unsafe_allow_html=True)
    st.info(f"Confidence Score: `{max_probability:.2f}` (Threshold: `{CONFIDENCE_THRESHOLD:.2f}`)")
else:
    st.info("Enter a message above to see its predicted intent.")


# Horizontal separator
st.markdown("---")

# Expandable sections view for performance metrics and confusion matrix
with st.expander("📊 View Model Performance Metrics"):
    st.subheader("Model Performance on Test Data")
    metrics_display = test_metrics.set_index("metric")
    st.dataframe(metrics_display)

with st.expander("📈 View Classification Confusion Matrix"):
    st.subheader("Classification Confusion Matrix")
    import plotly.express as px
    heatmap_fig = px.imshow(
        confusion_matrix,
        labels = dict(x = "Predicted Intent", y = "Actual Intent", color = "Frequency"),
        x = confusion_matrix.columns,
        y = confusion_matrix.index,
        text_auto = True
    )
    st.plotly_chart(heatmap_fig, use_container_width =True)


# Footer
st.write("---")
st.caption("BERT Intent Classification Dashboard - Built with Streamlit")
