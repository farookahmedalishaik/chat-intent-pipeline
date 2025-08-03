# app.py
import os
import streamlit as st
import sqlite3
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_download
import plotly.express as px



# Page Configuration
st.set_page_config(page_title = "BERT Intent Dashboard")


# Selecting a confidence threshold for 'other' intent ---
CONFIDENCE_THRESHOLD = 0.75

DB_PATH = "runtime_logs.db"

LOW_CONF_TABLE = "low_conf"
LOGS_TABLE     = "logs"

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



# --- Helpers for DB ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    # logs: ts, predicted intent, actual intent (NULLable), correct (0/1/NULL), confidence
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {LOGS_TABLE} (
            ts TEXT,
            pred TEXT,
            actual TEXT,
            correct INTEGER,
            confidence REAL
        )
    """)
    # low_conf: ts, text, confidence
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {LOW_CONF_TABLE} (
            ts TEXT,
            text TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    return conn

# --- Initialize DB ---
conn = init_db()










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
    # Tokenize & Predict
    enc = tokenizer(
        input_message, padding=True, truncation=True,
        return_tensors="pt", max_length=128
    )
    with torch.no_grad():
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=-1)
    max_prob = float(probs.max())
    pred_idx = int(probs.argmax())
    pred_intent = (
        available_intents[pred_idx]
        if max_prob >= CONFIDENCE_THRESHOLD else "other"
    )

    # Display
    st.markdown(
        f"**Predicted Intent:** "
        f"<span style='color:green; font-size:24px;'>{pred_intent}</span>",
        unsafe_allow_html=True
    )
    st.info(f"Confidence: `{max_prob:.2f}` (Threshold `{CONFIDENCE_THRESHOLD}`)")

    # Log to DB
    # actual & correct unknown here → set to NULL
    conn.execute(f"""
        INSERT INTO {LOGS_TABLE} (ts, pred, actual, correct, confidence)
        VALUES (?, ?, NULL, NULL, ?)
    """, (pd.Timestamp.now().isoformat(), pred_intent, max_prob))
    # If low confidence, also log to low_conf
    if max_prob < CONFIDENCE_THRESHOLD:
        conn.execute(f"""
            INSERT INTO {LOW_CONF_TABLE} (ts, text, confidence)
            VALUES (?, ?, ?)
        """, (pd.Timestamp.now().isoformat(), input_message, max_prob))
    conn.commit()
else:
    st.info("Enter a message to see its predicted intent.")

st.markdown("---")

# --- Expander: Model Performance & Confusion ---
with st.expander("📊 Model Performance Metrics"):
    st.dataframe(test_metrics.set_index("metric"))

with st.expander("📈 Classification Confusion Matrix"):
    fig_cm = px.imshow(
        confusion_matrix,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        text_auto=True
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# --- Expander: Precision & Recall Over Time ---
with st.expander("📈 Precision & Recall Over Time"):
    df_logs = pd.read_sql(f"SELECT * FROM {LOGS_TABLE}", conn)
    if df_logs["correct"].notna().any():
        # Compute daily TP/FP/FN per intent
        df_logs["day"] = pd.to_datetime(df_logs["ts"]).dt.date
        records = []
        intents = set(df_logs["pred"].dropna().unique()) | set(df_logs["actual"].dropna().unique())
        for day in sorted(df_logs["day"].unique()):
            day_df = df_logs[df_logs["day"] == day]
            for intent in intents:
                tp = len(day_df[(day_df.pred == intent) & (day_df.correct == 1)])
                fp = len(day_df[(day_df.pred == intent) & (day_df.correct == 0)])
                fn = len(day_df[(day_df.actual == intent) & (day_df.correct == 0)])
                precision = tp / (tp + fp) if (tp + fp) else None
                recall    = tp / (tp + fn) if (tp + fn) else None
                records.append({"day": day, "intent": intent,
                                "precision": precision, "recall": recall})
        df_pr = pd.DataFrame(records)
        if not df_pr.empty:
            fig_pr = px.line(df_pr, x="day", y="precision", color="intent", title="Precision Over Time")
            st.plotly_chart(fig_pr, use_container_width=True)
            fig_rc = px.line(df_pr, x="day", y="recall", color="intent", title="Recall Over Time")
            st.plotly_chart(fig_rc, use_container_width=True)
        else:
            st.write("Not enough labeled data to compute PR.")
    else:
        st.write("No ground-truth labels logged yet to compute precision/recall.")

# --- Expander: Low-Confidence Examples ---
with st.expander("⚠️ Low-Confidence Examples"):
    df_low = pd.read_sql(f"SELECT * FROM {LOW_CONF_TABLE} ORDER BY confidence ASC LIMIT 10", conn)
    if not df_low.empty:
        st.table(df_low[["ts", "text", "confidence"]])
    else:
        st.write("No low-confidence messages recorded yet.")

# --- Drill-Down Misclassifications (Test Set) ---
st.subheader("🔍 Drill-Down Misclassifications (Test Data)")
# Load test set from artifacts
data = torch.load("artifacts/test_data.pt")
texts = data.get("texts", None)
y_true = data["labels"].numpy()
with torch.no_grad():
    out = model(
        input_ids=data["input_ids"],
        attention_mask=data["attention_mask"]
    ).logits
y_pred = out.argmax(dim=-1).numpy()
df_errors = pd.DataFrame({"text": texts, "true": y_true, "pred": y_pred})

selected_true = st.selectbox("Actual Intent", options=confusion_matrix.index)
selected_pred = st.selectbox("Predicted Intent", options=confusion_matrix.columns)
df_drill = df_errors[
    (df_errors.true == selected_true) & (df_errors.pred == selected_pred)
]
st.write(df_drill["text"].head(10))


# Footer
st.write("---")
st.caption("BERT Intent Classification Dashboard - Built with Streamlit")
