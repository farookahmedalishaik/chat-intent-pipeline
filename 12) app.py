# 12) app.py

import os
import json
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import streamlit as st

from preprocess import normalize_placeholders
from sqlalchemy import text
from transformers import BertForSequenceClassification, BertTokenizer

# Import all settings from the central config file
from config import (
    # App Settings
    APP_DEFAULT_CONFIDENCE_THRESHOLD,
    APP_FALLBACK_SQLITE_DB_FILE,
    APP_LOW_CONFIDENCE_LOGS_TABLE,
    APP_PREDICTION_LOGS_TABLE,
    APP_SIDEBAR_GUIDANCE_TEXT,
    MAX_TEXT_LENGTH,
    # Secrets & Repo
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_DB,
    HF_REPO_ID,
    HF_DATASET_REPO_ID, 
    # File Paths
    MODEL_DIR_PATH,
    LABEL_MAPPING_FILE,
    TEST_METRICS_FILE,
    TEST_CONFUSION_FILE,
    TEST_DATA_FILE,
    CLASS_THRESHOLDS_FILE,
    TEST_PREDS_FILE
)
# Centralized helpers for loading artifacts & DB engine
from utils import load_artifact_from_hub, load_model_from_hub, tensor_to_numpy, get_db_engine


# Page Configuration
st.set_page_config(page_title="BERT Intent Dashboard")

# Robustly check for Hugging Face Repo ID from config
if not HF_REPO_ID:
    st.error(
        "CRITICAL ERROR: `HF_REPO_ID` is not configured. Please set HF_USER and HF_MODEL_REPO_NAME (or HF_REPO_ID) and restart."
    )
    st.stop()


# Create engine using centralized helper from utils, pass config values and fallback sqlite file
engine = get_db_engine(
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_DB,
    sqlite_fallback_path=APP_FALLBACK_SQLITE_DB_FILE
)


# Determine the correct SQL function for the current timestamp based on the DB engine
if engine.dialect.name == 'mysql':
    db_timestamp_func = "NOW()"
else: # Assume SQLite or others
    db_timestamp_func = "CURRENT_TIMESTAMP"


def init_db_tables(engine):
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {APP_PREDICTION_LOGS_TABLE} (
                ts TEXT,
                pred TEXT,
                actual TEXT,
                correct INTEGER,
                confidence REAL,
                slots TEXT
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {APP_LOW_CONFIDENCE_LOGS_TABLE} (
                ts TEXT,
                text TEXT,
                confidence REAL,
                slots TEXT
            )
        """))


# Centralized Artifact Loading with Caching
@st.cache_resource
def load_model_and_tokenizer():
    """Loads model and tokenizer using the utility function."""
    _tokenizer, _model = load_model_from_hub(repo_id=HF_REPO_ID)
    if _model is None:
        st.error("FATAL: Model could not be loaded. The application cannot start.")
        st.stop()
    _model.eval()
    return _tokenizer, _model


@st.cache_data
def load_all_artifacts():
    """Loads all data and evaluation artifacts using utility functions."""
    artifacts = {}
    
    # Load from Dataset Repo
    artifacts["label_mapping"] = load_artifact_from_hub(
        HF_DATASET_REPO_ID, "label_mapping.csv", pd.read_csv, LABEL_MAPPING_FILE
    )
    artifacts["test_data"] = load_artifact_from_hub(
        HF_DATASET_REPO_ID, "test_data.pt", torch.load, TEST_DATA_FILE
    )
    artifacts["test_metrics"] = load_artifact_from_hub(
        HF_DATASET_REPO_ID, "test_metrics_bert.csv", pd.read_csv, TEST_METRICS_FILE
    )
    artifacts["confusion_matrix"] = load_artifact_from_hub(
        HF_DATASET_REPO_ID, "test_confusion_bert.csv", lambda p: pd.read_csv(p, index_col=0), TEST_CONFUSION_FILE
    )
    artifacts["y_pred"] = load_artifact_from_hub(
        HF_DATASET_REPO_ID, "test_preds.npy", np.load, TEST_PREDS_FILE
    )
    
    # Load from Model Repo
    thresholds_json = load_artifact_from_hub(
        HF_REPO_ID, "class_thresholds.json", lambda p: json.load(open(p)), CLASS_THRESHOLDS_FILE
    )
    artifacts["class_thresholds"] = thresholds_json if thresholds_json else {}

    # Validations
    if artifacts["label_mapping"] is None:
        st.error("FATAL: Label mapping is missing. The application cannot start.")
        st.stop()
    if artifacts["test_data"] is None or artifacts["y_pred"] is None:
        st.warning("Test data or predictions missing. Drill-down will be unavailable.")

    return artifacts


# Load all necessary components
tokenizer, model = load_model_and_tokenizer()
artifacts = load_all_artifacts()

# Unpack artifacts for easier access
label_mapping = artifacts["label_mapping"]
available_intents = label_mapping["label"].tolist()
test_metrics = artifacts["test_metrics"]
confusion_matrix = artifacts["confusion_matrix"]
test_data = artifacts["test_data"]
y_pred = artifacts["y_pred"]
class_thresholds = artifacts["class_thresholds"]

# init DB tables 
init_db_tables(engine)

# Sidebar

st.sidebar.title("📚 Understanding the Categories")
st.sidebar.markdown("""
This tool classifies messages into the following intents. Here are some examples:

**1) account_management**
* "I would like to open an account."
* "Create an account."
* "Modify information on my account."
* "I don't know how to delete the account."

**2) check_promotions**
* "Where can I find a voucher for my family?"
* "Do you have any rebate for orders over $50?"
* "What's the discount on clothing?"
* "I'd like to know the promo code for students."

**3) complaint**
* "Where can I make a consumer claim against your organization?"
* "How do I lodge a consumer complaint?"
* "I am unhappy with your service."
* "I have to file a customer complaint."
                    
**4) contact_support**
* "What hours can I call customer service?"
* "How can I get in touch with customer support?"
* "I have to speak with someone from customer support."

**5) delivery_information**
* "Help me check what delivery methods I can choose."
* "How can I check what shipment methods are available?"
* "Do you ship to Finland?"

**6) inquire_business_hours**
* "Can you share your office business hours this week?"
* "What are your store opening hours today?"

**7) manage_shipping_address**
* "I have troubles editing my delivery address."
* "Help me to change the delivery address."
* "I need support modifying my delivery address."

**8) newsletter_subscription**
* "I need help canceling my subscription to the newsletter."
* "Help me sign up for your company newsletter."

**9) queries_related_to_order**
* "I can't add some items to order 370795561790."
* "How do I edit purchase order #419790?"
* "Can you help me swap something in purchase order #2680226?"
                  
**10) queries_related_to_payment/fee**
* "I want to see what payment options are accepted."
* "I need help checking the early termination penalties."
* "How can I see the cancellation fees?"

**11) queries_related_to_refund**
* "I need assistance to check in what cases I can request a refund."
* "Can you help me get my money back?"
* "In which cases can I ask for a reimbursement?"

**12) request_bill**
* "I want assistance downloading bill #37777."
* "Where can I check my bill #00108?"
* "I want to take a quick look at bill #85632."

**13) request_invoice**
* "I don't know how I can locate invoice #85632."
* "See invoices from Ms. Hawkings."
* "I want assistance finding my invoice #85632."

**14) review**
* "I need assistance to leave an opinion on your services."
* "How can I leave a review for your products?"
* "Is there an email to send some feedback to your company?"
                    
**15) other**
* "out-of-scope messages that do not fit any of the above categories"

""")


st.title("🔍 BERT Intent Classification Dashboard")
st.info("Please Note: This model is an expert on the 14 intents shown in the sidebar. For any out-of-scope messages, it will predict the most similar category it was trained on.")
st.header("💬 Enter Your Message")
input_message = st.text_area("Type your message here:", height=150)

st.header("🔮 Intent Prediction Result")
if input_message.strip():
    normalized_text, slot_mappings = normalize_placeholders(input_message)
    slots_json = json.dumps(slot_mappings, ensure_ascii=False)

    enc = tokenizer(
        normalized_text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=MAX_TEXT_LENGTH
    )

    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    max_prob = float(probs.max())
    pred_idx = int(np.argmax(probs))
    pred_label_name = available_intents[pred_idx]
    
    threshold = class_thresholds.get(pred_label_name, APP_DEFAULT_CONFIDENCE_THRESHOLD)
    pred_intent = pred_label_name if max_prob >= threshold else "other"

    st.markdown(f"**Predicted Intent:** <span style='color:green; font-size:24px;'>{pred_intent}</span>", unsafe_allow_html=True)
    st.info(f"Confidence: `{max_prob:.2f}` (Threshold for this class: `{threshold:.2f}`)")

    # Log prediction to centralized DB
    with engine.begin() as conn:
        # The SQL string now includes the database function for the timestamp
        conn.execute(
            text(f"INSERT INTO {APP_PREDICTION_LOGS_TABLE} (ts, pred, actual, correct, confidence, slots) VALUES ({db_timestamp_func}, :pred, :actual, :correct, :confidence, :slots)"),
            # Use None for actual and correct since we don't have ground truth here
            {"pred": pred_intent, "actual": None, "correct": None, "confidence": max_prob, "slots": slots_json}
        )
        if max_prob < threshold:
            conn.execute(
                text(f"INSERT INTO {APP_LOW_CONFIDENCE_LOGS_TABLE} (ts, text, confidence, slots) VALUES ({db_timestamp_func}, :text, :confidence, :slots)"),
                {"text": input_message, "confidence": max_prob, "slots": slots_json}
            )
else:
    st.info("Enter a message to see its predicted intent.")

st.markdown("---")

with st.expander("📊 Model Performance Metrics (test data)"):
    if test_metrics is not None and not test_metrics.empty:
        st.dataframe(test_metrics.set_index("metric"))
    else:
        st.warning("Test metrics not available.")

with st.expander("📈 Classification Confusion Matrix (test data)"):
    if confusion_matrix is not None and not confusion_matrix.empty:
        fig_cm = px.imshow(confusion_matrix, labels=dict(x="Predicted", y="Actual", color="Count"), text_auto=True)
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.warning("Confusion matrix not available.")


with st.expander("📈 Precision & Recall Over Time (history of live predictions)"):
    pr_query = f"""
        -- a Common Table Expression (CTE) to prepare the data
        WITH daily_counts AS (
            SELECT
                -- Use standard SQL DATE() function to group by day
                DATE(ts) AS day,
                intent,
                -- Calculate True Positives (TP): Predicted as this intent and was correct
                SUM(CASE WHEN role = 'predicted' AND correct = 1 THEN 1 ELSE 0 END) AS tp,
                -- Calculate False Positives (FP): Predicted as this intent and was incorrect
                SUM(CASE WHEN role = 'predicted' AND correct = 0 THEN 1 ELSE 0 END) AS fp,
                -- Calculate False Negatives (FN): Was actually this intent but the prediction was wrong
                SUM(CASE WHEN role = 'actual' AND correct = 0 THEN 1 ELSE 0 END) AS fn
            FROM (
                -- This subquery creates a unified list of all 'events' (predicted and actual)
                SELECT ts, pred AS intent, 'predicted' AS role, correct FROM {APP_PREDICTION_LOGS_TABLE} WHERE pred IS NOT NULL AND correct IS NOT NULL
                UNION ALL
                SELECT ts, actual AS intent, 'actual' AS role, correct FROM {APP_PREDICTION_LOGS_TABLE} WHERE actual IS NOT NULL AND correct IS NOT NULL
            ) AS intent_events
            GROUP BY day, intent
        )
        -- Final calculation of precision and recall from the counts
        SELECT
            day,
            intent,
            -- Precision = TP / (TP + FP), handle division by zero
            CAST(tp AS DECIMAL(10, 5)) / NULLIF(tp + fp, 0) AS `precision`,
            -- Recall = TP / (TP + FN), handle division by zero
            CAST(tp AS DECIMAL(10, 5)) / NULLIF(tp + fn, 0) AS `recall`
        FROM daily_counts
        -- Only show results where there was some activity
        WHERE tp + fp + fn > 0
        ORDER BY day, intent;
    """

    try:
        # Execute the query and load the aggregated results directly
        df_pr = pd.read_sql(text(pr_query), engine)

        if not df_pr.empty:
            # The plotting code remains the same, as it receives the dataframe it expects
            fig_pr = px.line(df_pr, x="day", y="precision", color="intent", title="Precision Over Time")
            st.plotly_chart(fig_pr, use_container_width=True)
            fig_rc = px.line(df_pr, x="day", y="recall", color="intent", title="Recall Over Time")
            st.plotly_chart(fig_rc, use_container_width=True)
        else:
            st.write("No ground-truth labels logged yet to compute precision/recall.")
    
    except Exception as e:
        st.write("Could not compute precision/recall. Error:", e)


with st.expander("⚠️ Low-Confidence Examples (history of live predictions)"):
    try:
        df_low = pd.read_sql(f"SELECT * FROM {APP_LOW_CONFIDENCE_LOGS_TABLE} ORDER BY confidence ASC LIMIT 10", engine)
        if not df_low.empty:
            st.table(df_low[["ts", "text", "confidence", "slots"]])
        else:
            st.write("No low-confidence messages recorded yet.")
    except Exception as e:
        st.write("Could not read low-confidence table:", e)


st.subheader("🔍 Drill-Down Misclassifications (Test Data)")

if test_data and "texts" in test_data and "labels" in test_data and y_pred is not None:
   
    test_texts = test_data["texts"]

    # Safely convert labels tensor to numpy. utils.tensor_to_numpy now raises on failure & we can catch exceptions and print instead of crashing.
    try:
        y_true = tensor_to_numpy(test_data["labels"])
        # Ensure y_true is a numpy array
        if not hasattr(y_true, "shape"):
            # Fallback if tensor_to_numpy returned something unexpected
            y_true = np.asarray(y_true)
    except Exception as e:
        st.error("Could not convert test_data['labels'] to a numpy array. Error: " + str(e))
        # Use an empty array so the drill down UI doesn't crash
        y_true = np.array([])

    df_errors = pd.DataFrame({"text": test_texts, "true": y_true, "pred": y_pred})


    label_to_id = {label: idx for idx, label in enumerate(available_intents)}
    
    true_options = list(confusion_matrix.index) if confusion_matrix is not None else available_intents
    pred_options = list(confusion_matrix.columns) if confusion_matrix is not None else available_intents

    selected_true = st.selectbox("Actual Intent", options=true_options)
    selected_pred = st.selectbox("Predicted Intent", options=pred_options)

    true_id = label_to_id.get(selected_true)
    pred_id = label_to_id.get(selected_pred)

    if true_id is None or pred_id is None:
        st.error("Selected labels not found in mapping.")
    else:
        df_drill = df_errors[(df_errors.true == true_id) & (df_errors.pred == pred_id)]
        if df_drill.empty:
            st.info("No examples found for this confusion pair.")
        else:
            st.write(df_drill["text"].head(50))
else:
    st.warning("Drill down not available: Required artifacts (test_data.pt, test_preds.npy) could not be loaded.")



with st.expander("🚀 Future Scope & Model Evolution"):
    st.markdown("""
    This dashboard demonstrates a highly effective model on a well-defined task. The following areas represent exciting opportunities for future development:

    * **Expanded Intent Coverage:** The model is currently trained on 14 core intents. A key next step is to broaden its capabilities by incorporating a wider range of user queries and needs, making the system even more versatile.

    * **Integration of More Real-World Data:** This version was built using a high-quality dataset that includes both real-world examples and carefully generated synthetic data. Future iterations will be enhanced by integrating larger volumes of authentic user interactions to further improve its robustness in diverse, real-world scenarios.
                
    * **Deepening Semantic Understanding:** The model currently excels at identifying intents with strong contextual clues, especially  the presence of an invoice  such as the presence of an invoice. A key future enhancement is to have data that has more variations  and train the model to infer intent from more subtle language, reducing its reliance on specific keywords and making it more robust to varied user phrasing.

    * **Scaling for Complexity:** The model's excellent performance on this 20,000+ entry dataset provides a strong baseline. As we scale the system by adding more data and intents, the model will face a more complex and realistic learning challenge. This will naturally lead to more nuanced performance metrics that reflect the ambiguity of real-world conversations.
                
    * **Validated with Rigorous Checks:** The model's high accuracy is confirmed by a series of robust sanity checks, which verify zero data leakage between the training, validation, and test sets. This ensures the performance is genuine and provides a reliable foundation for future enhancements.
    """)

st.write("---")
st.caption("BERT Intent Classification Dashboard - Built with Streamlit")

