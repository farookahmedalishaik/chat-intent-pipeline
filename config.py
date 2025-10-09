# config.py
# A central place for all project configurations and hyperparameters.

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. Project Structure ---
# Core directories for the project's data, artifacts, and outputs.
DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"
ANALYSIS_DIR = "analysis"
LOGS_DIR = "bert_logs"
MODEL_OUTPUT_DIR = "bert_output"

# --- 2. Data Sources & Database ---
# Input data files and database table names.
RAW_DATA_FILE = os.path.join(DATA_DIR, "raw_customer_data.csv")
CLEANED_DATA_FILE = os.path.join(DATA_DIR, "cleaned_all.csv")
DB_MESSAGES_TABLE = "messages"
DB_STAGING_TABLE = "messages_staging_tmp"

# --- 3. Model & Training Hyperparameters ---
# Settings for data preparation and model fine tuning.
SEED = 42
TEST_RATIO = 0.10
VALID_RATIO = 0.10
MAX_TEXT_LENGTH = 128
LEARNING_RATE = 3e-5
BATCH_SIZE = 16
EPOCHS = 6
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01

# Advanced training options
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
SAMPLING_STRATEGY = "sampler" # options: "sampler", "class_weight", "none"
SAMPLER_REPLACEMENT = False

# --- 4. Core Artifact & Model Paths ---
# Full paths to all generated files, models, and datasets.
MODEL_SUBDIR_NAME = "bert_intent_model"
MODEL_DIR_PATH = os.path.join(ARTIFACTS_DIR, MODEL_SUBDIR_NAME)

# Key artifacts for training and inference
LABEL_MAPPING_FILE = os.path.join(ARTIFACTS_DIR, "label_mapping.csv")
CLASS_WEIGHTS_FILE = os.path.join(ARTIFACTS_DIR, "class_weights.npy")
CLASS_THRESHOLDS_FILE = os.path.join(ARTIFACTS_DIR, "class_thresholds.json")

# Processed datasets
TRAIN_DATA_FILE = os.path.join(ARTIFACTS_DIR, "train_data.pt")
VAL_DATA_FILE = os.path.join(ARTIFACTS_DIR, "val_data.pt")
TEST_DATA_FILE = os.path.join(ARTIFACTS_DIR, "test_data.pt")

# Evaluation and analysis outputs
VAL_METRICS_FILE = os.path.join(ARTIFACTS_DIR, "val_classification_report.csv")
VAL_CONFUSION_FILE = os.path.join(ARTIFACTS_DIR, "val_confusion_matrix.csv")
TEST_PREDS_FILE = os.path.join(ARTIFACTS_DIR, "test_preds.npy")
TEST_METRICS_FILE = os.path.join(ARTIFACTS_DIR, "test_metrics_bert.csv")
TEST_CONFUSION_FILE = os.path.join(ARTIFACTS_DIR, "test_confusion_bert.csv")


LOW_F1_ERRORS_FILE = os.path.join(ARTIFACTS_DIR, "low_f1_errors.csv")
ANALYSIS_MAX_CONFUSION_PAIRS_TO_PRINT = 50
ANALYSIS_MISCLASSIFIED_SAMPLE_FILE = os.path.join(ANALYSIS_DIR, "test_misclassified_sample.csv")

# --- 5. Application (UI) Settings ---
# Settings for the Streamlit application (app.py)
APP_DEFAULT_CONFIDENCE_THRESHOLD = 0.75
APP_FALLBACK_SQLITE_DB_FILE = "runtime_logs.db"
APP_LOW_CONFIDENCE_LOGS_TABLE = "low_conf"
APP_PREDICTION_LOGS_TABLE = "logs"
APP_SIDEBAR_GUIDANCE_TEXT = "This tool classifies messages into predefined intents."


# --- 6. Deployment & Secrets (loaded from .env) ---
# Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USER = os.getenv("HF_USER")


# The base model to fine-tune
BASE_MODEL_ID = "bert-base-uncased"

# Define separate repo names for the model and the dataset artifacts
HF_MODEL_REPO_NAME = "intent-bert"
HF_DATASET_REPO_NAME = "intent-bert-data-artifacts"

# Automatically construct the full repository IDs from the variables above
HF_REPO_ID = f"{HF_USER}/{HF_MODEL_REPO_NAME}" if HF_USER else None
HF_DATASET_REPO_ID = f"{HF_USER}/{HF_DATASET_REPO_NAME}" if HF_USER else None


# MySQL Database
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DB = os.getenv("MYSQL_DB")