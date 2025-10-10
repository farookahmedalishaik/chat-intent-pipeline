# config.py
# A central place for all project configurations and hyperparameters.

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- 1. Project Structure ---
DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"
ANALYSIS_DIR = "analysis"
LOGS_DIR = "bert_logs"
MODEL_OUTPUT_DIR = "bert_output"

# --- 2. Data Sources & Database ---
RAW_DATA_FILE = os.path.join(DATA_DIR, "raw_customer_data.csv")
CLEANED_DATA_FILE = os.path.join(DATA_DIR, "cleaned_all.csv")
DB_MESSAGES_TABLE = "messages"
DB_STAGING_TABLE = "messages_staging_tmp"

# --- 3. Model & Training Hyperparameters ---
SEED = 42
TEST_RATIO = 0.15
VALID_RATIO = 0.15
MAX_TEXT_LENGTH = 128

# Training hyperparameters (recommended defaults)
LEARNING_RATE = 2e-5         # safe default for 2-4 epochs
BATCH_SIZE = 8               # per-device batch size; fallback to 4 if OOM on your laptop
GRADIENT_ACCUMULATION_STEPS = 2  # effective batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
EPOCHS = 3
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01

# Advanced training options
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 2.0
SAMPLING_STRATEGY = "class_weight"  # options: "sampler", "class_weight", "none"
SAMPLER_REPLACEMENT = False

# --- 4. Core Artifact & Model Paths ---
MODEL_SUBDIR_NAME = "bert_intent_model"
MODEL_DIR_PATH = os.path.join(ARTIFACTS_DIR, MODEL_SUBDIR_NAME)

LABEL_MAPPING_FILE = os.path.join(ARTIFACTS_DIR, "label_mapping.csv")
CLASS_WEIGHTS_FILE = os.path.join(ARTIFACTS_DIR, "class_weights.npy")
CLASS_THRESHOLDS_FILE = os.path.join(ARTIFACTS_DIR, "class_thresholds.json")

TRAIN_DATA_FILE = os.path.join(ARTIFACTS_DIR, "train_data.pt")
VAL_DATA_FILE = os.path.join(ARTIFACTS_DIR, "val_data.pt")
TEST_DATA_FILE = os.path.join(ARTIFACTS_DIR, "test_data.pt")

VAL_METRICS_FILE = os.path.join(ARTIFACTS_DIR, "val_classification_report.csv")
VAL_CONFUSION_FILE = os.path.join(ARTIFACTS_DIR, "val_confusion_matrix.csv")
TEST_PREDS_FILE = os.path.join(ARTIFACTS_DIR, "test_preds.npy")
TEST_METRICS_FILE = os.path.join(ARTIFACTS_DIR, "test_metrics_bert.csv")
TEST_CONFUSION_FILE = os.path.join(ARTIFACTS_DIR, "test_confusion_bert.csv")

LOW_F1_ERRORS_FILE = os.path.join(ARTIFACTS_DIR, "low_f1_errors.csv")
ANALYSIS_MAX_CONFUSION_PAIRS_TO_PRINT = 50
ANALYSIS_MISCLASSIFIED_SAMPLE_FILE = os.path.join(ANALYSIS_DIR, "test_misclassified_sample.csv")

# --- 5. Application (UI) Settings ---
APP_DEFAULT_CONFIDENCE_THRESHOLD = 0.75
APP_FALLBACK_SQLITE_DB_FILE = "runtime_logs.db"
APP_LOW_CONFIDENCE_LOGS_TABLE = "low_conf"
APP_PREDICTION_LOGS_TABLE = "logs"
APP_SIDEBAR_GUIDANCE_TEXT = "This tool classifies messages into predefined intents."

# --- 6. Deployment & Secrets (loaded from .env) ---
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USER = os.getenv("HF_USER")

# The base model to fine-tune
BASE_MODEL_ID = "bert-base-uncased"

HF_MODEL_REPO_NAME = "intent-bert"
HF_DATASET_REPO_NAME = "intent-bert-data-artifacts"

HF_REPO_ID = f"{HF_USER}/{HF_MODEL_REPO_NAME}" if HF_USER else None
HF_DATASET_REPO_ID = f"{HF_USER}/{HF_DATASET_REPO_NAME}" if HF_USER else None

# MySQL Database
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DB = os.getenv("MYSQL_DB")
