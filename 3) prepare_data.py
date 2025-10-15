# 3) prepare_data.py
"""
- Prepares cleaned data for model training. Reads cleaned human labeled data from MySQL database or falls back to the cleaned CSV.
- Encodes text labels into numbers so the model can use. Splits the data into training, validation, and test sets.
- Extracts slot-presence & sanitized slot tokens (ACCOUNT_TYPE preserved as a token like [ACCOUNT_TYPE_CHECKING],
  sensitive placeholders mapped to presence tokens like [ORDER_PRESENT]).
- Uses GroupShuffleSplit with an entity-derived group key (when available) to avoid entity-level leakage.
- Deduplicates final model input text to remove duplicates introduced by placeholder normalization.
- Tokenizes the text and saves everything as PyTorch files.
"""

import os
import shutil
import json
import re
import hashlib
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from utils import get_db_engine
from collections import Counter  # added for safe stratify fallback

# Import all settings from the central config file
from config import (
    ARTIFACTS_DIR,
    CLEANED_DATA_FILE,
    DB_MESSAGES_TABLE,
    TEST_RATIO,
    VALID_RATIO,
    MAX_TEXT_LENGTH,
    SEED,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_DB,
    BASE_MODEL_ID
)

# ---------------- Helper: Load data (DB first, fallback to cleaned CSV) ----------------
def load_data_from_source():
    """
    Tries to load from the DB table first. If the DB table does not include
    the 'mappings' column (or fails), falls back to reading the local CLEANED_DATA_FILE,
    which is expected to contain 'text', 'label', 'text_raw' and 'mappings'.
    """
    # Try DB first (keeps existing pipeline behavior)
    try:
        engine = get_db_engine(
            MYSQL_USER,
            MYSQL_PASSWORD,
            MYSQL_HOST,
            MYSQL_PORT,
            MYSQL_DB,
            sqlite_fallback_path=None,
            test_connection=True
        )
        df_db = pd.read_sql(text(f"SELECT text, label, mappings, text_raw FROM {DB_MESSAGES_TABLE}"), con=engine)
        print(f" Successfully loaded {len(df_db)} rows from DB table '{DB_MESSAGES_TABLE}'.")
    except Exception as e:
        print(f" Warning: could not load from DB: {e}")
        df_db = None

    # If db loaded but doesn't include mappings, fallback to CSV which includes mappings from ingest_clean.py
    if df_db is not None:
        # We need 'mappings' (dictionary per row) to build entity-aware groups. If it's not there, fallback.
        if 'mappings' in df_db.columns:
            return df_db
        else:
            print(" DB table does not contain 'mappings' column; falling back to reading the cleaned CSV with mappings.")
    else:
        print(" No DB dataframe available; attempting to read cleaned CSV.")

    # Fallback: read cleaned CSV (this file was produced by ingest_clean.py and contains mappings)
    if os.path.exists(CLEANED_DATA_FILE):
        try:
            df_csv = pd.read_csv(CLEANED_DATA_FILE, dtype=str).fillna("")
            print(f" Loaded cleaned CSV with {len(df_csv)} rows from '{CLEANED_DATA_FILE}'.")
            return df_csv
        except Exception as e:
            print(f" CRITICAL: Could not read cleaned CSV: {e}")
            raise RuntimeError("Failed to load data from DB and cleaned CSV.")
    else:
        raise RuntimeError("No valid data source found (DB failed and CLEANED_DATA_FILE missing).")

# ---------------- Helper: extract placeholder info & build model text ----------------
PH_PATTERN = re.compile(r"\[([A-Z0-9_]+)(?:=([^\]]+))?\]", flags=re.IGNORECASE)

def sanitize_token_value(v: str) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = re.sub(r"[^0-9a-z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if s == "":
        s = "unknown"
    return s.upper()

def extract_placeholders_and_build_text(original_text: str):
    """
    Given the normalized text (which may include placeholders like [ACCOUNT_TYPE=pro_account] or [ORDER_ID_PRESENCE]),
    this function:
      - finds placeholders and their values,
      - removes placeholders from the text,
      - builds a small set of feature tokens for model input:
          * For safe slots like ACCOUNT_TYPE -> produce [ACCOUNT_TYPE_<VALUE>]
          * For sensitive slots like ORDER_ID, PERSON_NAME, PHONE, EMAIL -> produce presence tokens like [ORDER_PRESENT]
          * For refund amount bins or date bins, preserve the bin token (e.g., [REFUND_AMT_BIN=50_100])
    Returns: (final_model_text, placeholders_info_dict)
    where placeholders_info_dict contains flags and any extracted sanitized values.
    """
    text = str(original_text or "")
    # Find placeholders
    found = PH_PATTERN.findall(text)
    # found is list of tuples: (NAME, VALUE or '')
    names_vals = [(n.upper(), v if v is not None else "") for n, v in found]

    info = {
        "order_present": False,
        "invoice_present": False,
        "bill_present": False,
        "phone_present": False,
        "email_present": False,
        "person_present": False,
        "account_type": None,
        "refund_amt_bin": None,
        "date_bin": None,
        "other_placeholders": []
    }

    feature_tokens = []

    for name, val in names_vals:
        # Normalize some known names (accept both ORDER_ID and ORDER_ID_PRESENCE etc)
        if "ORDER" in name:
            info["order_present"] = True
        if "INVOICE" in name and "BILL" not in name:
            info["invoice_present"] = True
        if "BILL" in name:
            info["bill_present"] = True
        if "PHONE" in name:
            info["phone_present"] = True
        if "EMAIL" in name:
            info["email_present"] = True
        if ("PERSON" in name) or (name.startswith("PERSON") and "NAME" in name):
            info["person_present"] = True

        # Special cases to keep useful, non-sensitive values
        if name == "ACCOUNT_TYPE":
            # value might look like "standard_account" in your pipeline; sanitize
            if val:
                info["account_type"] = sanitize_token_value(val)
        elif name.startswith("REFUND_AMT_BIN") or name.startswith("REFUND_AMOUNT") or name.startswith("REFUND_AMT"):
            # keep the bin token (clean)
            if val:
                info["refund_amt_bin"] = re.sub(r"\s+", "", val)
        elif name.startswith("DATE_REL_BIN") or name.startswith("DATE_MONTH") or name.startswith("DATE_YEAR") or name.startswith("DATE_ISO"):
            if val:
                info["date_bin"] = f"{name}={val}"
        else:
            # collect other placeholders for debugging/audit
            info["other_placeholders"].append((name, val))

    # Build tokens: account_type -> [ACCOUNT_TYPE_<VAL>] (keeps signal for checking vs savings)
    if info["account_type"]:
        feature_tokens.append(f"[ACCOUNT_TYPE_{info['account_type']}]")

    # Refund bin token if present
    if info["refund_amt_bin"]:
        token = re.sub(r"[^0-9A-Z_a-z=]+", "_", info["refund_amt_bin"])
        feature_tokens.append(f"[REFUND_AMT_BIN={token}]")

    # Date bin token
    if info["date_bin"]:
        token = re.sub(r"[^0-9A-Z_a-z=]+", "_", info["date_bin"])
        feature_tokens.append(f"[{token}]")

    # Presence flags for sensitive placeholders (no unique ids)
    if info["order_present"]:
        feature_tokens.append("[ORDER_PRESENT]")
    if info["invoice_present"]:
        feature_tokens.append("[INVOICE_PRESENT]")
    if info.get("bill_present", False):
        feature_tokens.append("[BILL_PRESENT]")
    if info["phone_present"]:
        feature_tokens.append("[PHONE_PRESENT]")
    if info["email_present"]:
        feature_tokens.append("[EMAIL_PRESENT]")
    if info["person_present"]:
        feature_tokens.append("[PERSON_PRESENT]")

    # Remove placeholders from the text (prevent identity leakage)
    model_text = PH_PATTERN.sub(" ", text)
    # Clean spacing and lowercase (keep consistent with earlier pipeline)
    model_text = re.sub(r"\s+", " ", model_text).strip().lower()

    # Append feature tokens to the end of the input text (keeps model simple, no architecture changes)
    if feature_tokens:
        model_text = f"{model_text} {' '.join(feature_tokens)}".strip()

    # FINAL CLEANUP (important)
    # 1) Remove any bracketed leftover tokens that contain digits or hex-like substrings (unique ids/hashes)
    model_text = re.sub(r"\[[^\]]*(?:\d|[0-9a-f]{6,}|hash)[^\]]*\]", " [ID_PRESENT] ", model_text, flags=re.IGNORECASE)
    # 2) Remove label-like raw tokens: e.g., tokens that follow 'request_' or 'queries_related_' patterns
    model_text = re.sub(r"\b(request_[a-z0-9_/]+|queries_related_[a-z0-9_/]+)\b", " ", model_text, flags=re.IGNORECASE)
    # 3) Normalize whitespace
    model_text = re.sub(r"\s+", " ", model_text).strip()

    return model_text, info

# ---------------- Main Script Execution ----------------

print(" Step 1: Preparing artifacts directory")
if os.path.exists(ARTIFACTS_DIR):
    shutil.rmtree(ARTIFACTS_DIR)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
print(f" Clean artifacts directory created at: '{ARTIFACTS_DIR}'")

print("\n Step 2: Loading and Cleaning Data")
df = load_data_from_source()

# If we still don't have mappings but have the cleaned CSV loaded, ensure mappings column exists (it should)
if 'mappings' not in df.columns:
    # It is OK — we will extract placeholders from the pre-normalized 'text' field itself.
    print(" Note: 'mappings' column not present. Placeholder extraction will be done from the normalized text.")

# Keep only the columns we care about (text + label). Some sources may have text_raw or mappings; keep them if present.
cols_keep = ['text', 'label']
for c in ['text_raw', 'mappings']:
    if c in df.columns:
        cols_keep.append(c)
df = df[cols_keep].dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str).str.strip()
df['label'] = df['label'].astype(str).str.strip()
print(f" Data loaded & cleaned. Total rows are: {len(df)}")

# ---------------- Extract placeholders and build model-ready text ----------------
print("\n Step 3: Extract placeholders -> build model text & extract slot info")

# Apply extraction to every row (vectorized apply)
extracted_texts = []
placeholder_infos = []

for idx, orig in df['text'].items():
    model_text, info = extract_placeholders_and_build_text(orig)
    extracted_texts.append(model_text)
    placeholder_infos.append(info)

df['text_model'] = extracted_texts
df['ph_info'] = placeholder_infos  # store dicts for audit and grouping

# Optionally, create simple boolean columns for quick checks & grouping
df['order_present'] = df['ph_info'].apply(lambda d: d.get('order_present', False))
df['invoice_present'] = df['ph_info'].apply(lambda d: d.get('invoice_present', False))
df['bill_present'] = df['ph_info'].apply(lambda d: d.get('bill_present', False))
df['phone_present'] = df['ph_info'].apply(lambda d: d.get('phone_present', False))
df['email_present'] = df['ph_info'].apply(lambda d: d.get('email_present', False))
df['person_present'] = df['ph_info'].apply(lambda d: d.get('person_present', False))
df['account_type_val'] = df['ph_info'].apply(lambda d: d.get('account_type'))

print(" Example transformed text (first 5):")
for t in df['text_model'].head(5).tolist():
    print("-", t[:200])

# ---------------- Detect conflicts between transformed text and labels ----------------
print("\n Step 4: Detect transformed-text conflicts (same text -> multiple labels)")
conflict_counts = df.groupby('text_model')['label'].nunique()
conflicts = conflict_counts[conflict_counts > 1]
num_conflicts = len(conflicts)
print(f" Transformed-text conflicts (same model-text -> multiple labels): {num_conflicts}")

if num_conflicts > 0:
    try:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        conflict_texts = conflicts.index.tolist()
        conflict_rows = df[df['text_model'].isin(conflict_texts)][['text', 'text_model', 'label']].drop_duplicates()
        conflict_path = os.path.join(ARTIFACTS_DIR, "model_text_conflicts_sample.csv")
        conflict_rows.head(200).to_csv(conflict_path, index=False, encoding="utf-8")
        print(f" -> Saved up to 200 conflicting examples to: {conflict_path}")
    except Exception as e:
        print(" -> Could not save conflicts CSV:", e)

# ---------------- Remove exact final-model-text duplicates (they were created by generic placeholders) ----------------
print("\n Step 5: Dropping exact duplicate model texts (to avoid identical training rows)")
before = len(df)
df = df.drop_duplicates(subset=['text_model'], keep='first').reset_index(drop=True)
after = len(df)
print(f" Dropped {before - after} duplicate rows. Remaining: {after}")

# ---------------- Build entity group_key for GroupShuffleSplit (to prevent entity leakage) ----------------
print("\n Step 6: Building group keys for entity-aware splitting (ORDER/INVOICE/EMAIL/PERSON/PHONE)")
def build_group_key(row):
    # Prefer using explicit mappings (if present in df) otherwise derive from ph_info
    # If mappings is available as a dict or JSON string, try to extract the original sensitive value and hash it.
    mapped = None
    if 'mappings' in row and row['mappings'] not in (None, "", "nan"):
        m = row['mappings']
        try:
            if isinstance(m, str):
                mapped = json.loads(m)
            elif isinstance(m, dict):
                mapped = m
        except Exception:
            # If parsing fails, ignore and fallback to ph_info
            mapped = None

    # Try to extract a stable identifier from mappings (ORDER_ID, INVOICE_NUMBER, BILL_NUMBER, EMAIL, PHONE, PERSON_NAME)
    if isinstance(mapped, dict):
        for k in ("[ORDER_ID]", "ORDER_ID", "[INVOICE_NUMBER]", "INVOICE_NUMBER", "[BILL_NUMBER]", "BILL_NUMBER", "[EMAIL_ADDRESS]", "EMAIL_ADDRESS", "[PHONE_NUMBER]", "PHONE_NUMBER", "[PERSON_NAME]", "PERSON_NAME"):
            vals = mapped.get(k)
            if vals and len(vals) > 0:
                v = str(vals[0])
                # Use a hashed stable id (we do not expose raw values)
                return hashlib.md5(v.encode("utf-8")).hexdigest()

    # Else, if ph_info has flags and maybe values (e.g., account_type not unique), build a composite keyed hash:
    ph = row.get('ph_info', {})
    if isinstance(ph, dict):
        # prefer non-sensitive stable tokens: account_type + invoice/order presence
        parts = []
        if ph.get('account_type'):
            parts.append(f"acct:{ph.get('account_type')}")
        if ph.get('order_present'):
            parts.append("order")
        if ph.get('invoice_present'):
            parts.append("invoice")
        if ph.get('bill_present'):
            parts.append("bill")
        if parts:
            # create deterministic hashed group key
            return hashlib.md5(("|".join(parts)).encode("utf-8")).hexdigest()

    # As last resort, use the model_text's hash (this groups exact duplicates)
    return hashlib.md5(row['text_model'].encode("utf-8")).hexdigest()

# Apply group key
df['group_key'] = df.apply(build_group_key, axis=1)

# ---------------- Encode labels ----------------
print("\n Step 7: Encoding labels")
label_encoder = LabelEncoder()
df['encoded_label'] = label_encoder.fit_transform(df['label'])
num_labels = len(label_encoder.classes_)
label_mapping = pd.DataFrame({"label": label_encoder.classes_, "id": range(num_labels)})
label_mapping_path = os.path.join(ARTIFACTS_DIR, "label_mapping.csv")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
label_mapping.to_csv(label_mapping_path, index=False)
print(f" Labels encoded into {num_labels} classes and saved mapping to: {label_mapping_path}")

# ---------------- Group-aware splitting ----------------
print("\n Step 8: Group-aware splitting (entity aware)")

groups = df['group_key'].astype(str)
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=SEED)
train_idx, test_idx = next(gss.split(df, groups=groups))

train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

# Now split train_df into train/val with GroupShuffleSplit again (relative size)
val_relative = VALID_RATIO / (1.0 - TEST_RATIO) if (1.0 - TEST_RATIO) > 0 else 0.0
if len(train_df) == 0 or val_relative <= 0:
    print("WARNING: Not enough data to create validation split after test split.")
    final_train_df = train_df
    val_df = train_df.iloc[0:0].reset_index(drop=True)
else:
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=SEED)
    train_idx2, val_idx = next(gss_val.split(train_df, groups=train_df['group_key']))
    final_train_df = train_df.iloc[train_idx2].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

X_train = final_train_df['text_model'].tolist()
y_train = final_train_df['encoded_label'].tolist()
X_val   = val_df['text_model'].tolist()
y_val   = val_df['encoded_label'].tolist()
X_test = test_df['text_model'].tolist()
y_test = test_df['encoded_label'].tolist()

print(f" Data split successfully (group-aware):")
print(f"   - Training examples:   {len(X_train)}")
print(f"   - Validation examples: {len(X_val)}")
print(f"   - Test examples:       {len(X_test)}")

# Overlap check (should be zero)
train_set = set(X_train)
val_overlap = sum([1 for t in X_val if t in train_set])
test_overlap = sum([1 for t in X_test if t in train_set or t in set(X_val)])
print(f" Overlap train<->val (should be 0): {val_overlap}")
print(f" Overlap train<->(val|test) (should be 0): {test_overlap}")

# Leakage detection for placeholder tokens crossing splits (audit)
print("\n Leakage check: scanning for suspicious placeholders crossing splits...")
def extract_ph_tokens(texts):
    toks = set()
    for t in texts:
        if not isinstance(t, str): continue
        for m in PH_PATTERN.findall(t):
            toks.add(m[0].lower())
    return toks

train_tokens = extract_ph_tokens(X_train)
val_tokens = extract_ph_tokens(X_val)
test_tokens = extract_ph_tokens(X_test)

suspicious_tokens = {t for t in (train_tokens & (val_tokens | test_tokens)) if "hash" in t or "order" in t or "invoice" in t or "person" in t or "phone" in t or "email" in t}
print(f"Suspicious placeholder tokens in both train and val/test: {len(suspicious_tokens)}")
if suspicious_tokens:
    try:
        with open(os.path.join(ARTIFACTS_DIR, "leakage_suspicious_tokens.txt"), "w", encoding="utf-8") as fh:
            for t in sorted(suspicious_tokens):
                fh.write(t + "\n")
        print("Saved suspicious tokens to artifacts/leakage_suspicious_tokens.txt")
    except Exception as e:
        print("Could not save leakage tokens:", e)

# Guard
if len(X_train) == 0:
    raise RuntimeError("No training examples after group-aware split. Check TEST_RATIO/VALID_RATIO and grouping logic.")

# ---------------- Save snapshots for auditing ----------------
try:
    # Save full dataframes (these include group_key and ph_info so you can audit splits easily)
    final_train_df[['text_model','label','encoded_label','group_key','ph_info']].to_csv(os.path.join(ARTIFACTS_DIR, "train_snapshot.csv"), index=False, encoding="utf-8")
    val_df[['text_model','label','encoded_label','group_key','ph_info']].to_csv(os.path.join(ARTIFACTS_DIR, "val_snapshot.csv"), index=False, encoding="utf-8")
    test_df[['text_model','label','encoded_label','group_key','ph_info']].to_csv(os.path.join(ARTIFACTS_DIR, "test_snapshot.csv"), index=False, encoding="utf-8")
    print(f" Snapshots saved to {ARTIFACTS_DIR} (train/val/test snapshots).")
except Exception as e:
    print(" Could not save snapshots for auditing:", e)

# ---------------- Compute Class Weights ----------------
print("\n Step 9: Computing Class Weights")
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_path = os.path.join(ARTIFACTS_DIR, "class_weights.npy")
np.save(class_weights_path, class_weights)
print(f" Class weights saved to: {class_weights_path}")

# ---------------- Tokenize and Save as .pt files ----------------
print("\n Step 10: Tokenizing and Saving Tensors")
tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_ID)

def tokenize_texts(texts_list):
    if texts_list is None:
        texts_list = []
    if len(texts_list) == 0:
        return {"input_ids": torch.empty((0, MAX_TEXT_LENGTH), dtype=torch.long), "attention_mask": torch.empty((0, MAX_TEXT_LENGTH), dtype=torch.long)}
    return tokenizer(texts_list, padding=True, truncation=True, return_tensors="pt", max_length=MAX_TEXT_LENGTH)

train_encodings = tokenize_texts(X_train)
val_encodings = tokenize_texts(X_val)
test_encodings = tokenize_texts(X_test)

# Save datasets: include 'texts' field (text_model) for error analysis
torch.save({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": torch.tensor(y_train, dtype=torch.long),
    "texts": final_train_df['text_model'].tolist()
}, os.path.join(ARTIFACTS_DIR, "train_data.pt"))

torch.save({
    "input_ids": val_encodings["input_ids"],
    "attention_mask": val_encodings["attention_mask"],
    "labels": torch.tensor(y_val, dtype=torch.long),
    "texts": val_df['text_model'].tolist()
}, os.path.join(ARTIFACTS_DIR, "val_data.pt"))

torch.save({
    "input_ids": test_encodings["input_ids"],
    "attention_mask": test_encodings["attention_mask"],
    "labels": torch.tensor(y_test, dtype=torch.long),
    "texts": test_df['text_model'].tolist()
}, os.path.join(ARTIFACTS_DIR, "test_data.pt"))

print(" All datasets tokenized and saved as .pt files in the 'artifacts' directory.")
print("\n Data preparation complete!")
