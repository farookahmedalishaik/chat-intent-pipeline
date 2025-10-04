# insert_synthetic_to_db.py
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from urllib.parse import quote_plus
import datetime

load_dotenv()
MYSQL_USER = os.getenv("MYSQL_USER", "intent_user")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "password")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DB = os.getenv("MYSQL_DB", "intent_db")

encoded_password = quote_plus(MYSQL_PASSWORD)
db_url = f"mysql+pymysql://{MYSQL_USER}:{encoded_password}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
engine = create_engine(db_url)

# CHANGE THIS: path to your reviewed augmented CSV (produced by augmentation pipeline and reviewed)
review_csv = "analysis/augmented_samples_full.csv"
if not os.path.exists(review_csv):
    raise FileNotFoundError(f"Please place your reviewed augmented CSV at {review_csv} (one row per augmented sample).")

df = pd.read_csv(review_csv)
# Expected columns: 'aug' (augmented text), 'class_name' (label string)
if 'aug' not in df.columns:
    # try alternative column names
    if 'augmented' in df.columns:
        df = df.rename(columns={'augmented': 'aug'})
    else:
        raise ValueError("CSV must contain a column named 'aug' (augmented text).")

if 'class_name' not in df.columns and 'class' in df.columns:
    df = df.rename(columns={'class': 'class_name'})

if 'class_name' not in df.columns:
    raise ValueError("CSV must contain a column named 'class_name' with label strings.")

df = df.rename(columns={"aug": "text", "class_name": "label"})
df['is_synthetic'] = True
df['augment_method'] = df.get('mid_lang', df.get('augment_method', 'back_translation'))
df['augment_time'] = datetime.datetime.now().isoformat()
df['source_seed'] = df.get('orig', '')

cols = ['text','label','is_synthetic','augment_method','augment_time','source_seed']
df = df[cols]

# Create synthetic table (if not exists)
with engine.begin() as conn:
    conn.execute(text(f"""
    CREATE TABLE IF NOT EXISTS messages_synthetic (
        id INT AUTO_INCREMENT PRIMARY KEY,
        text LONGTEXT NOT NULL,
        label VARCHAR(200) NOT NULL,
        is_synthetic BOOLEAN,
        augment_method VARCHAR(100),
        augment_time DATETIME,
        source_seed LONGTEXT
    )
    """))
# Insert rows (pandas to_sql)
df.to_sql("messages_synthetic", engine, if_exists="append", index=False)
print(f"Inserted {len(df)} synthetic rows into messages_synthetic.")
