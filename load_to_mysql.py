# load_to_mysql.py (MODIFIED)
import os
import pandas as pd
from sqlalchemy import create_engine, text

# --- Define the source CSV for loading (use gold_dataset if you run audit_corrections) ---
source_csv_path = "data/cleaned_all.csv" # Default to cleaned_all.csv
# If you run audit_corrections.py, change this to:
# source_csv_path = "data/gold_dataset.csv"

# 1. Load the cleaned CSV
if not os.path.exists(source_csv_path):
    raise FileNotFoundError(f"Source data for MySQL not found: {source_csv_path}. Please run ingest_clean.py (and optionally audit_corrections.py) first.")
df = pd.read_csv(source_csv_path)


# --- NEW: Select only the 'text' and 'label' columns ---
# This ensures that only these columns are attempted to be inserted into the MySQL table.
df = df[['text', 'label']]


# 2. Create SQLAlchemy engine for MySQL
conn_str = "mysql+pymysql://intent_user:password:intent_db@localhost:3306/intent_db"
engine = create_engine(conn_str)

# 3. Drop table if it exists and then create it (to ensure a clean load)
with engine.begin() as conn:
    print("Dropping existing 'messages' table (if any)...")
    conn.execute(text("DROP TABLE IF EXISTS messages;"))
    print("Creating new 'messages' table...")
    conn.execute(text("""
        CREATE TABLE messages (
            id INT AUTO_INCREMENT PRIMARY KEY,
            text LONGTEXT NOT NULL,
            label VARCHAR(100) NOT NULL
        );
    """))

# 4. Load data to the (now empty) table
df.to_sql("messages", engine, if_exists="append", index=False)
print(f"Loaded {len(df)} rows into MySQL table 'messages' from {source_csv_path}.")