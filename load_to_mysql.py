# load_to_mysql.py
import os
import pandas as pd
from sqlalchemy import create_engine, text

# for the CSV, can be switched between "default dataset" and "gold_dataset", if audit_corrections is run
csv_file = "data/cleaned_all.csv"

# Check if data file exists
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Data file not found: {csv_file}. Run ingest_clean.py first.")
  
df = pd.read_csv(csv_file)
# Select only the 'text' and 'label' columns and only these columns are inserted into the MySQL table.
df = df[['text', 'label']]


# Create SQLAlchemy engine for MySQL
db_url = "mysql+pymysql://intent_user:password:intent_db@localhost:3306/intent_db"
engine = create_engine(db_url)

# Drop table if it exists and then create a new again
with engine.begin() as conn:
    print("Dropping table if exists...")
    conn.execute(text("DROP TABLE IF EXISTS messages"))
    
    print("Creating messages table")
    conn.execute(text("""
CREATE TABLE messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    text LONGTEXT NOT NULL,
    label VARCHAR(100) NOT NULL
)"""))


# Load data into the table
df.to_sql("messages", engine, if_exists="append", index=False)
print(f"Loaded {len(df)} rows into MySQL table 'messages' from {csv_file}.")