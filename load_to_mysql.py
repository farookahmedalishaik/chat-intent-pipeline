# load_to_mysql.py

import pandas as pd
from sqlalchemy import create_engine, text

# 1. Load the cleaned CSV
df = pd.read_csv("data/cleaned_all.csv")

# 2. Create SQLAlchemy engine for MySQL
conn_str = "mysql+pymysql://intent_user:YourStrongPassword@localhost:3306/intent_db"
engine = create_engine(conn_str)

# 3. Create table if it doesn’t exist
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS messages (
            id INT AUTO_INCREMENT PRIMARY KEY,
            text LONGTEXT NOT NULL,
            label VARCHAR(100) NOT NULL
        );
    """))

# 4. Append data to the table
df.to_sql("messages", engine, if_exists="append", index=False)
print(f"Loaded {len(df)} rows into MySQL table 'messages'.")
