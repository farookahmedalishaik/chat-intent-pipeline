# 2) load_to_mysql.py

import os
import hashlib
import pandas as pd
from sqlalchemy import create_engine, text, types as sqltypes
from urllib.parse import quote_plus
import argparse
from pathlib import Path
from utils import get_db_engine

# Import required things from config file
from config import (
    CLEANED_DATA_FILE,
    DB_MESSAGES_TABLE,
    DB_STAGING_TABLE,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_DB
)

# 1. Configuration
HASH_COLUMN = "md5_hash"

# CLI: dry-run flag
DRY_RUN = False

# If in future decide to keep backups thats why having a small helpers folder exists here for later (but not used now)
Path("backups").mkdir(exist_ok=True)


# 2. Helper Functions

def compute_md5(text_series):
    """Compute MD5 for each entry in a pandas Series (returns a Series)."""
    return text_series.astype(str).apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())


def setup_database_table(conn):
    """
    Ensure main table exists and if the hash column is missing, add it.
    Populate missing hash values for existing rows, and attempt to create a UNIQUE index.
    """
    print(f"Ensuring table '{DB_MESSAGES_TABLE}' exists (and hash column/index are present if possible)...")

    # Create the table if it does not exist (but this won't change an existing table)
    create_table_sql = text(f"""
        CREATE TABLE IF NOT EXISTS {DB_MESSAGES_TABLE} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            text LONGTEXT NOT NULL,
            label VARCHAR(255) NOT NULL,
            {HASH_COLUMN} VARCHAR(32),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NULL ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    conn.execute(create_table_sql)
    print("Table existence ensured.")

    # Check if the hash column exists/not (simple information_schema lookup)
    try:
        col_check = conn.execute(
            text("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_schema = :db AND table_name = :table AND column_name = :col
            """),
            {"db": MYSQL_DB, "table": DB_MESSAGES_TABLE, "col": HASH_COLUMN}
        ).fetchone()[0]
    except Exception as e:
        # If information schema query fails for some reason this is non fatal
        print("Warning: could not check information_schema for column existence:", e)
        col_check = 1  # assume exists to avoid attempting an ALTER on older MySQL that might not support it

    if int(col_check) == 0:
        # Add the hash column if it is missing
        try:
            conn.execute(text(f"ALTER TABLE {DB_MESSAGES_TABLE} ADD COLUMN {HASH_COLUMN} VARCHAR(32) DEFAULT NULL;"))
            print(f"Added missing column '{HASH_COLUMN}' to '{DB_MESSAGES_TABLE}'.")
        except Exception as e:
            print(f"Warning: failed to add column '{HASH_COLUMN}': {e}")

    # Re check the presence of column — fail if it still doesn't exist (critical)
    try:
        col_check2 = conn.execute(
            text("""
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_schema = :db AND table_name = :table AND column_name = :col
            """),
            {"db": MYSQL_DB, "table": DB_MESSAGES_TABLE, "col": HASH_COLUMN}
        ).fetchone()[0]
    except Exception as e:
        print("Warning: second information_schema check failed:", e)
        col_check2 = 0

    if int(col_check2) == 0:
        # downstream logic expects the hash column to exist for dedupe/upsert
        raise RuntimeError(f"Critical: could not ensure required column '{HASH_COLUMN}' exists on table '{DB_MESSAGES_TABLE}'. Check DB permissions and schema.")

    # Populate missing hash values using SQL MD5 for performance (if any rows exist without it)
    try:
        conn.execute(text(f"UPDATE {DB_MESSAGES_TABLE} SET {HASH_COLUMN} = MD5(text) WHERE {HASH_COLUMN} IS NULL OR {HASH_COLUMN} = '';"))
        print("Populated missing hash values (where applicable).")
    except Exception as e:
        print("Warning: could not populate missing hashes:", e)

    # Try to create a unique index on the hash column. If duplicates exist this will fail.
    try:
        conn.execute(text(f"CREATE UNIQUE INDEX ux_{DB_MESSAGES_TABLE}_{HASH_COLUMN} ON {DB_MESSAGES_TABLE}({HASH_COLUMN});"))
        print(f"Created UNIQUE index on {HASH_COLUMN}.")
    except Exception as e:
        # If index creation fails most likeyly duplicates exist and print error
        print(f"Note: could not create UNIQUE index on {HASH_COLUMN} (may already exist or duplicates present): {e}")


def run_global_deduplication(conn):
    """
    Remove older duplicate rows like same hash, lower id. Respects DRY_RUN.
    This function will delete duplicates without backup.
    """
    print("Checking for and removing any old duplicate entries...")
    # The delete SQL removes older rows when a newer one exists with the same hash.
    delete_sql = text(f"""
        DELETE m1 FROM {DB_MESSAGES_TABLE} m1
        JOIN {DB_MESSAGES_TABLE} m2 ON m1.{HASH_COLUMN} = m2.{HASH_COLUMN} AND m1.id < m2.id
        WHERE m1.{HASH_COLUMN} IS NOT NULL;
    """)

    if DRY_RUN:
        print("[DRY-RUN] Would check for and remove older duplicates (no changes made).")
        return

    try:
        result = conn.execute(delete_sql)
        # result.rowcount may be None depending on driver; print a generic successful execution message
        print("Duplicate cleanup executed. Older duplicates are removed wherever they were present.")
    except Exception as e:
        print("Warning: deduplication delete failed:", e)


# 3. Main Execution Logic

def main():
    """Main function to run the data loading pipeline."""
    global DRY_RUN

    parser = argparse.ArgumentParser(description="A simple, robust script to load CSV data into MySQL.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without changing the database.")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    if DRY_RUN:
        print("RUNNING IN DRY-RUN MODE: No changes will be made to the database.")

    print("\nStep 1: Reading and preparing CSV data...")
    if not os.path.exists(CLEANED_DATA_FILE):
        raise FileNotFoundError(f"Error: The input file was not found at '{CLEANED_DATA_FILE}'")

    df = pd.read_csv(CLEANED_DATA_FILE, dtype=str).fillna("")
    if "text" not in df.columns or "label" not in df.columns:
        raise KeyError("CSV must have 'text' and 'label' columns.")

    # Ensure text_raw column exists (critical for stable MD5 dedupe)
    if "text_raw" not in df.columns:
        raise KeyError("Input cleaned CSV is missing required column 'text_raw' used to compute stable MD5 dedupe. Aborting.")

    # Pre flight check for duplicates in the source file
    duplicates = df[df.duplicated(subset=['text_raw'], keep=False)]
    if not duplicates.empty:
        print("\n WARNING: Found duplicate raw text entries in the source file.")
        print("The following entries appear more than once:")
        # Show which text is duplicated and how many times
        print(duplicates['text_raw'].value_counts())
        print("These will be automatically removed, keeping only the first instance.")

        df.drop_duplicates(subset=['text_raw'], keep='first', inplace=True)
        print(f"\nDuplicates have been removed. Proceeding with {len(df)} unique rows.")


    df[HASH_COLUMN] = compute_md5(df["text_raw"]) #calculates the hash based on the ORIGINAL text, ensuring true uniqueness
    print(f"Loaded {len(df)} rows from '{CLEANED_DATA_FILE}'.")

    # Create engine using centralized helper from utils
    engine = get_db_engine(
        MYSQL_USER,
        MYSQL_PASSWORD,
        MYSQL_HOST,
        MYSQL_PORT,
        MYSQL_DB,
        sqlite_fallback_path=None  
    )

    # Use a transaction so setup + staging + upsert + dedupe are atomic
    with engine.begin() as conn:
        # Ensure table exists, add/populate hash column if needed, and attempt index
        setup_database_table(conn)

        print("\nStep 2: Loading all CSV data to a temporary staging table...")

        if DRY_RUN:
            print(f"[DRY-RUN] Would load {len(df)} rows to staging and then perform a scalable upsert.")
        else:
            # 1. Load the ENTIRE dataframe into the staging table.
            df.to_sql(
                DB_STAGING_TABLE,
                conn,
                if_exists='replace',
                index=False,
                dtype={'text': sqltypes.Text(), 'label': sqltypes.String(255), HASH_COLUMN: sqltypes.String(32)}
            )
            print(f"Loaded {len(df)} rows into the temporary staging table '{DB_STAGING_TABLE}'.")

            # Using separate UPDATE and INSERT statements for clarity and compatibility
            print("\nStep 3: Updating existing records from the staging table...")

            # 2. UPDATE STEP: Update rows in the main table that have a matching hash in the staging table.
            update_sql = text(f"""
                UPDATE {DB_MESSAGES_TABLE} mt
                JOIN {DB_STAGING_TABLE} st ON mt.{HASH_COLUMN} = st.{HASH_COLUMN}
                SET
                    mt.text = st.text,
                    mt.label = st.label,
                    mt.updated_at = NOW()
                WHERE
                    -- Optional: only update rows where the data has actually changed to be more efficient.
                    mt.text != st.text OR mt.label != st.label;
            """)
            update_result = conn.execute(update_sql)
            print(f"Updated {update_result.rowcount} existing rows.")


            print("\nStep 4: Inserting new records from the staging table...")

            # 3. INSERT STEP: Insert rows from the staging table that do NOT have a matching hash in the main table.
            insert_sql = text(f"""
                INSERT INTO {DB_MESSAGES_TABLE} (text, label, {HASH_COLUMN})
                SELECT
                    st.text,
                    st.label,
                    st.{HASH_COLUMN}
                FROM
                    {DB_STAGING_TABLE} st
                LEFT JOIN
                    {DB_MESSAGES_TABLE} mt ON st.{HASH_COLUMN} = mt.{HASH_COLUMN}
                WHERE
                    mt.{HASH_COLUMN} IS NULL;
            """)
            insert_result = conn.execute(insert_sql)
            print(f"Inserted {insert_result.rowcount} new rows.")

            # 4. Drop the staging table as before
            conn.execute(text(f"DROP TABLE IF EXISTS {DB_STAGING_TABLE};"))
            print("Temporary staging table removed.")


        # Finally run dedupe to remove older duplicates (no preview/backups)
        run_global_deduplication(conn)

    print("\n--- Process complete! ---")


if __name__ == "__main__":
    main()
