import pandas as pd

# Define the path to your artifacts directory using a raw string
ARTIFACTS_DIR = r"D:\Projects\chat-intent-pipeline\artifacts"

# --- The rest of your code ---
for fname in ["train_snapshot.csv","val_snapshot.csv","test_snapshot.csv"]:
    df = pd.read_csv(f"{ARTIFACTS_DIR}/{fname}", dtype=str).fillna("")
    print(fname, "unique group_key:", df.get('group_key', pd.Series()).nunique())

# compute overlaps if group_key column present
train = pd.read_csv(f"{ARTIFACTS_DIR}/train_snapshot.csv", dtype=str).fillna("")
val   = pd.read_csv(f"{ARTIFACTS_DIR}/val_snapshot.csv", dtype=str).fillna("")
test  = pd.read_csv(f"{ARTIFACTS_DIR}/test_snapshot.csv", dtype=str).fillna("")

if 'group_key' in train.columns:
    print("group train∩val:", len(set(train['group_key']) & set(val['group_key'])))
    print("group train∩test:", len(set(train['group_key']) & set(test['group_key'])))