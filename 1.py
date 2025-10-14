import pandas as pd
from config import ARTIFACTS_DIR
for fname in ["train_snapshot.csv","val_snapshot.csv","test_snapshot.csv"]:
    try:
        df = pd.read_csv(f"{ARTIFACTS_DIR}/{fname}", dtype=str).fillna("")
        print(fname, "rows:", len(df))
    except Exception as e:
        print("missing", fname, e)
# overlap counts
t = pd.read_csv(f"{ARTIFACTS_DIR}/train_snapshot.csv", dtype=str).fillna("")
v = pd.read_csv(f"{ARTIFACTS_DIR}/val_snapshot.csv", dtype=str).fillna("")
te= pd.read_csv(f"{ARTIFACTS_DIR}/test_snapshot.csv", dtype=str).fillna("")
print("train∩val:", len(set(t['text_model']) & set(v['text_model'])))
print("train∩test:", len(set(t['text_model']) & set(te['text_model'])))
print("val∩test:", len(set(v['text_model']) & set(te['text_model'])))
