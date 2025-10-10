# check_artifact_overlap.py
import torch, numpy as np, os
from config import ARTIFACTS_DIR, TRAIN_DATA_FILE, VAL_DATA_FILE, TEST_DATA_FILE
def load(path):
    if not os.path.exists(path):
        print("missing", path); return None
    return torch.load(path, weights_only=False)
train = load(TRAIN_DATA_FILE)
val = load(VAL_DATA_FILE)
test = load(TEST_DATA_FILE)
def arrs_to_set(arr):
    if arr is None: return set()
    try:
        # input_ids is tensor (N, L)
        out = set()
        for i in range(min(len(arr), 5000)):
            t = arr[i]
            out.add(bytes(t.numpy()))
        return out
    except Exception as e:
        return set()
train_ids = arrs_to_set(train.get("input_ids") if train else None)
val_ids = arrs_to_set(val.get("input_ids") if val else None)
test_ids = arrs_to_set(test.get("input_ids") if test else None)
print("Train size (sampled):", len(train_ids))
print("Val size (sampled):", len(val_ids))
print("Test size (sampled):", len(test_ids))
print("Train∩Val (sampled):", len(train_ids & val_ids))
print("Train∩Test (sampled):", len(train_ids & test_ids))
print("Val∩Test (sampled):", len(val_ids & test_ids))

