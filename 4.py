import torch, numpy as np, os
from config import TRAIN_DATA_FILE, VAL_DATA_FILE, TEST_DATA_FILE
def arrs_to_set(arrpath):
    if not os.path.exists(arrpath): return set()
    d = torch.load(arrpath)
    ids = d.get("input_ids")
    out=set()
    for i in range(min(len(ids),5000)):
        out.add(bytes(ids[i].numpy()))
    return out
tr=arrs_to_set(TRAIN_DATA_FILE)
va=arrs_to_set(VAL_DATA_FILE)
te=arrs_to_set(TEST_DATA_FILE)
print("sizes:", len(tr), len(va), len(te))
print("train∩val:", len(tr&va))
print("train∩test:", len(tr&te))
print("val∩test:", len(va&te))
