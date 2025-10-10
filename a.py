# check_overlap_texts.py
import pandas as pd
from config import CLEANED_DATA_FILE
df = pd.read_csv(CLEANED_DATA_FILE, dtype=str).fillna("")
# make sure text is transformed the same way as pipeline: use preprocess.normalize_placeholders if available
from preprocess import normalize_placeholders
df['text_trans'] = df['text'].astype(str).apply(lambda x: normalize_placeholders(x)[0])
# save counts
from collections import Counter
c = Counter(df['text_trans'])
dups = [t for t,count in c.items() if count>1]
print("Unique transformed texts:", len(c))
print("Exact transformed duplicates:", len(dups))
# if duplicates exist, print some example texts
if dups:
    print("Sample duplicate transformed text examples (first 10):")
    for s in dups[:10]:
        print("-", s[:200])
