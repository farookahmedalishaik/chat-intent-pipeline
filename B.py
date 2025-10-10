# check_placeholder_leak.py
import pandas as pd, re
from preprocess import normalize_placeholders
from config import CLEANED_DATA_FILE
df = pd.read_csv(CLEANED_DATA_FILE, dtype=str).fillna("")
# compute text_trans
df['text_trans'] = df['text'].astype(str).apply(lambda x: normalize_placeholders(x)[0])
# token extractor
ph_re = re.compile(r"\[([A-Z0-9_]+)(?:=[^\]]+)?\]")
def get_ph_tokens(s):
    return set(m.lower() for m in ph_re.findall(s))
df['tokens'] = df['text_trans'].apply(get_ph_tokens)
# For each token, get number of unique labels it appears with and which splits (we don't have splits here)
token2labels = {}
for _, row in df.iterrows():
    for t in row['tokens']:
        token2labels.setdefault(t, set()).add(row['label'])
# tokens that look like hashes
suspicious = {t: len(token2labels[t]) for t in token2labels if 'hash' in t or 'order' in t or 'invoice' in t or 'person' in t}
# print top
for t,c in sorted(suspicious.items(), key=lambda x:-x[1])[:80]:
    print(f"{t} -> labels_count={c}")
# show tokens that appear in only a few labels (very predictive)
predictive = {t:c for t,c in suspicious.items() if c <= 2}
print("Tokens that are very label-specific (<=2 labels):", list(predictive.keys())[:50])
