#sample_for_audit.py

import pandas as pd
df = pd.read_csv("data/cleaned_all.csv")
sample = df.sample(n = 100, random_state = 42)
sample.to_csv("data/audit_sample.csv", index = False)
print("Saved audit sample to data/audit_sample.csv")
