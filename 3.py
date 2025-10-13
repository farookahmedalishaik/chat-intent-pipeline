# prints suspicious tokens file and count
import os
p = os.path.join("artifacts","leakage_suspicious_tokens.txt")
if os.path.exists(p):
    print("leakage file exists; head:")
    print(open(p).read().splitlines()[:50])
else:
    print("no leakage_suspicious_tokens.txt found (good).")
