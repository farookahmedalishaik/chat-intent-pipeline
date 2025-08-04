# preprocess.py
import re

def clean_text(s: str) -> str:
    """
    Normalize a text string by:
      - Lowercasing and trimming whitespace
      - Removing URLs
      - Removing non‐ASCII characters (e.g. emojis)
      - Collapsing multiple spaces
    **But keep question marks and exclamation points**, since they carry intent.
    """
    s = str(s).lower().strip()
    s = re.sub(r"http\S+", "", s)             # strip URLs
    s = re.sub(r"[^\x00-\x7F]", " ", s)       # replace non-ASCII with space
    # Remove punctuation *except* ? and !
    s = re.sub(r"[“”\"#\$%&'\(\)\*\+,\-\.\/:;<=>@\[\]^_`{\|}~]", "", s)
    s = re.sub(r"\s+", " ", s)                # collapse whitespace
    return s
