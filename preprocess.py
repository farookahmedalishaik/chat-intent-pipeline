"""
Model-based preprocessing using spaCy + Microsoft Presidio.
- Detects PERSON, GPE/LOC, MONEY, DATE, EMAIL, PHONE, ORG, etc.
- Replaces detected spans with uppercase placeholders like [PERSON_NAME]
- Returns (final_text, mappings) where mappings is a dict: placeholder -> [values]
"""

import re
import hashlib
from typing import Tuple, Dict, List
from collections import defaultdict

# external libs
import spacy
from spacy.pipeline import EntityRuler

from presidio_analyzer import AnalyzerEngine


def clean_text(s: str) -> str:
    """small cleanup: remove URLs and excessive whitespace; keep ascii characters."""
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"http\S+", "", s)  # remove URLs (simple)
    s = re.sub(r"[^\x00-\x7F]", " ", s)  # replace non-ascii with spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ---------- Model initialization (do this once) ----------
_nlp = None
_analyzer = None


def init_models(use_optional_order_invoice_patterns: bool = True):
    """
    Load spaCy & Presidio Analyzer. This is done on first call. And set use_optional_order_invoice_patterns=False to skip the simple ORDER/INV patterns.
    """
    global _nlp, _analyzer
    if _nlp is None:
        # load small English model
        _nlp = spacy.load("en_core_web_sm")

        # Optional: add a small EntityRuler to catch common order/invoice ID formats.
        if use_optional_order_invoice_patterns:
            ruler = _nlp.add_pipe("entity_ruler", before="ner")
            patterns = [
                {"label": "ORDER_ID", "pattern": [
                    {"LOWER": {"IN": ["order", "purchase", "ord"]}},
                    {"IS_PUNCT": True, "OP": "?"},
                    {"TEXT": {"REGEX": "^.*[0-9].*$"}, "IS_PUNCT": False}
                ]},

                # Invoice-looking tokens (do NOT include 'bill' here)
                {"label": "INVOICE_NUMBER", "pattern": [
                    {"LOWER": {"IN": ["invoice", "inv"]}},
                    {"IS_PUNCT": True, "OP": "?"},
                    {"TEXT": {"REGEX": "^.*[0-9].*$"}, "IS_PUNCT": False}
                ]},
                # Bill tokens handled separately
                {"label": "BILL_NUMBER", "pattern": [
                    {"LOWER": {"IN": ["bill", "billing", "billno", "bno"]},},
                    {"IS_PUNCT": True, "OP": "?"},
                    {"TEXT": {"REGEX": "^.*[0-9].*$"}, "IS_PUNCT": False}
                ]},


                {"label": "ACCOUNT_TYPE", "pattern": [{"LOWER": {"IN": ["pro", "standard", "freemium", "platinum", "gold", "savings", "checking"]}}, {"LOWER": "account"}]},
                {"label": "GPE", "pattern": [{"LOWER": "kent"}]},
                {"label": "GPE", "pattern": [{"LOWER": "streetsboro"}]},
                {"label": "STOP_ENTITY", "pattern": [{"LOWER": "eta"}]},
                {"label": "STOP_ENTITY", "pattern": [{"LOWER": "modify"}]},
                {"label": "STOP_ENTITY", "pattern": [{"LOWER": "cancel"}]},
                {"label": "STOP_ENTITY", "pattern": [{"LOWER": "signup"}]},
                {"label": "STOP_ENTITY", "pattern": [{"LOWER": "lodge"}]},
                {"label": "STOP_ENTITY", "pattern": [{"LOWER": "solve"}]},
                {"label": "STOP_ENTITY", "pattern": [{"LOWER": "delete"}]},
            ]
            ruler.add_patterns(patterns)

    if _analyzer is None:
        _analyzer = AnalyzerEngine()
    return _nlp, _analyzer


# ---------- Mapping from spaCy/Presidio entity types to our placeholders ----------
_ENTITY_TO_PLACEHOLDER = {
    "PERSON": "[PERSON_NAME]",
    "GPE": "[DELIVERY_LOCATION]",
    "LOC": "[DELIVERY_LOCATION]",
    "MONEY": "[REFUND_AMOUNT]",
    "DATE": "[DATE]",
    "DATE_TIME": "[DATE_TIME]",
    "ORG": "[ORGANIZATION]",
    "PRODUCT": "[PRODUCT]",
    "ACCOUNT_TYPE": "[ACCOUNT_TYPE]",

    "PHONE_NUMBER": "[PHONE_NUMBER]",
    "EMAIL_ADDRESS": "[EMAIL_ADDRESS]",
    "CREDIT_CARD": "[CREDIT_CARD]",
    "IBAN_CODE": "[IBAN]",
    "US_BANK_NUMBER": "[BANK_NUMBER]",
    "US_DRIVER_LICENSE": "[US_DRIVER_LICENSE]",

    "ORDER_ID": "[ORDER_ID]",
    "INVOICE_NUMBER": "[INVOICE_NUMBER]",
    "BILL_NUMBER": "[BILL_NUMBER]",

    "STOP_ENTITY": "[STOP_ENTITY]",
}

# -------------- Placeholder handling policy --------------
# Whitelist placeholders we will keep raw/sanitized values for (non-sensitive):
_WHITELIST_VALUE_PHS = {"ACCOUNT_TYPE", "DELIVERY_LOCATION"}  # keep readable value (safe)
# Placeholders considered semi-sensitive: we will NOT expose their unique values in the text (avoid leakage)
_HASHED_VALUE_PHS = {"ORDER_ID", "INVOICE_NUMBER", "BILL_NUMBER"}
# Sensitive placeholders (do not expose values in the text)
_SENSITIVE_PH_STRINGS = {"PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"}

# Amount bins for REFUND_AMOUNT
_AMOUNT_BINS = [(0, 10), (10, 50), (50, 100), (100, 500), (500, 1000), (1000, 1_000_000)]

# Date recency bins (in months)
_DATE_RECENCY_BINS_MONTHS = [(0, 1), (1, 6), (6, 12), (12, 1200)]  # 0-1, 1-6, 6-12, 12+
_DATE_RECENCY_LABELS = ["0_30days", "31_180days", "181_365days", "365_plus"]


def _amount_to_bin_label(num: float) -> str:
    for low, high in _AMOUNT_BINS:
        if low <= num < high:
            return f"{low}_{high}"
    return f"{_AMOUNT_BINS[-1][1]}_plus"


def _date_rel_to_bin_label(months: int) -> str:
    # months is positive integer representing how many months ago
    for (low, high), lab in zip(_DATE_RECENCY_BINS_MONTHS, _DATE_RECENCY_LABELS):
        if low <= months < high:
            return lab
    return _DATE_RECENCY_LABELS[-1]


# helper to detect month names
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}


def _normalize_date_value(value_raw: str) -> str:
    """
    Heuristic normalization of DATE/DATE_TIME strings into useful, privacy-preserving tokens.
    Returns a string like:
      - "[DATE_REL_BIN=6_12]" for relative phrases (e.g., '9 months ago')
      - "[DATE_MONTH=june]" for month tokens
      - "[DATE_YEAR=2023]" for year-like tokens
      - "[DATE_TIME_QUESTION]" for question-like phrases such as 'what hours'
      - "[DATE]" fallback
    """
    if not value_raw or not value_raw.strip():
        return "[DATE]"

    v = value_raw.strip().lower()

    # 1) question-like phrases (what time, what hours, when are)
    if re.search(r'\b(what|when|which|how)\b', v) and re.search(r'\b(hour|hours|time|when)\b', v):
        return "[DATE_TIME_QUESTION]"

    # 2) relative months e.g., '9 months ago', 'ten months ago', 'last month', 'a month ago'
    m = re.search(r'(\d+)\s+months?\s+ago', v)
    if m:
        months = int(m.group(1))
        return f"[DATE_REL_BIN={_date_rel_to_bin_label(months)}]"
    # spelled numbers like 'ten months ago' (basic)
    spelled = re.search(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b\s+months?\s+ago', v)
    if spelled:
        word_to_num = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12}
        months = word_to_num.get(spelled.group(1), None)
        if months is not None:
            return f"[DATE_REL_BIN={_date_rel_to_bin_label(months)}]"

    if re.search(r'\blast\s+month\b', v) or re.search(r'\bthis\s+month\b', v):
        return f"[DATE_REL_BIN={_date_rel_to_bin_label(1)}]"

    # 3) 'X months ago' with number words like '9 months ago' handled; also 'years ago'
    m_years = re.search(r'(\d+)\s+years?\s+ago', v)
    if m_years:
        years = int(m_years.group(1))
        months = years * 12
        return f"[DATE_REL_BIN={_date_rel_to_bin_label(months)}]"

    # 4) Month names like 'june'
    for mn in _MONTHS.keys():
        if re.search(r'\b' + re.escape(mn) + r'\b', v):
            return f"[DATE_MONTH={mn}]"

    # 5) Year-like tokens '2023' or '23' (simple)
    y = re.search(r'\b(19|20)\d{2}\b', v)
    if y:
        return f"[DATE_YEAR={y.group(0)}]"

    # 6) ISO-like or numeric dates with slashes or dashes e.g. 2023-06-15 or 06/15/2023
    d = re.search(r'(\d{1,4}[/-]\d{1,2}[/-]\d{1,4})', v)
    if d:
        # sanitize and keep short representation
        sanitized = re.sub(r'[^0-9/-]', '', d.group(1))
        return f"[DATE_ISO={sanitized}]"

    # 7) If the token is short and purely numeric (e.g., '9'), consider year or month depending on length
    if re.fullmatch(r'\d{1,4}', v):
        if len(v) == 4 and (1900 <= int(v) <= 2100):
            return f"[DATE_YEAR={v}]"
        # else fallback
        return "[DATE]"

    # 8) fallback
    return "[DATE]"


# ---------- Core function: normalize_placeholders ----------
def normalize_placeholders(text: str) -> Tuple[str, Dict[str, List[str]]]:
    """
    Replace recognized spans in `text` with placeholders and return (final_text, mappings).
    - Uses spaCy NER + Presidio Analyzer.
    - Preserves some non-sensitive entity values (e.g., account type),
      but DOES NOT expose unique per-entity values or short-hashes in the text stream.
    - Normalizes dates into bins/month/year/question tokens using heuristics.
    """
    if text is None:
        return "", {}

    raw = clean_text(text)
    nlp, analyzer = init_models()

    spans = []  # will hold dicts: {"start":int, "end":int, "ph":str, "value":str, "score":float}

    # 1) Presidio analysis
    try:
        presidio_results = analyzer.analyze(text=raw, language="en")
        for r in presidio_results:
            start, end = r.start, r.end
            ent_type = r.entity_type
            ph = _ENTITY_TO_PLACEHOLDER.get(ent_type, f"[{ent_type}]")
            spans.append({"start": start, "end": end, "ph": ph, "value": raw[start:end], "score": getattr(r, "score", None)})
    except Exception:
        presidio_results = []

    # 2) spaCy NER
    doc = nlp(raw)
    for ent in doc.ents:
        ent_type = ent.label_
        ph = _ENTITY_TO_PLACEHOLDER.get(ent_type)
        if ph is None:
            continue
        spans.append({"start": ent.start_char, "end": ent.end_char, "ph": ph, "value": ent.text, "score": None})

    # if no spans found, return normalized lowercased text and empty mappings
    if not spans:
        cleaned = re.sub(r"\s+", " ", raw).strip().lower()
        return cleaned, {}

    # 3) Filter and merge spans (remove STOP_ENTITY)
    filtered_spans = [s for s in spans if s['ph'] != '[STOP_ENTITY]']
    spans_sorted = sorted(filtered_spans, key=lambda x: (x["start"], -(x["score"] or 0)))
    merged = []
    last_end = -1
    for s in spans_sorted:
        if s["start"] >= last_end:
            merged.append(s)
            last_end = s["end"]

    # 4) Build final text by replacing spans with placeholders (with values or generic tokens) and collect mappings
    mappings = defaultdict(list)
    out_pieces = []
    last_idx = 0

    for s in merged:
        # add text before the placeholder
        pre_text = raw[last_idx:s["start"]].lower().strip()
        if pre_text:
            out_pieces.append(pre_text)

        ph_full = s['ph']  # e.g., "[ACCOUNT_TYPE]"
        ph_name = ph_full.strip("[]")
        value_raw = (s.get("value") or "").strip()

        chosen_piece = ph_full  # default

        if not value_raw:
            chosen_piece = ph_full
        else:
            # sanitize value for non-sensitive cases
            safe_val = re.sub(r"\s+", "_", value_raw.lower())

            # try numeric amount parsing for REFUND_AMOUNT
            amount = None
            if ph_name == "REFUND_AMOUNT":
                cleaned_amount = re.sub(r"[^\d\.\-]", "", value_raw)
                try:
                    amount = float(cleaned_amount) if cleaned_amount not in ("", "-", ".") else None
                except Exception:
                    amount = None

            # Decision logic: **do not** include unique per-entity values or short-hashes in the text stream.
            # Keep safe readable small values for whitelisted placeholders only.
            if ph_name in _WHITELIST_VALUE_PHS:
                # keep readable sanitized value for safe placeholders
                chosen_piece = f"[{ph_name}={safe_val}]"
            elif ph_name in _HASHED_VALUE_PHS:
                # Use a presence indicator token instead of exposing hashed values
                chosen_piece = f"[{ph_name}_PRESENCE]"
            elif ph_name in _SENSITIVE_PH_STRINGS:
                # hide PII completely in the token stream (no id/hash/value in the text)
                chosen_piece = f"[{ph_name}_PRESENCE]"
            elif ph_name == "REFUND_AMOUNT" and amount is not None:
                bin_label = _amount_to_bin_label(amount)
                chosen_piece = f"[REFUND_AMT_BIN={bin_label}]"
            elif ph_name in {"DATE", "DATE_TIME"}:
                # Use heuristic normalization for dates
                date_token = _normalize_date_value(value_raw)
                chosen_piece = date_token
            else:
                # fallback: keep placeholder only
                chosen_piece = ph_full

        out_pieces.append(chosen_piece)
        mappings[s['ph']].append(s['value'])
        last_idx = s["end"]

    # add final tail text
    post_text = raw[last_idx:].lower().strip()
    if post_text:
        out_pieces.append(post_text)

    final_text = " ".join(out_pieces)

    # Final sanitization: only target truly ID-like bracketed tokens (not short suffixes like pro_account_2)
    def _is_id_like_token(tok: str) -> bool:
        t = tok.lower()
        # explicit "hash" mention (e.g., order_hash)
        if "hash" in t:
            return True
        # long hex (>=8 hex chars)
        if re.search(r"\b[0-9a-f]{8,}\b", t):
            return True
        # long numeric runs (>=6 digits)
        if re.search(r"\d{6,}", t):
            return True
        # patterns like inv-12345 or ord_12345 (letters + separator + >=4 digits)
        if re.search(r"\b(?:inv|invoice|invno|ord|order|orderid|bill)[-_]?\d{4,}\b", t):
            return True
        return False

    def _replace_id_like(match):
        inner = match.group(0)  # e.g. "[ACCOUNT_TYPE=pro_account_2]"
        content = inner.strip("[]")
        return " [ID_PRESENT] " if _is_id_like_token(content) else inner

    # Replace only ID-like bracketed tokens
    final_text = re.sub(r"\[[^\]]+\]", lambda m: _replace_id_like(m), final_text, flags=re.IGNORECASE)

    # Normalize whitespace and lower-case
    final_text = re.sub(r"\s+", " ", final_text).strip().lower()

    # Deduplicate mapping lists while preserving order
    final_mappings = {}
    for k, vals in mappings.items():
        seen = {}
        deduped = []
        for v in vals:
            if v not in seen:
                deduped.append(v)
                seen[v] = True
        final_mappings[k] = deduped

    return final_text, final_mappings


# ---------- small helper for convenience ----------
def extract_slots(raw_text: str) -> Dict[str, List[str]]:
    """Return only the detected mappings (useful helper)."""
    _, mappings = normalize_placeholders(raw_text)
    return mappings


# ---------- Example run when executed directly ----------
if __name__ == "__main__":
    example = "Can you check order ORD-12345 for John Doe delivered to New York? Refund $10.99. Invoice INV-98765 also. Call me at +1 555-24-9999 or email alice@example.com."
    final_text, mappings = normalize_placeholders(example)
    print("FINAL TEXT:")
    print(final_text)
    print("\nMAPPINGS:")
    for k, v in mappings.items():
        print(k, "->", v)
