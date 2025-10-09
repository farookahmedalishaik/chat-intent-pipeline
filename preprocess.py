# preprocess.py
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
                {"label": "INVOICE_NUMBER", "pattern": [
                    {"LOWER": {"IN": ["invoice", "bill", "inv"]}},
                    {"IS_PUNCT": True, "OP": "?"},
                    {"TEXT": {"REGEX": "^.*[0-9].*$"}, "IS_PUNCT": False}
                ]},
                {"label": "ACCOUNT_TYPE", "pattern": [{"LOWER": {"IN": ["pro", "standard", "freemium", "platinum", "gold"]}}, {"LOWER": "account"}]},
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

    "STOP_ENTITY": "[STOP_ENTITY]",
}

# -------------- Placeholder handling policy --------------
# Whitelist placeholders we will keep raw/sanitized values for (non-sensitive):
_WHITELIST_VALUE_PHS = {"ACCOUNT_TYPE", "DELIVERY_LOCATION"}  # keep readable value (safe)
# Placeholders considered semi-sensitive but we keep a short hash to preserve uniqueness:
_HASHED_VALUE_PHS = {"ORDER_ID", "INVOICE_NUMBER"}
# Sensitive placeholders (hash them):
_SENSITIVE_PH_STRINGS = {"PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"}

# Amount bins for REFUND_AMOUNT
_AMOUNT_BINS = [(0, 10), (10, 50), (50, 100), (100, 500), (500, 1000), (1000, 1_000_000)]


def _amount_to_bin_label(num: float) -> str:
    for low, high in _AMOUNT_BINS:
        if low <= num < high:
            return f"{low}_{high}"
    return f"{_AMOUNT_BINS[-1][1]}_plus"


# ---------- Core function: normalize_placeholders ----------
def normalize_placeholders(text: str) -> Tuple[str, Dict[str, List[str]]]:
    """
    Replace recognized spans in `text` with placeholders and return (final_text, mappings).
    - Uses spaCy NER + Presidio Analyzer.
    - Preserves some non-sensitive entity values (e.g., account type),
      hashes sensitive values (person/email/phone) and short-hashes semi-sensitive IDs (order/invoice).
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

    # 4) Build final text by replacing spans with placeholders (with values or hashes) and collect mappings
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

            if ph_name in _WHITELIST_VALUE_PHS:
                # keep readable sanitized value for safe placeholders
                chosen_piece = f"[{ph_name}={safe_val}]"
            elif ph_name in _HASHED_VALUE_PHS:
                # use short hash for semi-sensitive ids (order/invoice)
                h = hashlib.md5(value_raw.encode("utf-8")).hexdigest()[:8]
                chosen_piece = f"[{ph_name}_HASH={h}]"
            elif ph_name in _SENSITIVE_PH_STRINGS:
                # hash sensitive PII
                h = hashlib.md5(value_raw.encode("utf-8")).hexdigest()[:8]
                chosen_piece = f"[{ph_name}_HASH={h}]"
            elif ph_name == "REFUND_AMOUNT" and amount is not None:
                bin_label = _amount_to_bin_label(amount)
                chosen_piece = f"[REFUND_AMT_BIN={bin_label}]"
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
