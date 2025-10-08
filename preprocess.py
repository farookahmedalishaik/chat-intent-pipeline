# preprocess.py
"""
Model-based preprocessing using spaCy + Microsoft Presidio.
- Detects PERSON, GPE/LOC, MONEY, DATE, EMAIL, PHONE, ORG, etc.
- Replaces detected spans with uppercase placeholders like [PERSON_NAME]
- Returns (final_text, mappings) where mappings is a dict: placeholder -> [values]
"""

import re
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
    s = re.sub(r"http\S+", "", s) # remove URLs (simple)
    s = re.sub(r"[^\x00-\x7F]", " ", s)  # replace common “smart” quotes / non-ascii with spaces (keeps tokens simple)
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

        # Optional: add a small EntityRuler to catch common order/invoice ID formats.These are spaCy patterns (optional).
        if use_optional_order_invoice_patterns:
            # 1. Add the component by its string name "entity_ruler".
            # This returns the ruler instance so you can configure it.
            ruler = _nlp.add_pipe("entity_ruler", before="ner")

            # 2. Define and add your patterns to the ruler instance you just got.

            # The entire patterns list has been updated to improve accuracy and reduce false positives.
            patterns = [
                # --- ORDER ID PATTERNS  ---
                # Catches: ORD-123, order 123, purchase123, etc.
                {"label": "ORDER_ID", "pattern": [{"LOWER": {"IN": ["ord", "ord-", "orderid", "order-id", "ord#"]}}, {"IS_PUNCT": True, "OP": "?"}, {"SHAPE": "dddd"}]},
                {"label": "ORDER_ID", "pattern": [{"LOWER": {"IN": ["order", "purchase"]}}, {"IS_SPACE": True, "OP": "?"}, {"SHAPE": "dddd"}]},
                # Catches standalone numbers that are between 6 and 12 digits long
                {"label": "ORDER_ID", "pattern": [{"SHAPE": "dddddd"}]},
                {"label": "ORDER_ID", "pattern": [{"SHAPE": "dddddddd"}]},
                {"label": "ORDER_ID", "pattern": [{"SHAPE": "dddddddddd"}]},
                {"label": "ORDER_ID", "pattern": [{"SHAPE": "dddddddddddd"}]},

                # --- INVOICE NUMBER PATTERNS ---
                # Catches: INV-123, invoice 123, bill #123, etc.
                {"label": "INVOICE_NUMBER", "pattern": [{"LOWER": {"IN": ["inv", "inv-", "invoice#"]}}, {"IS_PUNCT": True, "OP": "?"}, {"SHAPE": "dddd"}]},
                {"label": "INVOICE_NUMBER", "pattern": [{"LOWER": {"IN": ["bill", "invoice"]}}, {"TEXT": "#", "OP": "?"}, {"IS_DIGIT": True}]},

                # --- ACCOUNT TYPE PATTERNS ---
                # Catches: pro account, standard account, etc.
                {"label": "ACCOUNT_TYPE", "pattern": [{"LOWER": {"IN": ["pro", "standard", "freemium", "platinum", "gold"]}}, {"LOWER": "account"}]},
                
                # --- LOCATION FIX PATTERNS  ---
                # Prevents common locations from being misidentified as PERSON_NAME
                {"label": "GPE", "pattern": [{"LOWER": "kent"}]},
                {"label": "GPE", "pattern": [{"LOWER": "streetsboro"}]},

                # --- KEYWORD FIX PATTERNS  ---
                # Prevents common words from being misidentified as ORGANIZATION
                {"label": "STOP_ENTITY", "pattern": [{"LOWER": "eta"}]},
                {"label": "STOP_ENTITY", "pattern": [{"LOWER": "modify"}]},
            ]
            ruler.add_patterns(patterns)

    if _analyzer is None:
        # Presidio AnalyzerEngine (uses built-in recognizers + spaCy if available)
        _analyzer = AnalyzerEngine()
    return _nlp, _analyzer

# ---------- Mapping from spaCy/Presidio entity types to our placeholders ----------
# Added US_DRIVER_LICENSE and our custom STOP_ENTITY ---
_ENTITY_TO_PLACEHOLDER = {
    # spaCy types (common)
    "PERSON": "[PERSON_NAME]",
    "GPE": "[DELIVERY_LOCATION]",
    "LOC": "[DELIVERY_LOCATION]",
    "MONEY": "[REFUND_AMOUNT]",
    "DATE": "[DATE]",
    "ORG": "[ORGANIZATION]",
    "PRODUCT": "[PRODUCT]",
    "ACCOUNT_TYPE": "[ACCOUNT_TYPE]",

    # Presidio / other types (Presidio returns types like 'PHONE_NUMBER', 'EMAIL_ADDRESS', etc.)
    "PHONE_NUMBER": "[PHONE_NUMBER]",
    "EMAIL_ADDRESS": "[EMAIL_ADDRESS]",
    "CREDIT_CARD": "[CREDIT_CARD]",
    "IBAN_CODE": "[IBAN]",
    "US_BANK_NUMBER": "[BANK_NUMBER]",
    "US_DRIVER_LICENSE": "[US_DRIVER_LICENSE]", # ADDED
    
    # Custom labels  added in the EntityRuler
    "ORDER_ID": "[ORDER_ID]",
    "INVOICE_NUMBER": "[INVOICE_NUMBER]",

    # Custom label to ignore certain words (ADDED)
    "STOP_ENTITY": "[STOP_ENTITY]",
}

# ---------- Core function: normalize_placeholders ----------
def normalize_placeholders(text: str) -> Tuple[str, Dict[str, List[str]]]:
    """
    Replace recognized spans in `text` with uppercase placeholders and return (final_text, mappings).
    - Uses spaCy NER + Presidio Analyzer.
    - Keeps placeholders UPPERCASE so mapping keys match placeholders.
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
            ent_type = r.entity_type  # e.g. "PERSON" or "PHONE_NUMBER"
            ph = _ENTITY_TO_PLACEHOLDER.get(ent_type, f"[{ent_type}]")
            spans.append({"start": start, "end": end, "ph": ph, "value": raw[start:end], "score": getattr(r, "score", None)})
    except Exception:
        # If presidio fails for any reason, we continue with spaCy only (fail safe)
        presidio_results = []

    # 2) spaCy NER (complements Presidio; also picks up entities Presidio misses)
    doc = nlp(raw)
    for ent in doc.ents:
        ent_type = ent.label_  # e.g. "PERSON", "GPE", or our ruler labels like "ORDER_ID"
        ph = _ENTITY_TO_PLACEHOLDER.get(ent_type)
        if ph is None:
            # ignore entities we don't care about
            continue
        spans.append({"start": ent.start_char, "end": ent.end_char, "ph": ph, "value": ent.text, "score": None})

    # if no spans found, return normalized lowercased text and empty mappings
    if not spans:
        cleaned = re.sub(r"\s+", " ", raw).strip().lower()
        return cleaned, {}


    # 3) Filter and merge spans
    # First, remove any spans that we've marked as a STOP_ENTITY
    # Also, filter out weak phone number or driver license matches from Presidio on short numbers
    filtered_spans = []
    for s in spans:
        if s['ph'] == '[STOP_ENTITY]':
            continue # Ignore this span completely
        
        # Rule: If Presidio identifies a short number (less than 7 digits) as a phone number or driver's license, it's likely wrong. Ignore it.
        if s['ph'] in ('[PHONE_NUMBER]', '[US_DRIVER_LICENSE]') and len(s['value'].strip()) < 7:
            continue
            
        filtered_spans.append(s)

    # Now merge the remaining valid spans, handling overlaps
    spans_sorted = sorted(filtered_spans, key=lambda x: (x["start"], -(x["score"] or 0)))
    merged = []
    last_end = -1
    for s in spans_sorted:
        # If the current span does not overlap with the last one added, add it.
        if s["start"] >= last_end:
            merged.append(s)
            last_end = s["end"]
        # If it does overlap, the one we already added (merged[-1]) has priority
        # because of the initial sort (by start position and then score).
        # So, we simply do nothing and discard the current overlapping span `s`.
    # --- END OF MODIFICATIONS ---

    # 4) Build final text by replacing spans with placeholders (uppercase) and collect mappings
    mappings = defaultdict(list)
    out_pieces = []
    last_idx = 0
    for s in merged:
        # Add the text before the placeholder, if any
        pre_text = raw[last_idx:s["start"]].lower().strip()
        if pre_text:
            out_pieces.append(pre_text)
        
        # Add the placeholder
        out_pieces.append(s['ph'])
        mappings[s["ph"]].append(s["value"])
        last_idx = s["end"]
    
    # Add the final piece of text
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
    example = "Can you check order ORD-12345 for John Doe delivered to New York? Refund $10.99. Invoice INV-98765 also. Call me at +1 555-234-9999 or email alice@example.com."
    final_text, mappings = normalize_placeholders(example)
    print("FINAL TEXT:")
    print(final_text)
    print("\nMAPPINGS:")
    for k, v in mappings.items():
        print(k, "->", v)