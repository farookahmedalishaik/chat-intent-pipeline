# 11) run_all_sanity_checks.py --artifacts artifacts

#!/usr/bin/env python3
"""
run_all_sanity_checks.py

One-file runner that executes the must-do sanity checks described:
- overlap checks for raw text / text_model / tokenized input_ids / group_key
- placeholder-vs-label correlation check
- confusion matrix + top confusion pairs
- inspect misclassified examples for a target class (inquire_business_hours)
- confidence distribution / per-class mean confidence (correct vs incorrect)
- verify class_weights.npy vs label mapping
- preprocessing collisions (multiple raw -> same processed text_model)

Saves a few CSVs into ARTIFACTS_DIR (confusion_pairs_summary.csv,
placeholder_counts_by_label.csv, preprocessing_collisions_sample.csv)
and prints results to stdout.

Usage:
    python run_all_sanity_checks.py --artifacts artifacts --num_token_samples 5000
"""

import argparse
import os
import sys
import json
import math
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import re

try:
    import torch
except Exception:
    torch = None

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Run pipeline sanity checks")
parser.add_argument("--artifacts", "-a", default="artifacts", help="Path to artifacts directory")
parser.add_argument("--num_token_samples", "-n", type=int, default=5000, help="How many tokenized samples to inspect (per file)")
parser.add_argument("--min_confusion_thresh", "-m", type=int, default=5, help="Minimum count to report a confusion pair")
parser.add_argument("--inspect_class", "-c", default="inquire_business_hours", help="Class to inspect misclassifications for")
args = parser.parse_args()

ARTIFACTS_DIR = args.artifacts
N_TOKEN_SAMPLES = args.num_token_samples
MIN_CONF_THRESH = args.min_confusion_thresh
INSPECT_CLASS = args.inspect_class

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ---------- Helpers ----------
def safe_load_csv(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        print(f"[WARN] Failed to read CSV {path}: {e}")
        return None

def detect_text_columns(df):
    # prefer text_model, text_raw, text
    candidates = {}
    for col in df.columns:
        lc = col.lower()
        if "text_model" == lc or lc.endswith("text_model"):
            candidates['text_model'] = col
        if "text_raw" == lc or lc.endswith("text_raw"):
            candidates['text_raw'] = col
        if lc == "text" or lc.endswith("text"):
            candidates.setdefault('text', col)
    return candidates

def print_header(title):
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")

# ---------- 1) Overlap checks ----------
def check_overlaps():
    print_header("1) DATA OVERLAP CHECKS (raw text, text_model, group_key)")
    files = {
        "train": os.path.join(ARTIFACTS_DIR, "train_snapshot.csv"),
        "val": os.path.join(ARTIFACTS_DIR, "val_snapshot.csv"),
        "test": os.path.join(ARTIFACTS_DIR, "test_snapshot.csv"),
    }
    dfs = {}
    for k, p in files.items():
        df = safe_load_csv(p)
        if df is None:
            print(f"[WARN] {k} snapshot not found at {p}")
        else:
            print(f"{k}_snapshot rows: {len(df)}")
        dfs[k] = df

    # check textual overlaps
    def overlap_for_col(colkey):
        print(f"\n-- Overlap check for column: {colkey} --")
        present = {k: (dfs[k] is not None and colkey in dfs[k].columns) for k in dfs}
        if not any(present.values()):
            print(f"  No snapshots have column {colkey}.")
            return
        sets = {}
        for k in dfs:
            df = dfs[k]
            if df is None or colkey not in df.columns:
                sets[k] = set()
            else:
                sets[k] = set(df[colkey].astype(str).fillna("").tolist())
                print(f"  {k} unique {colkey}: {len(sets[k])}")
        # pairwise intersections
        pairs = [("train","val"), ("train","test"), ("val","test")]
        for a,b in pairs:
            inter = len(sets[a] & sets[b])
            print(f"  {a} ∩ {b}: {inter}")
    # try text_model, text_raw, text
    for col in ("text_model", "text_raw", "text"):
        overlap_for_col(col)

    # group_key overlap
    if dfs["train"] is not None and "group_key" in dfs["train"].columns:
        t = set(dfs["train"]["group_key"].astype(str).tolist())
        v = set(dfs["val"]["group_key"].astype(str).tolist()) if dfs["val"] is not None and "group_key" in dfs["val"].columns else set()
        te = set(dfs["test"]["group_key"].astype(str).tolist()) if dfs["test"] is not None and "group_key" in dfs["test"].columns else set()
        print("\n-- group_key overlap --")
        print("train unique group_key:", len(t))
        print("val unique group_key:", len(v))
        print("test unique group_key:", len(te))
        print("train∩val:", len(t & v))
        print("train∩test:", len(t & te))
        print("val∩test:", len(v & te))
    else:
        print("\n[group_key] not present in train_snapshot.csv (or other snapshots)")

# ---------- 2) Tokenized input_ids overlap ----------
def check_tokenized_overlap():
    print_header("2) TOKENIZED INPUT_IDS OVERLAP (byte-level)")

    if torch is None:
        print("[WARN] torch not available; skipping tokenized overlap checks.")
        return

    pt_files = {
        "train": os.path.join(ARTIFACTS_DIR, "train_data.pt"),
        "val": os.path.join(ARTIFACTS_DIR, "val_data.pt"),
        "test": os.path.join(ARTIFACTS_DIR, "test_data.pt"),
    }
    sets = {}
    for k, p in pt_files.items():
        if not os.path.exists(p):
            print(f"[WARN] {p} not found.")
            sets[k] = set()
            continue
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            ids = d.get("input_ids") or d.get("input_ids_tensor") or d.get("input_ids_list") or None
            if ids is None:
                print(f"[WARN] Couldn't find 'input_ids' in {p}; keys present: {list(d.keys())}")
                sets[k] = set()
                continue
            out = set()
            N = min(len(ids), N_TOKEN_SAMPLES)
            for i in range(N):
                row = ids[i]
                # support both numpy/tensor and lists
                try:
                    arr = row.numpy() if hasattr(row, "numpy") else np.array(row)
                    out.add(arr.tobytes())
                except Exception:
                    # fallback to string
                    out.add(str(row))
            sets[k] = out
            print(f"{k} input_ids samples: {len(out)}")
        except Exception as e:
            print(f"[WARN] failed to load {p}: {e}")
            sets[k] = set()

    pairs = [("train","val"), ("train","test"), ("val","test")]
    for a,b in pairs:
        inter = len(sets[a] & sets[b])
        print(f"  {a} ∩ {b} (tokenized bytes): {inter}")

# ---------- 3) Placeholder vs label correlation ----------
def placeholders_vs_labels():
    print_header("3) PLACEHOLDER -> LABEL CORRELATION (on validation set)")
    # Try to locate a validation predictions file first
    val_preds_path = os.path.join(ARTIFACTS_DIR, "val_predictions_with_probs.csv")
    val_snapshot_path = os.path.join(ARTIFACTS_DIR, "val_snapshot.csv")

    if os.path.exists(val_preds_path):
        df = safe_load_csv(val_preds_path)
        # try columns: 'text' or 'text' presence
        if df is None:
            print("[WARN] Could not read val_predictions_with_probs.csv")
            return None
        text_col = None
        for c in ("text", "texts", "text_raw", "text_model"):
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            # maybe 'text' wasn't included; try 'true' or else abort
            print("[WARN] val_predictions_with_probs.csv has no text column; attempting to use 'true'/'pred' only.")
            return None
        # label column: 'true' or 'true_id' or 'true_label'
        label_col = None
        for c in ("true","true_id","true_label","label"):
            if c in df.columns:
                label_col = c
                break
        if label_col is None:
            print("[WARN] val_predictions_with_probs.csv has no true label column.")
        # extract placeholders regex
        ph_re = re.compile(r"\[([A-Z0-9_=-]+)\]", flags=re.IGNORECASE)
        counts = defaultdict(Counter)
        for _, row in df.iterrows():
            txt = str(row.get(text_col,""))
            lbl = str(row.get(label_col,""))
            for m in ph_re.findall(txt):
                counts[lbl][m] += 1
        # convert to DataFrame for saving
        out = []
        labels = sorted(counts.keys())
        ph_keys = sorted({k for lbl in counts for k in counts[lbl].keys()})
        for lbl in labels:
            row = {"label": lbl, "support": int((df[label_col]==lbl).sum()) if label_col in df.columns else ""}
            for ph in ph_keys:
                row[ph] = int(counts[lbl].get(ph, 0))
            out.append(row)
        out_df = pd.DataFrame(out).fillna(0)
        out_csv = os.path.join(ARTIFACTS_DIR, "placeholder_counts_by_label.csv")
        out_df.to_csv(out_csv, index=False)
        print(f"Saved placeholder counts per label to {out_csv}")
        # Print top placeholder counts for suspicious ones
        print("\nTop placeholders for labels (sample):")
        for lbl in labels:
            top = counts[lbl].most_common(8)
            if top:
                print(f"  {lbl} top placeholders: {top[:5]}")
        return out_df
    elif os.path.exists(val_snapshot_path):
        df = safe_load_csv(val_snapshot_path)
        if df is None:
            print("[WARN] Could not read val_snapshot.csv")
            return None
        # find likely text column
        cols = detect_text_columns(df)
        text_col = cols.get("text_model") or cols.get("text") or cols.get("text_raw")
        label_col = "label" if "label" in df.columns else None
        if text_col is None or label_col is None:
            print("[WARN] val_snapshot.csv missing text or label column(s).")
            return None
        ph_re = re.compile(r"\[([A-Z0-9_=-]+)\]", flags=re.IGNORECASE)
        counts = defaultdict(Counter)
        for _, row in df.iterrows():
            txt = str(row[text_col])
            lbl = str(row[label_col])
            for m in ph_re.findall(txt):
                counts[lbl][m] += 1
        out = []
        labels = sorted(counts.keys())
        ph_keys = sorted({k for lbl in counts for k in counts[lbl].keys()})
        for lbl in labels:
            row = {"label": lbl, "support": int((df[label_col]==lbl).sum())}
            for ph in ph_keys:
                row[ph] = int(counts[lbl].get(ph, 0))
            out.append(row)
        out_df = pd.DataFrame(out).fillna(0)
        out_csv = os.path.join(ARTIFACTS_DIR, "placeholder_counts_by_label.csv")
        out_df.to_csv(out_csv, index=False)
        print(f"Saved placeholder counts per label to {out_csv}")
        for lbl in labels:
            top = counts[lbl].most_common(6)
            if top:
                print(f"  {lbl}: {top[:4]}")
        return out_df
    else:
        print("[WARN] No val_predictions_with_probs.csv or val_snapshot.csv found — cannot compute placeholder counts.")
        return None

# ---------- 4) Confusion matrix and top confusion pairs ----------
def confusion_and_top_pairs():
    print_header("4) CONFUSION MATRIX & TOP CONFUSION PAIRS")
    preds_path = os.path.join(ARTIFACTS_DIR, "val_predictions_with_probs.csv")
    if not os.path.exists(preds_path):
        print(f"[WARN] {preds_path} not found. Please run evaluation/predictions to generate it first.")
        return None
    df = safe_load_csv(preds_path)
    if df is None:
        return None
    # detect required columns
    col_true = None
    for c in ("true","true_id","label","true_label"):
        if c in df.columns:
            col_true = c
            break
    col_pred = None
    for c in ("pred","pred_id","pred_label","prediction"):
        if c in df.columns:
            col_pred = c
            break
    if col_true is None or col_pred is None:
        print("[WARN] val_predictions_with_probs.csv lacks 'true'/'pred' columns.")
        return None
    # ensure both are strings
    df[col_true] = df[col_true].astype(str)
    df[col_pred] = df[col_pred].astype(str)
    pairs = Counter()
    for t,p in zip(df[col_true].tolist(), df[col_pred].tolist()):
        pairs[(t,p)] += 1
    pairs_list = sorted(pairs.items(), key=lambda x: -x[1])
    rows = []
    print("\nTop confusion pairs (true -> pred):")
    for (t,p),cnt in pairs_list[:50]:
        rows.append({"true": t, "pred": p, "count": cnt})
        print(f"  {t:40s} -> {p:40s} : {cnt}")
    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(ARTIFACTS_DIR, "confusion_pairs_summary.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved confusion pairs summary to {out_csv}")
    # print those above threshold
    print(f"\nPairs with count >= {MIN_CONF_THRESH}:")
    for (t,p), cnt in pairs.items():
        if cnt >= MIN_CONF_THRESH:
            print(f"  {t} -> {p} : {cnt}")
    return df

# ---------- 5) Show misclassified examples for a target class ----------
def show_misclassified_examples(df_preds):
    print_header(f"5) MISCLASSIFIED EXAMPLES FOR CLASS: {INSPECT_CLASS}")
    if df_preds is None:
        print("[WARN] No predictions DataFrame available.")
        return
    # detect columns
    text_col = None
    for c in ("text","texts","text_model","text_raw"):
        if c in df_preds.columns:
            text_col = c
            break
    col_true = None
    for c in ("true","true_id","label","true_label"):
        if c in df_preds.columns:
            col_true = c
            break
    col_pred = None
    for c in ("pred","pred_id","pred_label","prediction"):
        if c in df_preds.columns:
            col_pred = c
            break
    if col_true is None or col_pred is None:
        print("[WARN] predictions CSV missing true/pred columns.")
        return
    # where true == INSPECT_CLASS but pred !=
    true_to_pred = df_preds[(df_preds[col_true]==INSPECT_CLASS) & (df_preds[col_pred] != INSPECT_CLASS)]
    pred_to_true = df_preds[(df_preds[col_pred]==INSPECT_CLASS) & (df_preds[col_true] != INSPECT_CLASS)]
    print(f"Examples where true=={INSPECT_CLASS} but pred!= ({len(true_to_pred)} rows). Showing up to 20:")
    for _, r in true_to_pred.head(20).iterrows():
        txt = r.get(text_col, "(no text)") if text_col else "(no text)"
        print("----")
        print("text:", txt)
        print("true:", r[col_true], "pred:", r[col_pred])
        if "probs" in r and r["probs"]:
            print("probs:", r["probs"])
    print(f"\nExamples where pred=={INSPECT_CLASS} but true!= ({len(pred_to_true)} rows). Showing up to 20:")
    for _, r in pred_to_true.head(20).iterrows():
        txt = r.get(text_col, "(no text)") if text_col else "(no text)"
        print("----")
        print("text:", txt)
        print("true:", r[col_true], "pred:", r[col_pred])
        if "probs" in r and r["probs"]:
            print("probs:", r["probs"])

# ---------- 6) Confidence distribution / calibration stats ----------
def confidence_stats(df_preds):
    print_header("6) CONFIDENCE DISTRIBUTION / PER-CLASS CONFIDENCE STATS")
    if df_preds is None:
        print("[WARN] No predictions DataFrame available.")
        return
    # Expect a 'probs' column saved as list-string or actual list
    if "probs" not in df_preds.columns:
        print("[WARN] predictions file has no 'probs' column; cannot compute confidence stats.")
        return
    # parse probs entries into numpy arrays if needed
    probs_list = []
    for v in df_preds["probs"].tolist():
        if isinstance(v, str):
            try:
                arr = json.loads(v.replace("'", '"'))  # try to be tolerant
                probs_list.append(np.array(arr, dtype=float))
            except Exception:
                probs_list.append(None)
        elif isinstance(v, (list, tuple, np.ndarray)):
            probs_list.append(np.array(v, dtype=float))
        else:
            probs_list.append(None)
    df_preds = df_preds.copy()
    df_preds["_probs_arr"] = probs_list
    # compute predicted class prob
    pred_probs = []
    for row in df_preds["_probs_arr"]:
        if row is None:
            pred_probs.append(None)
        else:
            pred_probs.append(float(np.max(row)))
    df_preds["_pred_prob"] = pred_probs
    # detect true/pred columns
    col_true = next((c for c in ("true","true_id","label","true_label") if c in df_preds.columns), None)
    col_pred = next((c for c in ("pred","pred_id","pred_label","prediction") if c in df_preds.columns), None)
    if col_true is None or col_pred is None:
        print("[WARN] true/pred columns missing; cannot compute per-class stats.")
        return
    classes = sorted(set(df_preds[col_true].astype(str).tolist() + df_preds[col_pred].astype(str).tolist()))
    summary_rows = []
    for cls in classes:
        cls_rows = df_preds[(df_preds[col_true]==cls)]
        pred_rows = df_preds[(df_preds[col_pred]==cls)]
        # mean prob when model predicted cls (for all rows where pred==cls)
        mean_pred_prob = float(np.nanmean([p for p in pred_rows["_pred_prob"].tolist() if p is not None])) if len(pred_rows)>0 else float('nan')
        # mean prob assigned to correct cls when true==cls
        mean_true_prob = float(np.nanmean([float(row["_probs_arr"][classes.index(cls)]) if (row["_probs_arr"] is not None and len(row["_probs_arr"])>classes.index(cls)) else np.nan for _, row in cls_rows.iterrows()])) if len(cls_rows)>0 else float('nan')
        # average prob on misclassified rows where model predicted cls but true!=cls
        wrong_pred_rows = pred_rows[pred_rows[col_true] != pred_rows[col_pred]]
        mean_prob_wrong = float(np.nanmean([p for p in wrong_pred_rows["_pred_prob"].tolist() if p is not None])) if len(wrong_pred_rows)>0 else float('nan')
        summary_rows.append({
            "class": cls,
            "support_true": len(cls_rows),
            "support_pred": len(pred_rows),
            "mean_prob_when_predicted": mean_pred_prob,
            "mean_prob_on_true_rows": mean_true_prob,
            "mean_prob_on_wrong_predictions": mean_prob_wrong,
        })
    sum_df = pd.DataFrame(summary_rows).sort_values("support_true", ascending=False)
    print(sum_df.head(40).to_string(index=False))
    out_csv = os.path.join(ARTIFACTS_DIR, "per_class_confidence_stats.csv")
    sum_df.to_csv(out_csv, index=False)
    print(f"\nSaved per-class confidence stats to {out_csv}")

# ---------- 7) Verify class weights file vs label mapping ----------
def verify_class_weights():
    print_header("7) VERIFY CLASS_WEIGHTS (class_weights.npy) vs label mapping")
    cw_path = os.path.join(ARTIFACTS_DIR, "class_weights.npy")
    label_map_path = os.path.join(ARTIFACTS_DIR, "label_mapping.csv")
    exists_cw = os.path.exists(cw_path)
    exists_lm = os.path.exists(label_map_path)
    if not exists_cw:
        print("[WARN] class_weights.npy not found at", cw_path)
    else:
        try:
            cw = np.load(cw_path)
            print("Class weights shape:", cw.shape)
        except Exception as e:
            print("[WARN] Failed to load class_weights.npy:", e)
            cw = None
    if not exists_lm:
        print("[WARN] label_mapping.csv not found at", label_map_path)
        labels = None
    else:
        try:
            lm = pd.read_csv(label_map_path, dtype=str).fillna("")
            if "label" in lm.columns:
                labels = lm["label"].tolist()
                print("Label mapping classes:", labels)
            else:
                print("[WARN] label_mapping.csv missing 'label' column. keys:", lm.columns.tolist())
                labels = None
        except Exception as e:
            print("[WARN] Failed to read label_mapping.csv:", e)
            labels = None
    if cw is not None and labels is not None:
        if len(cw) != len(labels):
            print(f"[ERROR] class_weights length ({len(cw)}) != number of labels ({len(labels)}).")
        else:
            print("Class weights appear to match number of labels. Mapping label -> weight:")
            for lab, w in zip(labels, cw.tolist()):
                print(f"  {lab}: {w:.4f}")
    return

# ---------- 8) Preprocessing collisions (raw -> processed) ----------
def preprocessing_collisions():
    print_header("8) PREPROCESSING COLLISIONS (distinct raw -> same text_model)")
    # look for snapshot files that likely contain raw -> text_model mappings
    candidates = []
    for fname in ("train_snapshot.csv", "val_snapshot.csv", "test_snapshot.csv"):
        p = os.path.join(ARTIFACTS_DIR, fname)
        if os.path.exists(p):
            candidates.append(p)
    if not candidates:
        print("[WARN] No train/val/test snapshots found for preprocessing collisions check.")
        return
    for p in candidates:
        df = safe_load_csv(p)
        if df is None:
            continue
        cols = detect_text_columns(df)
        text_raw_col = cols.get("text_raw") or cols.get("text")
        text_model_col = cols.get("text_model")
        if text_model_col is None:
            print(f"  {os.path.basename(p)} has no 'text_model' column; skipping.")
            continue
        if text_raw_col is None:
            print(f"  {os.path.basename(p)} has no 'text_raw' or 'text' column; skipping.")
            continue
        # group by text_model and count distinct raw entries
        grouped = df.groupby(text_model_col)[text_raw_col].nunique().reset_index(name="n_unique_raw")
        collisions = grouped[grouped["n_unique_raw"] > 1].sort_values("n_unique_raw", ascending=False)
        print(f"  {os.path.basename(p)}: {len(collisions)} text_model values map from >1 raw inputs.")
        # If any found, show examples (join back)
        if len(collisions) > 0:
            sample = collisions.head(200)
            rows = []
            for idx, row in sample.iterrows():
                tm = row[text_model_col]
                raws = df[df[text_model_col]==tm][text_raw_col].unique().tolist()
                rows.append({"text_model": tm, "n_unique_raw": row["n_unique_raw"], "examples_raw": raws[:10]})
            out_df = pd.DataFrame(rows)
            out_csv = os.path.join(ARTIFACTS_DIR, f"preprocessing_collisions_{os.path.basename(p)}.csv")
            out_df.to_csv(out_csv, index=False)
            print(f"  Saved collisions sample to {out_csv}")

# ---------- Run everything ----------
def run_all():
    check_overlaps()
    check_tokenized_overlap()
    placeholder_df = placeholders_vs_labels()
    preds_df = confusion_and_top_pairs()
    # If confusion_and_top_pairs returned df, pass it to next steps
    show_misclassified_examples(preds_df)
    confidence_stats(preds_df)
    verify_class_weights()
    preprocessing_collisions()
    print_header("SUMMARY")
    print("All checks finished. Results & CSV artifacts (if generated) are in:", os.path.abspath(ARTIFACTS_DIR))
    print("Important artifacts produced (if available):")
    for f in ("placeholder_counts_by_label.csv","confusion_pairs_summary.csv","per_class_confidence_stats.csv"):
        p = os.path.join(ARTIFACTS_DIR,f)
        if os.path.exists(p):
            print("  -", f)
    print("\nIf any warnings/errors were printed above, inspect the corresponding CSVs and samples to find the root cause.")
    print("Common next steps if you see leakage: (1) inspect placeholder_counts_by_label.csv (2) inspect preprocessing_collisions CSVs (3) examine raw -> model text transforms)")

if __name__ == "__main__":
    run_all()
