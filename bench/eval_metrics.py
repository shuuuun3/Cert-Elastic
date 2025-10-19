# eval_metrics.py
from difflib import SequenceMatcher
import re

def normalize_text(s: str):
    if s is None:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def exact_match(pred: str, ref: str):
    return 1 if normalize_text(pred) == normalize_text(ref) else 0

def seq_similarity(pred: str, ref: str):
    p = normalize_text(pred)
    r = normalize_text(ref)
    if len(p)==0 and len(r)==0:
        return 1.0
    return SequenceMatcher(None, p, r).ratio()

def aggregate_scores(results):
    # results: list of {"id","pred","ref"}
    n = 0
    em_total = 0
    sim_total = 0.0
    for r in results:
        pred = r.get("pred","")
        ref  = r.get("ref","")
        if ref is None or ref=="":
            continue
        n += 1
        em_total += exact_match(pred, ref)
        sim_total += seq_similarity(pred, ref)
    if n==0:
        return {"n_ref_items": 0}
    return {"n_ref_items": n, "exact_match": em_total / n, "avg_seq_similarity": sim_total / n}