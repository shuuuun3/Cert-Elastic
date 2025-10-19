# bench/datasets_gsm8k_math_mbpp_humaneval.py
from datasets import load_dataset
import re, json, random

def load_gsm8k(split="test", n=None):
    ds = load_dataset("gsm8k","main")[split]
    items = [{"id": f"gsm8k_{i}", "prompt": ex["question"]+" A:", "answer": ex["answer"]} for i,ex in enumerate(ds)]
    return items[:n] if n else items

def normalize_number(s: str) -> str:
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    return m.group(1) if m else ""

def judge_gsm8k(ref: str, pred: str) -> bool:
    return normalize_number(ref) == normalize_number(pred)

def load_mbpp(n=None):
    ds = load_dataset("mbpp","sanitized")["test"]
    items = [{"id": f"mbpp_{i}", "prompt": ex["text"], "answer": ex["code"]} for i, ex in enumerate(ds)]
    return items[:n] if n else items

def judge_mbpp(pred: str) -> bool:
    # 簡易pass@1近似（雰囲気一致）。本番はharness推奨。
    return "def " in pred

def load_humaneval():
    ds = load_dataset("openai_humaneval")["test"]
    items = [{"id": ex["task_id"], "prompt": ex["prompt"], "answer": ex["canonical_solution"]} for ex in ds]
    return items

def judge_humaneval(pred: str) -> bool:
    return "def " in pred

def load_math(split="test", n=None):
    ds = load_dataset("hendrycks/competition_math")[split]
    items = [{"id": f"math_{i}", "prompt": ex["problem"], "answer": ex["solution"]} for i,ex in enumerate(ds)]
    return items[:n] if n else items
