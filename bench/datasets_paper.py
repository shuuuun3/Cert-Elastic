# bench/datasets_paper.py
from datasets import load_dataset
import re, json

def normalize_number(s: str) -> str:
    if not s: return ""
    m = re.search(r"(-?\d+(?:\.\d+)?)", s.replace(",", ""))
    return m.group(1) if m else ""

# ---- GSM8K
def load_gsm8k(split="test", n=None):
    ds = load_dataset("gsm8k", "main")[split]
    items = [{"id": f"gsm8k_{i}", "prompt": ex["question"] + " A:", "answer": ex["answer"]} for i, ex in enumerate(ds)]
    return items[:n] if n else items

def judge_gsm8k(ref: str, pred: str) -> bool:
    return normalize_number(ref) == normalize_number(pred)

# ---- MATH (Hendrycks)
def load_math(split="test", n=None):
    ds = load_dataset("hendrycks/competition_math")[split]
    items = [{"id": f"math_{i}", "prompt": ex["problem"], "answer": ex["solution"]} for i, ex in enumerate(ds)]
    return items[:n] if n else items

def judge_math(ref: str, pred: str) -> bool:
    return normalize_number(ref) == normalize_number(pred)

# ---- HumanEval（簡易。厳密pass@1が必要なら lm-eval を別途使用）
def load_humaneval(n=None):
    ds = load_dataset("openai_humaneval")["test"]
    items = [{"id": ex["task_id"], "prompt": ex["prompt"], "answer": ex["canonical_solution"]} for ex in ds]
    return items[:n] if n else items

def judge_humaneval(ref: str, pred: str) -> bool:
    return "def " in pred  # 近似

# ---- MBPP（簡易）
def load_mbpp(n=None):
    ds = load_dataset("mbpp", "sanitized")["test"]
    items = [{"id": f"mbpp_{i}", "prompt": ex["text"], "answer": ex["code"]} for i, ex in enumerate(ds)]
    return items[:n] if n else items

def judge_mbpp(ref: str, pred: str) -> bool:
    return "def " in pred  # 近似
