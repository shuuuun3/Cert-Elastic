# bench/run_llada_experiments.py
import time, argparse, json, os, torch
from typing import Dict, Any
from dlm.llada_fastdllm_adapter import LLaDAFastDLLMEngine
from dlm.elastic_cache import CertElasticRunner
from .datasets_gsm8k_math_mbpp_humaneval import (
    load_gsm8k, judge_gsm8k, load_math, load_mbpp, judge_mbpp, load_humaneval, judge_humaneval
)

TASKS = {
  "gsm8k":   (load_gsm8k,   judge_gsm8k),
  "math":    (load_math,    judge_gsm8k),   # 数値一致近似
  "mbpp":    (load_mbpp,    judge_mbpp),
  "humaneval": (load_humaneval, judge_humaneval),
}

def run_task(engine, task: str, n_items: int, gen_len: int, beta: int, gamma: float, conf_eps: float) -> Dict[str, Any]:
    loader, judge = TASKS[task]
    data = loader(n=n_items) if task in ("gsm8k","math","mbpp") else loader()
    data = data[:n_items]
    runner = CertElasticRunner(engine, beta=beta, gamma=gamma, conf_eps=conf_eps)

    total_tok = 0; total_sec = 0.0; correct = 0
    for ex in data:
        t0 = time.perf_counter()
        out = runner.run_generate(ex["prompt"], gen_len=gen_len, collect_traces=False)
        dt = time.perf_counter() - t0
        total_sec += dt
        # 生成トークン長を近似カウント（出力長）
        pred_text = out["text"]
        total_tok += len(pred_text.split())
        ok = judge(ex.get("answer",""), pred_text)
        correct += 1 if ok else 0
    tps = total_tok / total_sec if total_sec > 0 else 0.0
    acc = correct / max(1,len(data))
    return {"tokens_sec": tps, "accuracy": acc, "n": len(data), "gen_len": gen_len, "beta": beta, "gamma": gamma, "eps": conf_eps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llada_ckpt", required=True)
    ap.add_argument("--tasks", default="gsm8k,math,humaneval,mbpp")
    ap.add_argument("--n_items", type=int, default=200)
    ap.add_argument("--gen_len", type=int, default=512)
    ap.add_argument("--beta", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--eps", type=float, default=0.9)
    ap.add_argument("--out", default="runs/results.llada.certelastic.json")
    args = ap.parse_args()

    torch.cuda.empty_cache()
    engine = LLaDAFastDLLMEngine(args.llada_ckpt, device="cuda")
    engine.to(torch.device("cuda"))

    allres = {}
    for task in args.tasks.split(","):
        allres[task] = run_task(engine, task, args.n_items, args.gen_len, args.beta, args.gamma, args.eps)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(allres, f, ensure_ascii=False, indent=2)
    print(json.dumps(allres, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
