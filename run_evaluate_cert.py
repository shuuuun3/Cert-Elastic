#!/usr/bin/env python3
"""
Run full Cert-Elastic evaluation pipeline and emit summary.json.

Usage (example):
python run_evaluate_cert.py \
  --model_id mistralai/Mistral-7B-Instruct-v0.3 \
  --dtype float16 --load_in_4bit 0 --attn_impl eager \
  --max_new_tokens 128 --eval_prompts_n 20 --epsilon 0.02 \
  --alpha 0.5 --beta 2.0 --out_dir runs \
  --run_fast 1

Notes:
- This imports your existing cert_elastic modules (main logging function).
- If you set --run_fast 1 the script will call run_cert_fast.py as a subprocess
  to gather an empirical fast-run speed (it may reload the model).
"""
import argparse, json, subprocess, sys, time, os
from pathlib import Path
import numpy as np
import pandas as pd

# import project modules
from cert_elastic.loader import load_model_tokenizer
from cert_elastic.cert_core import decode_with_cert_logging
from cert_elastic.utils import make_run_dir, dump_json, cuda_clean

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3")
    p.add_argument("--dtype", default="float16", choices=["bfloat16","float16"])
    p.add_argument("--load_in_4bit", type=int, default=0)
    p.add_argument("--device_map", default="auto")
    p.add_argument("--attn_impl", default="eager")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=0.02)
    p.add_argument("--topk", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--eval_prompts_n", type=int, default=20)
    p.add_argument("--out_dir", default="./runs")
    p.add_argument("--run_fast", type=int, default=0, help="If 1 call run_cert_fast.py to measure empirical speed")
    return p.parse_args()

def aggregate_from_log(log, epsilon):
    # flatten entries
    rows = []
    total_tokens = 0
    total_time = 0.0
    for r in log["results"]:
        total_tokens += r.get("tokens",0)
        total_time += r.get("time_sec",0.0)
        for step_idx, layers in enumerate(r["attn_logs"]):
            for d in layers:
                rows.append({
                    "prompt_id": r["prompt_id"],
                    "step": step_idx,
                    "layer": d["layer"],
                    "f": d["f"],
                    "gamma": d["gamma"],
                    "delta": d["delta"],
                    "c_eff": d["c_eff"],
                    "safe": 1 if d["f"] <= epsilon else 0
                })
    if len(rows) == 0:
        return {}
    df = pd.DataFrame(rows)
    total_checks = len(df)
    safe_count = int(df["safe"].sum())
    safe_ratio = float(safe_count) / total_checks
    per_layer = df.groupby("layer")["safe"].mean().to_dict()
    per_step = df.groupby("step")["safe"].mean().to_dict()
    metrics = {
        "total_prompts": len(log["results"]),
        "total_tokens": int(total_tokens),
        "total_time_sec": float(total_time),
        "avg_tok_per_sec": float(np.mean([r.get("tok_per_sec",0.0) for r in log["results"]])),
        "total_checks": int(total_checks),
        "safe_count": int(safe_count),
        "safe_ratio": float(safe_ratio),
        "per_layer_safe_ratio": {int(k): float(v) for k,v in per_layer.items()},
        "per_step_safe_ratio": {int(k): float(v) for k,v in per_step.items()}
    }
    # predicted op reduction and naive speedup estimate (optimistic)
    pred_reduction = safe_ratio
    if pred_reduction >= 0.999:
        pred_speedup = 100.0
    else:
        pred_speedup = 1.0 / max(1e-6, (1.0 - pred_reduction))
    metrics.update({
        "predicted_op_reduction_fraction": float(pred_reduction),
        "predicted_speedup_estimate_naive": float(pred_speedup)
    })
    return metrics, df

def main():
    args = parse_args()
    run_dir = make_run_dir(args.out_dir)
    cfg = vars(args)
    dump_json(cfg, Path(run_dir/"config.eval.json"))

    # 1) load model/tokenizer
    print("[load] model:", args.model_id)
    model, tokenizer = load_model_tokenizer(args.model_id, args.dtype, args.device_map, args.attn_impl, bool(args.load_in_4bit))

    # 2) perform cert logging decoding (this produces full attn logs)
    prompts = None
    try:
        from cert_elastic.prompts import make_demo_prompts
        prompts = make_demo_prompts(args.eval_prompts_n)
    except Exception:
        prompts = ["Q: 2+2=? A:"] * args.eval_prompts_n

    print("[run] decode_with_cert_logging ...")
    start = time.time()
    log = decode_with_cert_logging(
        model=model, tokenizer=tokenizer, prompts=prompts,
        epsilon=args.epsilon, topk=args.topk, alpha=args.alpha, beta=args.beta,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k
    )
    end = time.time()
    print(f"[done] logging run in {end-start:.2f}s")
    dump_json(log, Path(run_dir/"results.logging.json"))

    # 3) aggregate metrics from log
    agg = {}
    aggres = aggregate_from_log(log, args.epsilon)
    if aggres:
        metrics, df = aggres
        agg.update(metrics)
        # save csvs for reproducibility
        df.to_csv(Path(run_dir/"cert_checks.csv"), index=False)
    else:
        agg["error"] = "no attn logs collected"

    # 4) optional: run fast mode (run_cert_fast.py) to measure empirical speed
    fast_info = {}
    if args.run_fast:
        cmd = [
            sys.executable, str(Path(__file__).parent / "run_cert_fast.py"),
            "--model_id", args.model_id,
            "--dtype", args.dtype,
            "--load_in_4bit", str(int(args.load_in_4bit)),
            "--device_map", args.device_map,
            "--attn_impl", args.attn_impl,
            "--max_new_tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
            "--top_k", str(args.top_k),
            "--epsilon", str(args.epsilon),
            "--alpha", str(args.alpha),
            "--beta", str(args.beta),
            "--eval_prompts_n", str(args.eval_prompts_n),
            "--out_dir", str(run_dir)
        ]
        print("[fast] running run_cert_fast.py ... (this reloads the model, may take time)")
        subprocess.run(cmd, check=False)
        # read outputs if present
        fast_json = Path(run_dir/"results.fast.json")
        speed_txt = Path(run_dir/"speed.fast.txt")
        if fast_json.exists():
            try:
                with open(fast_json, "r", encoding="utf-8") as f:
                    fast_info["results_fast"] = json.load(f)
            except Exception as e:
                fast_info["results_fast_error"] = str(e)
        if speed_txt.exists():
            try:
                txt = speed_txt.read_text()
                fast_info["speed_fast_txt"] = txt
            except Exception as e:
                fast_info["speed_fast_error"] = str(e)

    # 5) baseline speed: use logging run avg tok/s
    agg["baseline_avg_tok_per_sec"] = agg.get("avg_tok_per_sec", None)
    # 6) assemble summary
    summary = {
        "config": cfg,
        "metrics": agg,
        "fast_run": fast_info,
        "files": {
            "results_logging": str(Path(run_dir/"results.logging.json").resolve()),
            "cert_checks_csv": str(Path(run_dir/"cert_checks.csv").resolve()) if aggres else None
        },
        "timestamp": time.strftime("%Y-%m-%d_%H%M%S")
    }
    # save summary
    dump_json(summary, Path(run_dir/"summary.json"))
    print("[summary written]", Path(run_dir/"summary.json").resolve())

    # cleanup
    try:
        cuda_clean()
    except Exception:
        pass

if __name__ == "__main__":
    main()
