# run_experiment.py
"""
Run baseline vs cert-elastic on a JSONL dataset and produce summary_compare.json.

Usage example (local RTX3080):
python run_experiment.py --dataset data/myset.jsonl --model_id mistralai/Mistral-7B-Instruct-v0.3 --out_dir ./runs --max_new_tokens 64 --load_in_4bit 0 --run_cert 1

On A100 (with bitsandbytes):
python run_experiment.py --dataset data/huge.jsonl --model_id mistralai/Mistral-7B-Instruct-v0.3 --out_dir ./runs --max_new_tokens 128 --load_in_4bit 1 --run_cert 1 --device cuda:0
"""
import argparse, json, time, threading, subprocess, os
from pathlib import Path
import psutil
import csv
import sys
import math

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from cert_elastic.utils import make_run_dir, dump_json, cuda_clean
from cert_elastic.cert_core import decode_with_cert_logging  # logging routine (optional)
from cert_elastic.loader import load_model_tokenizer
from cert_elastic.cert_wrap_mistral import enable_cert_elastic_mistral
from bench.eval_metrics import aggregate_scores

# --- GPU memory polling thread
def nvidia_sampler(pid, interval, out_list, stop_event):
    # record (timestamp, gpu_mem_used(MiB)) by parsing nvidia-smi
    # requires nvidia-smi available
    import subprocess, time
    while not stop_event.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], capture_output=True, text=True)
            if r.returncode==0:
                lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
                # use first GPU
                mem = float(lines[0]) if lines else 0.0
                out_list.append((time.time(), mem))
        except Exception:
            pass
        time.sleep(interval)

def gen_one_run(model, tokenizer, prompt, max_new_tokens, temperature, top_p, top_k, device):
    # greedy generation; returns (text, tokens_generated)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature>0.0),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    out_text = tokenizer.decode(gen_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    num_tokens = gen_ids.shape[1] - inputs["input_ids"].shape[1]
    return out_text, int(num_tokens)

def run_mode(model_id, tokenizer, model, dataset, run_dir, mode, args):
    """
    mode: 'baseline' or 'cert'
    returns: dict with results list and metrics
    """
    device = next(model.parameters()).device
    results = []
    mem_samples = []
    stop_event = threading.Event()
    sampler = threading.Thread(target=nvidia_sampler, args=(os.getpid(), 0.5, mem_samples, stop_event), daemon=True)
    sampler.start()

    try:
        if mode == "cert":
            # enable wrapper (in-place)
            enable_cert_elastic_mistral(model, epsilon=args.epsilon, alpha=args.alpha, beta=args.beta)
        # run over dataset
        t_total_start = time.time()
        for rec in dataset:
            pid = rec.get("id", None)
            prompt = rec["prompt"]
            # generate and measure per-prompt time
            t0 = time.time()
            text, toks = gen_one_run(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p, args.top_k, device)
            t1 = time.time()
            results.append({"id": pid, "prompt": prompt, "pred": text, "gen_time": t1-t0, "tokens": toks})
        t_total_end = time.time()
    finally:
        stop_event.set()
        sampler.join(timeout=1)

    # compute memory stats
    mem_vals = [m for (_,m) in mem_samples] if mem_samples else []
    mem_max = max(mem_vals) if mem_vals else None
    mem_mean = sum(mem_vals)/len(mem_vals) if mem_vals else None

    # speed metrics
    total_tokens = sum([r["tokens"] for r in results])
    total_time = sum([r["gen_time"] for r in results])
    avg_tok_per_sec = (total_tokens / total_time) if total_time>0 else None
    avg_prompt_sec = (total_time/len(results)) if results else None

    # evaluation metrics if references exist
    refs_present = any("reference" in rec and rec["reference"] for rec in dataset)
    eval_res = {}
    if refs_present:
        # assemble predictions vs refs
        pred_list = []
        for r, rec in zip(results, dataset):
            pred_list.append({"id": rec.get("id"), "pred": r["pred"], "ref": rec.get("reference","")})
        eval_res = aggregate_scores(pred_list)

    out = {
        "mode": mode,
        "n_prompts": len(results),
        "total_tokens": int(total_tokens),
        "total_time_sec": float(total_time),
        "avg_tok_per_sec": float(avg_tok_per_sec) if avg_tok_per_sec else None,
        "avg_prompt_sec": float(avg_prompt_sec) if avg_prompt_sec else None,
        "mem_max_mib": mem_max,
        "mem_mean_mib": mem_mean,
        "eval_metrics": eval_res,
        "results": results
    }
    # dump per-mode results
    dump_json(out, Path(run_dir)/f"results_{mode}.json")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--dtype", default="float16", choices=["bfloat16","float16"])
    ap.add_argument("--load_in_4bit", type=int, default=0)
    ap.add_argument("--device", default=None, help="explicit device (e.g. cuda:0) or leave None for loader device_map")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--out_dir", default="./runs")
    ap.add_argument("--run_cert", type=int, default=1, help="run cert-elastic mode after baseline")
    args = ap.parse_args()

    run_dir = make_run_dir(args.out_dir)
    cfg = vars(args); dump_json(cfg, Path(run_dir/"config.run.json"))

    # load dataset (jsonl)
    dataset = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            dataset.append(json.loads(line.strip()))
    if len(dataset)==0:
        print("Empty dataset", file=sys.stderr); sys.exit(1)

    # load model/tokenizer
    model, tokenizer = load_model_tokenizer(args.model_id, args.dtype, device_map="auto", attn_impl="eager", load_in_4bit=bool(args.load_in_4bit))

    # if explicit device arg
    if args.device:
        device = torch.device(args.device)
        model.to(device)

    # baseline run
    print("[bench] running baseline...")
    baseline = run_mode(args.model_id, tokenizer, model, dataset, run_dir, mode="baseline", args=args)

    # cert-elastic run (re-load model to clear any wrapper side-effects)
    cert = None
    if args.run_cert:
        # reload model to ensure baseline state clean
        print("[bench] reloading model for cert run...")
        model2, tokenizer2 = load_model_tokenizer(args.model_id, args.dtype, device_map="auto", attn_impl="eager", load_in_4bit=bool(args.load_in_4bit))
        if args.device:
            model2.to(torch.device(args.device))
        print("[bench] running cert-elastic...")
        cert = run_mode(args.model_id, tokenizer2, model2, dataset, run_dir, mode="cert", args=args)

    # compare and produce summary
    summary = {"config": cfg, "files": {}, "timestamp": time.strftime("%Y-%m-%d_%H%M%S")}
    summary["baseline"] = baseline
    summary["files"]["baseline"] = str(Path(run_dir)/"results_baseline.json")
    if cert:
        summary["cert"] = cert
        summary["files"]["cert"] = str(Path(run_dir)/"results_cert.json")
        # compute percent changes
        def pct(a,b):
            if a is None or b is None: return None
            if a==0: return None
            return (b-a)/a
        summary["compare"] = {
            "avg_tok_per_sec_change_frac": pct(baseline.get("avg_tok_per_sec"), cert.get("avg_tok_per_sec")),
            "total_time_change_frac": pct(baseline.get("total_time_sec"), cert.get("total_time_sec")),
            "mem_max_change_frac": pct(baseline.get("mem_max_mib"), cert.get("mem_max_mib")),
            "exact_match_change": None
        }
        # if both have eval metrics
        if baseline.get("eval_metrics") and cert.get("eval_metrics"):
            bm = baseline["eval_metrics"]
            cm = cert["eval_metrics"]
            if "exact_match" in bm and "exact_match" in cm:
                summary["compare"]["exact_match_change"] = cm["exact_match"] - bm["exact_match"]

    dump_json(summary, Path(run_dir)/"summary_compare.json")
    print("Wrote summary_compare.json ->", Path(run_dir)/"summary_compare.json")
    cuda_clean()

if __name__ == "__main__":
    main()
