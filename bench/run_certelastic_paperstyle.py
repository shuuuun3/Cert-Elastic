# bench/run_certelastic_paperstyle.py
import argparse, json, time, os
from pathlib import Path
import torch

from cert_elastic.loader import load_model_tokenizer
from cert_elastic.cert_wrap_mistral import enable_cert_elastic_mistral
from cert_elastic.utils import make_run_dir, dump_json, cuda_clean

from .datasets_paper import (
    load_gsm8k, judge_gsm8k,
    load_math, judge_math,
    load_humaneval, judge_humaneval,
    load_mbpp, judge_mbpp
)

TASKS = {
    "gsm8k":     (load_gsm8k,     judge_gsm8k),
    "math":      (load_math,      judge_math),
    "humaneval": (load_humaneval, judge_humaneval),
    "mbpp":      (load_mbpp,      judge_mbpp),
}

def gen_one(mdl, tok, prompt, gen_len, temperature, top_p, top_k):
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(
        **inputs,
        max_new_tokens=gen_len,
        do_sample=(temperature>0.0),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
    )
    pred = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    gen_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    return pred, int(gen_tokens)

def run_task(mdl, tok, task, n_items, gen_len, temperature, top_p, top_k):
    loader, judge = TASKS[task]
    data = loader(n=n_items)
    total_sec = 0.0
    total_tok = 0
    correct = 0
    for ex in data:
        t0 = time.perf_counter()
        pred, toks = gen_one(mdl, tok, ex["prompt"], gen_len, temperature, top_p, top_k)
        dt = time.perf_counter() - t0
        total_sec += dt
        total_tok += toks
        if judge(ex.get("answer",""), pred): correct += 1
    tps = total_tok / total_sec if total_sec > 0 else 0.0
    acc = correct / max(1,len(data))
    return {"tokens_sec": tps, "accuracy": acc, "n": len(data), "gen_len": gen_len}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16"])
    ap.add_argument("--load_in_4bit", type=int, default=0)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--attn_impl", default="eager")
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--tasks", default="gsm8k,math,humaneval,mbpp")
    ap.add_argument("--n_items", type=int, default=512)
    ap.add_argument("--gen_len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--out", default="runs/results.certelastic.paperstyle.json")
    args = ap.parse_args()

    run_dir = make_run_dir("./runs")
    dump_json(vars(args), Path(run_dir/"config.paperstyle.json"))

    # モデル読込
    mdl, tok = load_model_tokenizer(args.model_id, args.dtype, args.device_map, args.attn_impl, bool(args.load_in_4bit))
    # Cert-Elastic（層スキップ）有効化
    enable_cert_elastic_mistral(mdl, epsilon=args.epsilon, alpha=args.alpha, beta=args.beta)

    # 実行
    results = {}
    for task in args.tasks.split(","):
        task = task.strip()
        print(f"[run] {task} n={args.n_items} gen_len={args.gen_len}")
        results[task] = run_task(
            mdl, tok, task, args.n_items, args.gen_len,
            args.temperature, args.top_p, args.top_k
        )

    # 保存（vizがそのまま読める形）
    out_path = Path(run_dir) / Path(args.out).name
    dump_json(results, out_path)
    print("[done]", out_path.resolve())
    cuda_clean()

if __name__ == "__main__":
    main()
