import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from cert_elastic.config import Config
from cert_elastic.loader import load_model_tokenizer
from cert_elastic.prompts import make_demo_prompts
from cert_elastic.cert_core import decode_with_cert_logging
from cert_elastic.utils import make_run_dir, dump_json, cuda_clean
from cert_elastic.plots import plot_safe_ratio_per_step, plot_mean_f_by_layer

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default=Config.model_id)
    ap.add_argument("--dtype", type=str, default=Config.dtype, choices=["bfloat16","float16"])
    ap.add_argument("--load_in_4bit", type=int, default=int(Config.load_in_4bit))
    ap.add_argument("--device_map", type=str, default=Config.device_map)
    ap.add_argument("--attn_impl", type=str, default=Config.attn_impl)
    ap.add_argument("--max_new_tokens", type=int, default=Config.max_new_tokens)
    ap.add_argument("--temperature", type=float, default=Config.temperature)
    ap.add_argument("--top_p", type=float, default=Config.top_p)
    ap.add_argument("--top_k", type=int, default=Config.top_k)
    ap.add_argument("--epsilon", type=float, default=Config.epsilon)
    ap.add_argument("--topk", type=int, default=Config.topk_attn)
    ap.add_argument("--alpha", type=float, default=Config.alpha)
    ap.add_argument("--beta", type=float, default=Config.beta)
    ap.add_argument("--eval_prompts_n", type=int, default=Config.eval_prompts_n)
    ap.add_argument("--out_dir", type=str, default=Config.out_dir)
    return ap.parse_args()

def main():
    args = parse_args()
    run_dir = make_run_dir(args.out_dir)
    cfg = vars(args)
    dump_json(cfg, Path(run_dir/"config.json"))

    print("[load] model:", args.model_id)
    model, tokenizer = load_model_tokenizer(args.model_id, args.dtype, args.device_map, args.attn_impl, bool(args.load_in_4bit))

    prompts = make_demo_prompts(args.eval_prompts_n)
    print("[run] decoding with Cert-Elastic logging ...")
    log = decode_with_cert_logging(
        model=model, tokenizer=tokenizer, prompts=prompts,
        epsilon=args.epsilon, topk=args.topk, alpha=args.alpha, beta=args.beta,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k
    )
    dump_json(log, Path(run_dir/"results.json"))

    # ---- aggregate
    rows = []
    for r in log["results"]:
        for step_idx, layers in enumerate(r["attn_logs"]):
            for d in layers:
                rows.append({
                    "prompt_id": r["prompt_id"], "step": step_idx, "layer": d["layer"],
                    "f": d["f"], "gamma": d["gamma"], "delta": d["delta"],
                    "c_eff": d["c_eff"], "safe_ratio": d["safe_ratio"]
                })
    df = pd.DataFrame(rows)
    df.to_csv(Path(run_dir/"cert_metrics.csv"), index=False)

    layer_mean = df.groupby("layer").mean(numeric_only=True).reset_index()
    step_mean = df.groupby("step").mean(numeric_only=True).reset_index()
    layer_mean.to_csv(Path(run_dir/"cert_layer_mean.csv"), index=False)
    step_mean.to_csv(Path(run_dir/"cert_step_mean.csv"), index=False)

    plot_safe_ratio_per_step(df, Path(run_dir/"safe_ratio_per_step.png"))
    plot_mean_f_by_layer(df, Path(run_dir/"mean_f_by_layer.png"))

    speeds = [r["tok_per_sec"] for r in log["results"]]
    with open(Path(run_dir/"speed.txt"), "w") as f:
        f.write(f"avg tok/s: {np.mean(speeds):.2f}\n")

    print("[done]", run_dir)
    cuda_clean()

if __name__ == "__main__":
    main()
