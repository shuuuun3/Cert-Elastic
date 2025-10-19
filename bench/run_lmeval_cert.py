# bench/run_lmeval_cert.py（先頭～mainのtasks処理のみ重要差分）
import argparse, json
from pathlib import Path
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from bench.lmeval_cert_runner import HFCertElasticLM
from cert_elastic.utils import make_run_dir, dump_json
import os

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

ALIASES = {
    "humaneval": ["openai_humaneval", "humaneval", "human_eval", "humaneval_python"],
    "mbpp": ["mbpp", "mbpp_sanitized", "mbppplus", "mbpp_plus"],
    "gsm8k": ["gsm8k"],
    "hendrycks_math": ["hendrycks_math", "math"],
}

def resolve_tasks(user_csv: str) -> list[str]:
    tm = TaskManager()
    available = set(tm.task_index.keys())
    out = []
    for raw in [t.strip() for t in user_csv.split(",") if t.strip()]:
        # 完全一致を優先
        if raw in available:
            out.append(raw); continue
        # エイリアス解決
        cands = ALIASES.get(raw.lower(), [raw.lower()])
        hit = next((c for c in cands if c in available), None)
        if hit is None:
            raise SystemExit(
                f"[error] task '{raw}' not found. available example hits: "
                f"{[t for t in available if any(a in t for a in cands[:2])] or list(sorted(available))[:20]}"
            )
        out.append(hit)
    print("[tasks]", out)
    return out

def pick_primary(name, resdict):
    name = name.split(":")[0].lower()
    for k in ({"gsm8k":"exact_match","hendrycks_math":"exact_match",
               "humaneval":"pass@1","mbpp":"pass@1"}.get(name), "exact_match","pass@1","acc","accuracy"):
        if k and k in resdict and isinstance(resdict[k], (int,float)):
            return k, float(resdict[k])
    for k,v in resdict.items():
        if isinstance(v,(int,float)): return k, float(v)
    return None, None

def run_once(enable_cert: bool, args):
    lm = HFCertElasticLM(
        pretrained=args.model_id, dtype=args.dtype, device=(args.device or "auto"),
        enable_cert=enable_cert, epsilon=args.epsilon, alpha=args.alpha, beta=args.beta,
        attn_impl=args.attn_impl, batch_size=1, trust_remote_code=True,
    )
    tasks = resolve_tasks(args.tasks)
    import sys
    if sys.platform == "win32":
        tasks = [t for t in tasks if t not in ("humaneval","openai_humaneval","mbpp","mbpp_sanitized")]
        print("[win32] code-eval 未対応のため HumanEval/MBPP を除外:", tasks)
    # Allow running tasks marked as "unsafe" (e.g. mbpp) by confirming explicitly.
    # lm_eval will execute task code when these tasks require code evaluation.
    res = evaluator.simple_evaluate(model=lm, tasks=tasks, num_fewshot=args.fewshot,
                                    limit=args.limit, bootstrap_iters=0,
                                    confirm_run_unsafe_code=True)
    ours = {}
    for tname, tres in res["results"].items():
        k, v = pick_primary(tname, tres)
        ours[tname.lower()] = {"metric_key": k, "accuracy": v}
    toks = (lm.total_new_tokens / lm.total_gen_time) if lm.total_gen_time>0 else None
    return {"results_raw": res, "ours_primary": ours, "tokens_sec": toks}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--dtype", default="float16", choices=["float16","bfloat16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--attn_impl", default="eager")
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--tasks", default="gsm8k,humaneval,mbpp,hendrycks_math")
    ap.add_argument("--fewshot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out_dir", default="./runs")
    args = ap.parse_args()

    run_dir = make_run_dir(args.out_dir)
    base = run_once(False, args); dump_json(base, Path(run_dir/"lmeval_baseline.json"))
    cert = run_once(True,  args); dump_json(cert, Path(run_dir/"lmeval_cert.json"))
    out_for_viz = {k: {"accuracy": v["accuracy"], "tokens_sec": cert["tokens_sec"]}
                   for k,v in cert["ours_primary"].items()}
    dump_json(out_for_viz, Path(run_dir/"results.certelastic.paperstyle.json"))
    print("[done]", Path(run_dir).resolve())

if __name__ == "__main__":
    main()
