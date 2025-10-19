import argparse, json, time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from cert_elastic.cert_wrap_mistral import enable_cert_elastic_mistral
from cert_elastic.prompts import make_demo_prompts
from cert_elastic.utils import make_run_dir, dump_json

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16","float16"])
    ap.add_argument("--load_in_4bit", type=int, default=1)
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--attn_impl", type=str, default="eager")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    # Cert-Elastic
    ap.add_argument("--epsilon", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--eval_prompts_n", type=int, default=20)
    ap.add_argument("--out_dir", type=str, default="./runs")
    return ap.parse_args()

def main():
    args = parse_args()
    run_dir = make_run_dir(args.out_dir)
    dump_json(vars(args), Path(run_dir/"config.fast.json"))

    compute_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    qconf = None
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            qconf = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
            )
        except Exception as e:
            print("[warn] bitsandbytes未適用:", e)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(torch_dtype=compute_dtype, device_map=args.device_map, attn_implementation=args.attn_impl)
    if qconf is not None:
        kwargs["quantization_config"] = qconf

    print("[load] model:", args.model_id)
    mdl = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs).eval()

    # --- Cert-Elastic 有効化（層forwardをラップ）
    enable_cert_elastic_mistral(mdl, epsilon=args.epsilon, alpha=args.alpha, beta=args.beta)
    print("[cert] enabled: epsilon=%.4f alpha=%.3f beta=%.3f" % (args.epsilon, args.alpha, args.beta))

    prompts = make_demo_prompts(args.eval_prompts_n)

    # --- 逐次生成（batch=1）
    results = []
    t0_all = time.time()
    for i, prompt in enumerate(prompts):
        inputs = tok(prompt, return_tensors="pt").to(mdl.device)
        # HF generate でもOK（内部で順次forward→ラップが効く）。速度測定用にgreedy。
        gen_ids = mdl.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=(args.temperature > 0),
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )
        text = tok.decode(gen_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({"prompt_id": i, "prompt": prompt, "gen_text": text})
    t1_all = time.time()

    dump_json(results, Path(run_dir/"results.fast.json"))
    with open(Path(run_dir/"speed.fast.txt"), "w") as f:
        f.write(f"total_sec: {t1_all - t0_all:.3f}\n")
        f.write(f"avg_per_prompt_sec: {(t1_all - t0_all)/max(len(prompts),1):.3f}\n")

    print("[done]", run_dir)

if __name__ == "__main__":
    main()
