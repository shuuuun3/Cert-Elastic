import time
from typing import List, Dict
import torch
import torch.nn as nn
import numpy as np

def logit_from_prob(p: torch.Tensor, eps: float = 1e-6):
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)

def c_eff_from_margin(gamma: torch.Tensor, alpha: float, beta: float):
    gamma = torch.clamp(gamma, 0)
    return alpha / (1.0 + beta * gamma)

@torch.inference_mode()
def decode_with_cert_logging(model: nn.Module,
                             tokenizer,
                             prompts: List[str],
                             epsilon: float,
                             topk: int,
                             alpha: float,
                             beta: float,
                             max_new_tokens: int,
                             temperature: float,
                             top_p: float,
                             top_k: int):
    """
    逐次生成で output_attentions=True を使い、各ステップ・各層で
      f(Δ,γ)=c_eff(γ)*|Δ| を記録。スキップは行わずログ収集のみ。
    """
    results = []
    device = next(model.parameters()).device
    layer_state = []  # per-layer last z1 (for Δ計算)

    for pid, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        past_key_values = None
        generated = []
        attn_logs = []
        t0 = time.time()

        # 逐次greedy（温度>0ならsamplingにしても良い）
        for step in range(max_new_tokens):
            out = model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                use_cache=True,
                past_key_values=past_key_values,
                output_attentions=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            elif top_k and top_k > 0:
                values, indices = torch.topk(logits, k=top_k, dim=-1)
                probs = torch.softmax(values, dim=-1)
                choice = torch.multinomial(probs, num_samples=1)
                next_id = indices.gather(-1, choice)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(next_id.item())

            # attention: List[num_layers] of (B,H,q_len,kv_len)
            step_layers = []
            for l, att in enumerate(out.attentions):
                p = att[:, :, -1, :].mean(dim=1)  # (B, kv_len) ヘッド平均
                k_eff = min(topk, p.size(-1)) if topk > 0 else 1
                topk_vals, _ = torch.topk(p, k=k_eff, dim=-1)
                p1 = topk_vals[:, 0]
                p2 = topk_vals[:, 1] if topk_vals.size(1) >= 2 else torch.zeros_like(p1)
                z1 = logit_from_prob(p1)
                z2 = logit_from_prob(p2)
                gamma = (z1 - z2).abs()

                if len(layer_state) <= l or "last_z1" not in layer_state[l]:
                    delta = torch.zeros_like(z1)
                else:
                    delta = (z1 - layer_state[l]["last_z1"]).abs()

                c_eff = c_eff_from_margin(gamma, alpha, beta)
                f_val = c_eff * delta
                safe = (f_val <= epsilon).float().mean().item()

                step_layers.append({
                    "layer": l,
                    "p1": float(p1.mean().item()),
                    "gamma": float(gamma.mean().item()),
                    "delta": float(delta.mean().item()),
                    "c_eff": float(c_eff.mean().item()),
                    "f": float(f_val.mean().item()),
                    "safe_ratio": safe,
                })
                if len(layer_state) <= l:
                    layer_state.append({"last_z1": z1.detach()})
                else:
                    layer_state[l]["last_z1"] = z1.detach()

            attn_logs.append(step_layers)
            input_ids = torch.cat([input_ids, next_id], dim=-1)

        t1 = time.time()
        text_out = tokenizer.decode(input_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({
            "prompt_id": pid,
            "prompt": prompt,
            "gen_text": text_out,
            "tokens": len(generated),
            "time_sec": round(t1 - t0, 3),
            "tok_per_sec": round(len(generated) / max((t1 - t0), 1e-6), 2),
            "attn_logs": attn_logs,
        })
        torch.cuda.empty_cache()

    return {"results": results}
