import math
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn

# Mistral内部のRoPE適用関数
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

def _logit_from_prob(p: torch.Tensor, eps: float = 1e-6):
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)

def _c_eff(gamma: torch.Tensor, alpha: float, beta: float):
    gamma = torch.clamp(gamma, 0)
    return alpha / (1.0 + beta * gamma)

def _cheap_z1z2_for_layer(layer, hidden_states: torch.Tensor,
                          past_k: torch.Tensor, position_ids: torch.Tensor,
                          state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
    層の入力 hidden_states から q を計算（q_proj + RoPE）し、
    前ステップ保持の top1/top2 位置にある K とだけ内積を取り z1,z2 を近似。
    返り値: z1, z2, gamma=|z1-z2|, delta=|z1 - last_z1|, idx1, idx2
    """
    bsz, seqlen, hdim = hidden_states.shape
    assert bsz == 1, "batch=1のみ対応"

    # Pre-norm → q_proj
    q_in = layer.input_layernorm(hidden_states)  # (1, 1, hidden_size) を想定（逐次生成）
    q = layer.self_attn.q_proj(q_in)            # (1, 1, hidden_size)

    num_heads = layer.self_attn.num_heads
    head_dim  = layer.self_attn.head_dim
    assert hdim == num_heads * head_dim

    # 形状: (1, 1, nH, dH) → (1, nH, 1, dH)
    q = q.view(1, seqlen, num_heads, head_dim).transpose(1, 2)

    # RoPE適用（K側は既に適用済みでcacheに入っている前提）
    # rotary_emb は (cos, sin) を返す。position_idsは(1,1)
    cos, sin = layer.self_attn.rotary_emb(q, seq_len=past_k.shape[2] + 1)
    q, _ = apply_rotary_pos_emb(q, None, cos, sin, position_ids)

    # 参照する top1/top2 のインデックス（前回値を使う。未初期化なら末尾とその一つ前）
    seq_len = past_k.shape[2]
    idx1 = state.get("top1_idx", seq_len - 1)
    idx2 = state.get("top2_idx", max(seq_len - 2, 0))
    idx1 = int(max(0, min(seq_len - 1, idx1)))
    idx2 = int(max(0, min(seq_len - 1, idx2)))

    k1 = past_k[:, :, idx1, :]   # (1, nH, dH)
    k2 = past_k[:, :, idx2, :]

    scale = 1.0 / math.sqrt(head_dim)
    # (1, nH, 1, dH)*(1, nH, dH) → (1, nH)
    z1_h = (q[:, :, -1, :] * k1).sum(-1) * scale
    z2_h = (q[:, :, -1, :] * k2).sum(-1) * scale
    z1 = z1_h.mean(dim=1)  # (1,)
    z2 = z2_h.mean(dim=1)  # (1,)

    last_z1 = state.get("last_z1", z1.detach())
    delta = (z1 - last_z1).abs()
    gamma = (z1 - z2).abs()

    return z1, z2, gamma, delta, idx1, idx2

def enable_cert_elastic_mistral(model: nn.Module, epsilon: float, alpha: float, beta: float):
    """
    MistralForCausalLM の各層forwardをラップして Cert-Elastic を有効化。
    - f(Δ,γ)=c_eff(γ)|Δ| <= ε で skip:
        * 出力 hidden_states は恒等通過
        * KVは "直前エントリ複製" で長さ+1
    - それ以外は元のforward

    制約: batch=1, generate/逐次decode前提。検証用に十分。
    """
    if model.__class__.__name__ != "MistralForCausalLM":
        raise NotImplementedError("MistralForCausalLM 以外は未対応")

    layers = model.model.layers
    cert_states: Dict[int, Dict[str, torch.Tensor]] = {}

    for lid, layer in enumerate(layers):
        orig_forward = layer.forward

        def make_wrapper(lid, layer, orig_forward):
            def wrapped_forward(
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor = None,
                position_ids: torch.Tensor = None,
                past_key_value: Tuple[torch.Tensor, torch.Tensor] = None,
                output_attentions: bool = False,
                use_cache: bool = True,
                **kwargs
            ):
                # HFの新旧API両対応: 重複して渡されたキャッシュ引数を整理（警告抑制）
                if 'past_key_values' in kwargs and past_key_value is not None:
                    kwargs.pop('past_key_values')
                # 安全側条件: 以下でなければ元のforward
                if (hidden_states.size(0) != 1) or (past_key_value is None) or (position_ids is None):
                    return orig_forward(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs
                    )

                k_cache, v_cache = past_key_value  # (1, nH, T, dH)
                state = cert_states.setdefault(lid, {})

                # 近似 z1,z2 の計算（qのみ投影 + RoPE）
                with torch.no_grad():
                    z1, z2, gamma, delta, idx1, idx2 = _cheap_z1z2_for_layer(
                        layer, hidden_states, k_cache, position_ids, state
                    )
                    c = _c_eff(gamma, alpha=alpha, beta=beta)
                    f_val = (c * delta).item()

                if f_val <= epsilon:
                    # ---- skip: 恒等通過 + KV複製で長さ+1
                    new_hidden = hidden_states
                    new_k = torch.cat([k_cache, k_cache[:, :, -1:, :]], dim=2).contiguous()
                    new_v = torch.cat([v_cache, v_cache[:, :, -1:, :]], dim=2).contiguous()
                    present = (new_k, new_v)

                    # 次回のために state 更新
                    state["last_z1"] = z1.detach()
                    state["top1_idx"] = idx1
                    state["top2_idx"] = idx2

                    # 出力の形は (hidden_states, self_attn_weights, present_key_value)
                    return (new_hidden, None, present)
                else:
                    # ---- 通常計算: 出力attentionは不要（速度）
                    out = orig_forward(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=False,
                        use_cache=use_cache,
                        **kwargs
                    )
                    # out: (hidden_states, attn_weights(None), present_kv)
                    # 近似z1は更新しておく（skip/非skip一貫のため）
                    state["last_z1"] = z1.detach()
                    state["top1_idx"] = idx1
                    state["top2_idx"] = idx2
                    return out
            return wrapped_forward
        layer.forward = make_wrapper(lid, layer, orig_forward)
