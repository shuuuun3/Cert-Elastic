# dlm/elastic_cache.py
from typing import Dict, Any, List, Tuple
import torch, math
from .interface import DLMEngine

@torch.no_grad()
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: [m, n] 同次元。行列としてcos類似度（全要素）を算出。
    va = a.reshape(-1).float(); vb = b.reshape(-1).float()
    denom = (va.norm() * vb.norm()).item()
    return 1.0 if denom == 0 else (va @ vb).item() / denom

class CertElasticRunner:
    """
    Elastic-Cache (論文の Algorithm 1 相当) の実装。
    主要ハイパラ:
      beta:  スライディングウィンドウ幅 (β)
      gamma: 注意分布のcos類似度閾値 (γ)。下回った層の次層から再計算。
      conf_eps: 信頼度しきい値 (ϵ) による並列unmask（Fast-dLLM互換）
    """
    def __init__(self, engine: DLMEngine, beta: int = 16, gamma: float = 0.9, conf_eps: float = 0.9):
        self.engine = engine
        self.beta = beta
        self.gamma = gamma
        self.conf_eps = conf_eps

    def run_generate(
        self,
        prompt: str,
        gen_len: int,
        max_steps: int = 4096,
        collect_traces: bool = False,
    ) -> Dict[str, Any]:
        tok = self.engine.tokenize(prompt)
        state = self.engine.init_state(tok, gen_len)
        traces = []
        T_prev: List[int] = []   # 直近ステップの"最注目トークン"集合（層ごと重複あり→集合化）
        attn_prev: List[torch.Tensor] | None = None

        step = 0
        while state["masked_pos"] and step < max_steps:
            step += 1
            # スライディング窓 Mt_β
            mpos = state["masked_pos"]
            window = sorted(mpos)[: self.beta]

            # 直近の最注目トークンを窓に加える（軽量トリガ）
            query_positions = sorted(set(window + T_prev))

            # まずキャッシュ再利用で前向き
            new_state, attns, newly = self.engine.step_denoise(
                state, query_positions=query_positions,
                refresh_from_layer=None, reuse_shallow_cache=True
            )

            # 各層の"最注目トークン" T_t,l を抽出（列最大の列インデックス）
            # attns[l]: [Q, N]（Qはquery_positionsの数, Nは全文脈）
            top_tokens_per_layer: List[int] = []
            for A in attns:
                # 各列への総注目量でトップ列を求める（列方向sum→argmax）
                # 実装簡略化：Aの総和ベクトルを列方向で求める
                col_sum = A.float().sum(dim=0)  # [N]
                top_tokens_per_layer.append(int(col_sum.argmax().item()))
            T_cur = sorted(set(top_tokens_per_layer))

            # 注意分布の変化を層ごとに測る（cos類似度）
            l_star = None
            if attn_prev is not None:
                for l, (A_prev, A_cur) in enumerate(zip(attn_prev, attns)):
                    # 直前ステップで選ばれたT_prevの列だけを比較（軽量）
                    if len(T_prev) > 0:
                        A_prev_sub = A_prev[:, T_prev]  # [Q_prev, |T_prev|]
                        A_cur_sub  = A_cur[:,  T_prev]
                        sim = cosine_similarity(A_prev_sub, A_cur_sub)
                    else:
                        sim = 1.0
                    if sim < self.gamma and l_star is None:
                        l_star = l  # この層の次層から更新
                if l_star is not None:
                    # 深い層のみ再計算
                    new_state, attns, newly = self.engine.step_denoise(
                        state, query_positions=query_positions,
                        refresh_from_layer=l_star + 1,
                        reuse_shallow_cache=True
                    )

            # 信頼度しきい値(ϵ)で並列unmask（engine側でconfidence算出してnewly選別する前提）
            state = new_state
            attn_prev = attns
            T_prev = T_cur

            if collect_traces:
                traces.append({
                    "step": step,
                    "window": window,
                    "top_tokens": T_prev,
                    "decoded_now": newly,
                    "decoded_total": len(state["decoded_pos"]),
                    "masked_total": len(state["masked_pos"]),
                })

        text = self.engine.detokenize(state["x_t"])
        return {"text": text, "steps": step, "traces": traces}
