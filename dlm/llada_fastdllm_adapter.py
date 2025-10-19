# dlm/llada_fastdllm_adapter.py
from typing import Dict, Any, List, Tuple
import torch
from .interface import DLMEngine

class LLaDAFastDLLMEngine(DLMEngine):
    def __init__(self, llada_ckpt: str, device: str = "cuda"):
        """
        llada_ckpt: HF もしくはローカルの LLaDA 系チェックポイント名
        """
        # ここでは擬似コード。実際には Fast-dLLM / LLaDA のロード関数を呼ぶ。
        try:
            from fast_dllm.api import load_llada_model  # 仮のAPI名（実装環境に合わせて修正）
        except Exception as e:
            raise RuntimeError("Fast-dLLM/LLaDA のAPIに合わせてこの部分を実装してください") from e
        self.model, self.tokenizer = load_llada_model(llada_ckpt, device=device)
        self.device = torch.device(device)

    def to(self, device: torch.device): self.device = device; self.model.to(device)

    def tokenize(self, text: str) -> torch.LongTensor:
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        return ids[0]

    def detokenize(self, ids: torch.LongTensor) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def init_state(self, prompt_ids: torch.LongTensor, gen_len: int) -> Dict[str, Any]:
        # モデル側の「全MASKから順次unmaskする拡散初期化」を利用。
        # ここもFast-dLLM/LLaDAの実装に従って置換。
        x_t, kv = self.model.init_mask_sequence(prompt_ids, gen_len)   # 仮API
        decoded_pos = list(range(len(prompt_ids)))
        masked_pos  = list(range(len(prompt_ids), len(prompt_ids)+gen_len))
        return {"x_t": x_t, "kv": kv, "decoded_pos": decoded_pos, "masked_pos": masked_pos}

    def step_denoise(
        self,
        state: Dict[str, Any],
        query_positions: List[int],
        refresh_from_layer: int | None,
        reuse_shallow_cache: bool,
    ) -> Tuple[Dict[str, Any], List[torch.Tensor], List[int]]:
        """
        注意S_t,lの取得と、refresh_from_layer以深の再計算（KV更新）をモデルに指示。
        """
        x_t, kv = state["x_t"], state["kv"]
        # 仮API: model.denoise_step は、与えたクエリ位置のS_t,l、更新後x_t/kv、新規確定位置、信頼度マスクを返す
        x_new, kv_new, attns, newly_decoded = self.model.denoise_step(
            x_t, kv,
            query_positions=query_positions,
            refresh_from_layer=refresh_from_layer,
            reuse_shallow_cache=reuse_shallow_cache,
        )
        decoded_pos = sorted(set(state["decoded_pos"] + newly_decoded))
        masked_pos  = [i for i in state["masked_pos"] if i not in newly_decoded]
        new_state = {"x_t": x_new, "kv": kv_new, "decoded_pos": decoded_pos, "masked_pos": masked_pos}
        return new_state, attns, newly_decoded
