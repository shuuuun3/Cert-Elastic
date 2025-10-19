# dlm/interface.py
from typing import Dict, Any, List, Tuple
import torch

class DLMEngine:
    """
    拡散LLM用の最小インタフェース。
    1サンプル（バッチ1）での反復denoiseをstep実行し、各層の注意行列S_t,lと
    KV/hiddenの内部状態アクセス手段を提供すること。
    """

    def to(self, device: torch.device): raise NotImplementedError

    def tokenize(self, text: str) -> torch.LongTensor:
        """プロンプトをトークン化。"""
        raise NotImplementedError

    def detokenize(self, ids: torch.LongTensor) -> str:
        raise NotImplementedError

    def init_state(self, prompt_ids: torch.LongTensor, gen_len: int) -> Dict[str, Any]:
        """
        生成長(gen_len)までMASK初期化した時刻t=0の状態を作る。
        返り値stateには、少なくとも
          - 'x_t' : 現在シーケンス（MASK含む）
          - 'decoded_pos': D<t（既に確定した位置の集合: list[int]）
          - 'masked_pos':  M_t（未確定位置の集合: list[int]）
          - 'kv' : 層ごとのKVキャッシュ
        を含める。
        """
        raise NotImplementedError

    def step_denoise(
        self,
        state: Dict[str, Any],
        query_positions: List[int],
        refresh_from_layer: int | None,
        reuse_shallow_cache: bool,
    ) -> Tuple[Dict[str, Any], List[torch.Tensor], List[int]]:
        """
        1ステップのdenoiseを実行。
        - query_positions: 今回再計算する位置（スライディング窓 ∪ 直近の最注目トークン集合）
        - refresh_from_layer: l*+1 を指定すると、その層以深はKV再計算。Noneなら全面再利用。
        - reuse_shallow_cache: 浅い層はキャッシュ再利用フラグ。
        戻り値:
          - 新state
          - attns: 各層の注意S_t,l（torch.Tensor[Q,N]）
          - newly_decoded: 新たに確定した位置集合 D_t（list[int]）
        """
        raise NotImplementedError
