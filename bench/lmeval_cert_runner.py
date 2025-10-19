# bench/lmeval_cert_runner.py
import time
from typing import Optional, List, Dict, Any
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from cert_elastic.cert_wrap_mistral import enable_cert_elastic_mistral

class HFCertElasticLM(HFLM):
    """
    HFLM を継承して Cert-Elastic を注入。
    - enable_cert=True で Mistral 層forwardをラップ
    - tokens/sec を内部で計測（generateのみ対象）
    """
    def __init__(
        self,
        pretrained: str,
        dtype: str = "float16",
        device: str = "auto",
        enable_cert: bool = False,
        epsilon: float = 0.02,
        alpha: float = 0.5,
        beta: float = 2.0,
        attn_impl: str = "eager",
        **hf_kwargs,
    ):
        self._pretrained = pretrained
        self._dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self._device_map = device  # "auto" or "cuda:0"
        self._enable_cert = enable_cert
        self._eps = epsilon
        self._alpha = alpha
        self._beta = beta
        self._attn_impl = attn_impl
        self._hf_kwargs = hf_kwargs

        # 計測
        self.total_gen_time = 0.0
        self.total_new_tokens = 0

        # HFLM初期化（model/tokenizerは _model, _tokenizer に入る）
        super().__init__(pretrained=pretrained, dtype=dtype, device=device, **hf_kwargs)

        # Cert-Elastic を必要に応じて有効化するフラグ
        self._cert_enabled_applied = False

    # モデル/トークナイザ取得（HFLMの実装差異に対応: model/tokenizer または _model/_tokenizer）
    def _get_model(self):
        return getattr(self, "model", getattr(self, "_model", None))

    def _get_tokenizer(self):
        return getattr(self, "tokenizer", getattr(self, "_tokenizer", None))

    # ---- HFLMの load を上書きして Cert-Elastic を適用
    def generate_until(self, requests):
        outs = []
        for req in requests:
            # lm-eval 0.4.x: req is Instance
            context = None
            until = None
            max_new_tokens = 128

            if hasattr(req, "args"):
                a = req.args
                if isinstance(a, dict):
                    context = a.get("context") or a.get("prompt") or a.get("inputs")
                    until = a.get("until") or a.get("stop_sequences")
                    max_new_tokens = int(a.get("max_gen_toks", a.get("max_tokens", 128)))
                elif isinstance(a, (list, tuple)):
                    # 位置引数スタイル: (context, genconf)
                    if len(a) >= 1:
                        context = a[0]
                    genconf = a[1] if len(a) >= 2 and isinstance(a[1], dict) else {}
                    until = genconf.get("until") or genconf.get("stop_sequences")
                    max_new_tokens = int(genconf.get("max_gen_toks", genconf.get("max_tokens", 128)))
            else:
                # 旧APIフォールバック: (context, genconf)
                context, genconf = req
                until = genconf.get("until")
                max_new_tokens = int(genconf.get("max_gen_toks", 128))

            if isinstance(context, list) and context:
                context = context[0]
            assert isinstance(context, str) and len(context) > 0, "empty context"

            tok = self._get_tokenizer()
            mdl = self._get_model()
            assert tok is not None, "tokenizer is not loaded"
            assert mdl is not None, "model is not loaded"

            # 必要なら一度だけ Cert-Elastic を適用
            if self._enable_cert and not self._cert_enabled_applied:
                try:
                    # 可能なら注意機構の実装指定を反映（存在しない場合は無視）
                    if hasattr(mdl, "config") and hasattr(mdl.config, "attn_implementation"):
                        mdl.config.attn_implementation = self._attn_impl
                    enable_cert_elastic_mistral(mdl, epsilon=self._eps, alpha=self._alpha, beta=self._beta)
                    self._cert_enabled_applied = True
                except Exception:
                    # モデル型が対象外などの場合はそのまま通常動作
                    self._cert_enabled_applied = False

            enc = tok(context, return_tensors="pt")
            # device は最初のパラメータから取得
            dev = next(mdl.parameters()).device
            input_ids = enc["input_ids"].to(dev)
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
            )
            t0 = time.perf_counter()
            out = mdl.generate(input_ids, **gen_kwargs)
            t1 = time.perf_counter()

            new_tokens = int(out.shape[1] - input_ids.shape[1])
            text = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)

            if until:
                for u in until:
                    i = text.find(u)
                    if i >= 0:
                        text = text[:i]
                        break

            self.total_gen_time += (t1 - t0)
            self.total_new_tokens += max(new_tokens, 0)
            outs.append(text)
        return outs