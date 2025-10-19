from dataclasses import dataclass

@dataclass
class Config:
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"   # 7B既定。8B級は4bitを推奨
    dtype: str = "bfloat16"                                 # "float16"でも可
    load_in_4bit: bool = True                               # RTX3080向け
    device_map: str = "auto"
    attn_impl: str = "eager"                                # attention確率取得のため
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    # Cert-Elastic 判定
    epsilon: float = 0.02
    topk_attn: int = 1
    alpha: float = 0.5                                      # c_eff(γ)=α/(1+βγ)
    beta: float = 2.0
    # 評価
    eval_prompts_n: int = 20
    out_dir: str = "./runs"
    seed: int = 42
