import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_tokenizer(model_id: str, dtype_str: str, device_map: str, attn_impl: str, load_in_4bit: bool):
    compute_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    qconf = None
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            print("[warn] bitsandbytes未利用で継続:", e)

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = dict(torch_dtype=compute_dtype, device_map=device_map, attn_implementation=attn_impl)
    if qconf is not None:
        kwargs["quantization_config"] = qconf

    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    mdl.eval()
    return mdl, tok
