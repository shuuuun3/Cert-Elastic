# Cert-Elastic (local, RTX3080)

- 目的: Elastic-Cacheの経験則に「誤差上界」(f(Δ,γ)≤ε)を付与してKV再利用の安全判定をログ収集・可視化。
- 既定モデル: mistralai/Mistral-7B-Instruct-v0.3（8B級は4bit前提）

## セットアップ (Windows / PowerShell)

PyTorch の公式ホイールは現時点で Python 3.13 を未サポートのため、Python 3.12 を利用してください。

```powershell
# (任意) Python 3.12 が無い場合はインストール
winget install -e --id Python.Python.3.12 -s winget

# 仮想環境の作成と有効化
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 依存関係のインストール
python -m pip install --upgrade pip
# GPU (CUDA) を使う場合は先に PyTorch をインストール（例: CUDA 12.4）
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
# 残りの依存をインストール
python -m pip install -r requirements.txt
```

注意: Windows では `bitsandbytes` は使用しません（requirements.txt でも Linux 限定にしています）。

## 実行

以下のいずれかで実行できます。

```powershell
# ルートのスクリプトを直接実行
python .\main.py --model_id mistralai/Mistral-7B-Instruct-v0.3 --epsilon 0.02 --topk 1 --max_new_tokens 128

# もしくはパッケージエントリ（cert_elastic/__main__.py）経由
python -m cert_elastic --model_id mistralai/Mistral-7B-Instruct-v0.3 --epsilon 0.02 --topk 1 --max_new_tokens 128
```

## 出力

`runs/run_YYYYmmdd_HHMMSS/` 以下に成果物を保存します。

```text
runs/run_YYYYmmdd_HHMMSS/
  ├─ config.json
  ├─ results.json
  ├─ cert_metrics.csv
  ├─ cert_layer_mean.csv
  ├─ cert_step_mean.csv
  ├─ safe_ratio_per_step.png
  └─ mean_f_by_layer.png
```
