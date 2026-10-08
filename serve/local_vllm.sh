#!/usr/bin/env bash
# Reference path on your own GPU (Linux or WSL2). 8B BF16 needs ~16 GB weights + KV;
# a 24 GB card is comfortable. 70B BF16 needs ~150 GB - use serve/modal_vllm.py.
set -euo pipefail
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
pip install -q vllm
export REF_API_KEY="${REF_API_KEY:-local-dev-key}"
exec vllm serve "$MODEL" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --seed 1234 \
  --port 8000 \
  --api-key "$REF_API_KEY"
