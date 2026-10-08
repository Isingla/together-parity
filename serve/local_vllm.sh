#!/usr/bin/env bash
# Reference path on a GPU box you control (your machine, WSL2, or a RunPod pod).
#   8B : bash serve/local_vllm.sh                                   (one 24 GB GPU)
#   70B: TP=2 bash serve/local_vllm.sh meta-llama/Llama-3.3-70B-Instruct   (2x 80 GB GPUs)
# Needs HF_TOKEN exported with the Llama license accepted on huggingface.co.
set -euo pipefail
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
TP="${TP:-1}"
python -c "import vllm" 2>/dev/null || pip install -q vllm
export REF_API_KEY="${REF_API_KEY:-local-dev-key}"
exec vllm serve "$MODEL" \
  --dtype bfloat16 \
  --tensor-parallel-size "$TP" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --seed 1234 \
  --port 8000 \
  --api-key "$REF_API_KEY"
