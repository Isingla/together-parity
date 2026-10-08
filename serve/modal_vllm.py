"""Reference path: the SAME model in full BF16 precision, served by stock vLLM on Modal.

One-time setup (PowerShell or bash):
    pip install modal
    modal setup
    modal secret create parity-ref HF_TOKEN=hf_xxx VLLM_API_KEY=<any-random-string>
    # HF_TOKEN must have accepted the Llama 3.3 license on huggingface.co

Run (prints the URL; keep the terminal open while benchmarking):
    modal serve serve/modal_vllm.py
    # or for 8B:  $env:PARITY_MODEL="meta-llama/Llama-3.1-8B-Instruct"; $env:PARITY_GPU="L40S:1"

Then:  REF_BASE_URL=https://<printed-url>/v1   REF_API_KEY=<the VLLM_API_KEY above>

Cost: 2x H100 for ~1-1.5 h (first boot downloads ~140 GB of weights into a
Volume; later boots reuse it). Stop the app as soon as both quality runs and the
speed run finish - an idle GPU still bills until scaledown.
"""
import os
import subprocess

import modal

MODEL = os.environ.get("PARITY_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
GPU = os.environ.get("PARITY_GPU", "H100:2")
TP = int(GPU.split(":")[1]) if ":" in GPU else 1
MAX_LEN = int(os.environ.get("PARITY_MAX_LEN", "8192"))  # 70B BF16 on 2x80GB leaves little KV room

image = (
    modal.Image.debian_slim(python_version="3.12")
    # Unpinned on purpose for the first run; the harness records the server's
    # /version in results/<profile>/meta.json. Pin it here afterwards.
    .pip_install("vllm", "huggingface_hub[hf_transfer]")
    # bake the choice into the container so the remote process sees the same values
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PARITY_MODEL": MODEL, "PARITY_GPU": GPU,
          "PARITY_MAX_LEN": str(MAX_LEN)})
)
hf_cache = modal.Volume.from_name("parity-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("parity-vllm-cache", create_if_missing=True)

app = modal.App("parity-ref")


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("parity-ref")],
    timeout=3 * 60 * 60,
    scaledown_window=10 * 60,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=40 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--host", "0.0.0.0", "--port", "8000",
        "--dtype", "bfloat16",               # the un-quantized reference
        "--tensor-parallel-size", str(TP),
        "--max-model-len", str(MAX_LEN),
        "--gpu-memory-utilization", "0.92",
        "--seed", "1234",
        "--api-key", os.environ["VLLM_API_KEY"],
    ]
    print(" ".join(c if c != os.environ["VLLM_API_KEY"] else "***" for c in cmd))
    subprocess.Popen(cmd)
