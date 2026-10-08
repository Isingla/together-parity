# Does Together's FP8 "Turbo" serving change Llama 3.3 70B's answers?

*Ishaan Singla · Oct 8, 2026 · [repo](https://github.com/<you>/together-parity)*

## Question
Together AI says its serving stack is much faster while keeping output quality. Its
`Llama-3.3-70B-Instruct-Turbo` endpoint is listed as FP8-quantized. I ran the same model
through Together and through the full-precision BF16 weights on stock vLLM, on identical
prompts, and asked two things: does the optimized path change answers, and how much faster is it?

## Setup
- **Reference:** Llama 3.3 70B Instruct, BF16, vLLM 0.31.0, 2× H100 on Modal, tensor parallel 2.
  Weights from `unsloth/Llama-3.3-70B-Instruct`, an ungated mirror of Meta's BF16 release
  (my Meta access request was still pending).
- **Together:** `meta-llama/Llama-3.3-70B-Instruct-Turbo` via the serverless API, Oct 8 2026, ~1–2 pm ET.
- **Tasks:** GSM8K test (all 1,319) and HumanEval (all 164). Greedy decoding, seed 1234,
  identical prompt templates. HumanEval graded by running the hidden tests.
- Each endpoint ran the full set **twice**, to measure how much it disagrees with itself.

## Result: quality — equivalent
| | Reference (BF16) | Together (FP8) | Δ (95% CI) | McNemar p |
|---|---|---|---|---|
| GSM8K (n=1,319) | 96.2% | 95.7% | −0.5 pp [−1.2, +0.2] | 0.19 |
| HumanEval (n=164) | 86.6% | 85.4% | −1.2 pp [−3.7, +1.2] | 0.63 |
| Pooled (n=1,483) | 95.1% | 94.5% | −0.6 pp [−1.3, +0.1] | 0.11 |

Pooled, the 90% CI sits inside ±2 points, so the two paths are **statistically equivalent**
within ±2 pp (TOST), not merely "not significantly different." HumanEval alone is too small
to claim that: only 4 of 164 problems change correctness (3 favor the reference, 1 Together).

**Answers do move, a little.** On GSM8K the final number differs on 2.1% of problems, against
a 1.0% noise floor for the reference vs itself and 2.0% for Together vs itself. The flips
cancel out in accuracy (14 reference-only correct vs 7 Together-only).

**Together isn't deterministic at temperature 0.** Asked the same HumanEval prompt twice,
Together returned different code 25.0% of the time, vs 3.7% for the reference. Most of the
30.5% cross-endpoint code difference is Together disagreeing with itself, not FP8 changing
the model. Likely cause: batch-dependent kernels on a shared server. It's harmless for
accuracy here, but it matters to anyone who caches, diffs, or tests against model outputs.

## Result: speed — not 2–3× in this setup
Batch of one, sequential, measured from my laptop, prefix cache defeated with a unique first line.

| Prompt tokens | Decode tok/s (ref → Together) | Speedup (95% CI) | TTFT p50 (ref → Together) | TTFT p90 (ref → Together) |
|---|---|---|---|---|
| ~180 | 46.9 → 76.6 | 1.64× [1.53, 1.85] | 0.28 → 0.37 s | 0.35 → 1.00 s |
| ~1,070 | 46.7 → 56.9 | 1.22× [0.95, 1.37] | 0.38 → 0.63 s | 0.42 → 1.09 s |
| ~4,090 | 46.3 → 37.8 | 0.82× [0.64, 0.96] | 0.76 → 1.16 s | 0.81 → 1.64 s |
| ~5,920 | 46.2 → 61.0 | 1.32× [1.00, 1.64] | 0.98 → 1.23 s | 1.04 → 1.93 s |

Together decoded at **0.8–1.6×** the reference's speed, and its time to first token was
slower and much more variable (p90 roughly 2× the reference's). The reference was nearly
constant: 46–47 tok/s every request.

## Verdict
**No significant quality difference — equivalent within ±2 points — but not 2–3× faster than
stock vLLM on two dedicated H100s for a single user: 0.8–1.6× on decode, with slower and less
predictable first-token latency.**

That second half is not a refutation of Together's claim. The comparison is deliberately
unfair to Together in one direction: my reference was a private 2× H100 serving exactly one
request at a time, while Together is a shared service optimizing throughput and cost across
many users. Their claim is against hyperscaler endpoints under real load, which I didn't test.
What this does show is the quality side: FP8 Turbo costs ~0.5 points, within noise.

## What this does and doesn't show
- Compared with stock vLLM on dedicated H100s, not with AWS/Azure/GCP. The harness accepts any
  OpenAI-compatible endpoint, so the hyperscaler comparison is one config block away.
- Batch-of-one latency on one afternoon, not throughput under load.
- Two benchmarks. FP8 effects could show up on long-context, multilingual, or rare-token tasks.
- Cost: the Together side cost cents; the reference needed a 2× H100 box for about an hour [$ from Modal billing].

## Reproduce
`python run.py quality/speed --endpoint {reference,together}`, then `python run.py report`.
Raw per-prompt outputs are in `results/llama70b/`.
