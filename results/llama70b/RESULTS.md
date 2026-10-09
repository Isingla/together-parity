# Results

Reference (`reference`): `unsloth/Llama-3.3-70B-Instruct`  
Compared (`together`): `meta-llama/Llama-3.3-70B-Instruct-Turbo`

## Verdict (auto-generated — read the tables before quoting it)

- **gsm8k**: equivalent within ±2.0 pp: -0.5 pp (95% CI [-1.2 pp, +0.2 pp], McNemar p=0.189)
- **humaneval**: inconclusive: no significant difference but CI too wide to claim equivalence within ±2.0 pp (-1.2 pp, 95% CI [-3.7 pp, +1.2 pp])
- **pooled**: equivalent within ±2.0 pp: -0.6 pp (95% CI [-1.3 pp, +0.1 pp], McNemar p=0.108)
- **speed**: together decodes at 0.82–1.64x reference's tokens/sec across prompt lengths; TTFT speedup ranges 0.60–0.80x.

## Quality parity

Greedy decoding (temperature 0, seed 1234), identical prompts. Paired per-prompt comparison, 10,000 bootstrap resamples. Equivalence = 90% CI of Δ inside ±2.0 pp (TOST).

| Task | n | reference acc | together acc | Δ (pp) | 95% CI (pp) | McNemar p | ref-only / test-only | Verdict |
|---|---|---|---|---|---|---|---|---|
| gsm8k | 1319 | 96.2% | 95.7% | -0.5 | [-1.2, +0.2] | 0.189 | 14 / 7 | equivalent within ±2.0 pp |
| humaneval | 164 | 86.6% | 85.4% | -1.2 | [-3.7, +1.2] | 0.625 | 3 / 1 | inconclusive |
| **pooled** | 1483 | 95.1% | 94.5% | -0.6 | [-1.3, +0.1] | 0.108 | 17 / 8 | equivalent within ±2.0 pp |

### Do answers change more than run-to-run noise?

Flip rate = share of prompts whose answer differs (GSM8K: the extracted number; HumanEval: the exact code text, so formatting-only changes count). *Self-noise* = the same endpoint run twice on identical prompts at temperature 0. The excess is measured against the **larger** of the two self-noise rates: if the endpoints disagree with each other no more than one of them disagrees with itself, the difference is indistinguishable from serving nondeterminism.

| Task | reference vs together | reference self-noise | together self-noise | Excess over larger self-noise (pp) | 95% CI | Byte-identical outputs |
|---|---|---|---|---|---|---|
| gsm8k | 2.1% | 1.0% | 2.0% | +0.2 | [-0.5, +0.8] | 11.6% |
| humaneval | 30.5% | 3.7% | 25.0% | +5.5 | [-1.8, +12.8] | 69.5% |

## Speed (batch of one, sequential, prefix cache defeated)

TTFT is measured from the client, so every endpoint's TTFT includes the network round trip to its host. Decode tok/s excludes the first token, so it is mostly network-independent.

| Prompt tok | reference TTFT p50 / p90 (s) | together TTFT p50 / p90 (s) | TTFT speedup (95% CI) | reference decode tok/s | together decode tok/s | Decode speedup (95% CI) |
|---|---|---|---|---|---|---|
| 182 | 0.284 / 0.346 | 0.365 / 0.996 | 0.78x [0.40, 1.00] | 46.9 | 76.6 | 1.64x [1.53, 1.85] |
| 1072 | 0.376 / 0.421 | 0.630 / 1.086 | 0.60x [0.42, 0.82] | 46.7 | 56.9 | 1.22x [0.95, 1.37] |
| 4091 | 0.760 / 0.813 | 1.163 / 1.637 | 0.65x [0.56, 0.86] | 46.3 | 37.8 | 0.82x [0.64, 0.96] |
| 5917 | 0.984 / 1.036 | 1.229 / 1.926 | 0.80x [0.58, 0.90] | 46.2 | 61.0 | 1.32x [1.00, 1.64] |

![speed](speed.png)
