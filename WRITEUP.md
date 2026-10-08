# Does Together's FP8 "Turbo" serving change Llama 3.3 70B's answers?

*Ishaan Singla · Oct 2026 · [repo](https://github.com/<you>/together-parity)*

<!-- Fill every [bracket] from results/llama70b/RESULTS.md. Keep it to one page.
     If the numbers say the opposite of what you hoped, write that. -->

## Question
Together AI says its serving stack is much faster while keeping output quality. Its
`Llama-3.3-70B-Instruct-Turbo` endpoint is listed as FP8-quantized. I ran the same model
through Together and through Meta's BF16 weights on stock vLLM, on identical prompts, and
asked two things: does the optimized path change answers, and how much faster is it?

## Setup
- Reference: `meta-llama/Llama-3.3-70B-Instruct`, BF16, vLLM [version], 2x H100 (Modal).
- Together: `meta-llama/Llama-3.3-70B-Instruct-Turbo` via API, [date].
- Tasks: GSM8K test (1,319) and HumanEval (164). Greedy, seed 1234, same prompt templates.
- Each endpoint run twice to measure its own run-to-run noise.

## Result: quality
| | Reference | Together | Δ (95% CI) | McNemar p |
|---|---|---|---|---|
| GSM8K | [ ]% | [ ]% | [ ] pp [ , ] | [ ] |
| HumanEval | [ ]% | [ ]% | [ ] pp [ , ] | [ ] |

Answers flipped on [ ]% of prompts between Together and the reference, versus [ ]% between
two identical reference runs — an excess of [ ] pp (95% CI [ , ]).

## Result: speed
At ~1k prompt tokens Together reached [ ] decode tok/s vs [ ] for the reference
([ ]x, 95% CI [ , ]); median TTFT [ ] s vs [ ] s. Chart: `results/llama70b/speed.png`.

## Verdict
[One sentence, e.g. "No significant quality difference — equivalent within ±2 pp — at [X]x
the decode speed of stock vLLM." OR "Together's Turbo path lowered GSM8K accuracy by [X] pp
(95% CI [ , ]), beyond run-to-run noise, while decoding [X]x faster."]

## What this does and doesn't show
- Compared with stock vLLM on dedicated H100s, not with hyperscalers; the "vs hyperscalers"
  number needs a hyperscaler endpoint (the harness supports one).
- Batch-of-one latency, not throughput under load. Two benchmarks, not every workload.
- [Anything surprising you saw in the raw outputs — e.g. where flipped answers cluster.]

## Reproduce
`python run.py quality/speed --endpoint {reference,together}` then `python run.py report`.
Total cost: [$ ].
