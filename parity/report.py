"""Builds results/<profile>/RESULTS.md (+ speed.png) from raw JSONL."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from parity.stats import flip_rate, paired_accuracy, ratio_of_medians

BLUE, ORANGE = "#2a78d6", "#eb6834"     # validated categorical slots 1-2


def _load(p: Path) -> dict[str, dict]:
    if not p.exists():
        return {}
    rows = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                rows[r["id"]] = r            # last write wins (retries)
    return rows


def _pct(x):
    return f"{100 * x:.1f}%"


def _pp(x):
    return f"{100 * x:+.1f}"


def quality_section(cfg, root: Path, REF: str, TEST: str) -> tuple[list[str], list[str]]:
    q, b = cfg["quality"], cfg["bootstrap"]
    margin = q["equivalence_margin_pp"] / 100
    lines = ["## Quality parity", "",
             f"Greedy decoding (temperature 0, seed {q['seed']}), identical prompts. "
             f"Paired per-prompt comparison, {b['n_resamples']:,} bootstrap resamples. "
             f"Equivalence = 90% CI of Δ inside ±{q['equivalence_margin_pp']} pp (TOST).", "",
             f"| Task | n | {REF} acc | {TEST} acc | Δ (pp) | 95% CI (pp) | McNemar p | ref-only / test-only | Verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    flips = ["", "### Do answers change more than run-to-run noise?", "",
             "Flip rate = share of prompts whose extracted answer differs. "
             "*Noise floor* = reference run 1 vs reference run 2 (same server, same weights). "
             "If the optimized path only changes answers as often as the reference "
             "disagrees with itself, the change is indistinguishable from serving noise.", "",
             f"| Task | {REF} vs {TEST} | {REF} vs {REF} (noise) | {TEST} vs {TEST} | Excess flips (pp) | 95% CI | Byte-identical outputs |",
             "|---|---|---|---|---|---|---|"]
    verdicts = []
    all_ref, all_test = [], []
    for task in ("gsm8k", "humaneval"):
        d = {(e, r): _load(root / "quality" / e / f"run{r}" / f"{task}.jsonl")
             for e in (REF, TEST) for r in (1, 2)}
        ids = sorted(set(d[(REF, 1)]) & set(d[(TEST, 1)]))
        if not ids:
            continue
        ref = np.array([d[(REF, 1)][i]["correct"] for i in ids], float)
        test = np.array([d[(TEST, 1)][i]["correct"] for i in ids], float)
        all_ref.append(ref)
        all_test.append(test)
        pa = paired_accuracy(ref, test, margin=margin, n_resamples=b["n_resamples"], seed=b["seed"])
        verdicts.append((task, pa))
        lines.append(f"| {task} | {pa.n} | {_pct(pa.acc_ref)} | {_pct(pa.acc_test)} | {_pp(pa.delta)} | "
                     f"[{_pp(pa.ci95[0])}, {_pp(pa.ci95[1])}] | {pa.mcnemar_p:.3g} | "
                     f"{pa.ref_only} / {pa.test_only} | {pa.verdict().split(':')[0]} |")
        ids3 = [i for i in ids if i in d[(REF, 2)]]
        if ids3:
            fr = flip_rate([d[(REF, 1)][i]["pred"] for i in ids3], [d[(REF, 2)][i]["pred"] for i in ids3],
                           [d[(TEST, 1)][i]["pred"] for i in ids3],
                           n_resamples=b["n_resamples"], seed=b["seed"])
            ids4 = [i for i in ids if i in d[(TEST, 2)]]
            tt = (np.mean([d[(TEST, 1)][i]["pred"] != d[(TEST, 2)][i]["pred"] for i in ids4])
                  if ids4 else float("nan"))
            ident = np.mean([d[(REF, 1)][i]["text"] == d[(TEST, 1)][i]["text"] for i in ids])
            flips.append(f"| {task} | {_pct(fr.cross)} | {_pct(fr.noise)} | "
                         f"{_pct(tt) if ids4 else 'n/a'} | {_pp(fr.excess)} | "
                         f"[{_pp(fr.ci95[0])}, {_pp(fr.ci95[1])}] | {_pct(ident)} |")
    if len(all_ref) > 1:
        pa = paired_accuracy(np.concatenate(all_ref), np.concatenate(all_test), margin=margin,
                             n_resamples=b["n_resamples"], seed=b["seed"])
        verdicts.append(("pooled", pa))
        lines.append(f"| **pooled** | {pa.n} | {_pct(pa.acc_ref)} | {_pct(pa.acc_test)} | {_pp(pa.delta)} | "
                     f"[{_pp(pa.ci95[0])}, {_pp(pa.ci95[1])}] | {pa.mcnemar_p:.3g} | "
                     f"{pa.ref_only} / {pa.test_only} | {pa.verdict().split(':')[0]} |")
    if len(flips) == 6:
        flips.append("| _(needs reference run 2)_ | | | | | | |")
    return lines + flips, [f"- **{t}**: {pa.verdict()}" for t, pa in verdicts]


def speed_section(cfg, root: Path, REF: str, TEST: str) -> tuple[list[str], list[str]]:
    b = cfg["bootstrap"]
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8")) if (root / "meta.json").exists() else {}
    labels = {e: e for e in (REF, TEST)}
    labels.update({r["endpoint"]: f'{r["endpoint"]} ({r["model"].split("/")[-1]})' for r in meta.get("runs", [])})
    rows = {e: [json.loads(l) for l in (root / "speed" / f"{e}.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()] if (root / "speed" / f"{e}.jsonl").exists() else []
            for e in (REF, TEST)}
    if not rows[REF] or not rows[TEST]:
        return ["## Speed", "", "_(run `speed` on both endpoints)_"], []
    lines = ["## Speed (batch of one, sequential, prefix cache defeated)", "",
             "TTFT is measured from the client, so every endpoint's TTFT includes the network "
             "round trip to its host. Decode tok/s excludes the first "
             "token, so it is mostly network-independent.", "",
             f"| Prompt tok | {REF} TTFT p50 / p90 (s) | {TEST} TTFT p50 / p90 (s) | TTFT speedup (95% CI) | "
             f"{REF} decode tok/s | {TEST} decode tok/s | Decode speedup (95% CI) |",
             "|---|---|---|---|---|---|---|"]
    summary, plot = [], {e: ([], [], []) for e in (REF, TEST)}
    for target in cfg["speed"]["prompt_tokens"]:
        g = {e: [r for r in rows[e] if r["target_prompt_tokens"] == target and not r.get("error")
                 and r["ttft_s"]] for e in rows}
        if not g[REF] or not g[TEST]:
            continue
        tt = {e: np.array([r["ttft_s"] for r in g[e]]) for e in g}
        ds = {e: np.array([r["decode_tok_s"] for r in g[e] if r["decode_tok_s"]]) for e in g}
        ptok = int(np.median([r["prompt_tokens"] or target for r in g[REF]]))
        s_tt, ci_tt = ratio_of_medians(tt[REF], tt[TEST], n_resamples=b["n_resamples"], seed=b["seed"])
        s_ds, ci_ds = ratio_of_medians(ds[TEST], ds[REF], n_resamples=b["n_resamples"], seed=b["seed"])
        lines.append(f"| {ptok} | {np.median(tt[REF]):.3f} / {np.percentile(tt[REF], 90):.3f} | "
                     f"{np.median(tt[TEST]):.3f} / {np.percentile(tt[TEST], 90):.3f} | "
                     f"{s_tt:.2f}x [{ci_tt[0]:.2f}, {ci_tt[1]:.2f}] | {np.median(ds[REF]):.1f} | "
                     f"{np.median(ds[TEST]):.1f} | {s_ds:.2f}x [{ci_ds[0]:.2f}, {ci_ds[1]:.2f}] |")
        summary.append((ptok, s_tt, s_ds))
        for e in (REF, TEST):
            plot[e][0].append(ptok)
            plot[e][1].append(np.median(tt[e]))
            plot[e][2].append(np.median(ds[e]))
    _plot(plot, root / "speed.png", REF, TEST, labels)
    lines += ["", "![speed](speed.png)"]
    if not summary:
        return lines, []
    lo = min(s for _, _, s in summary)
    hi = max(s for _, _, s in summary)
    return lines, [f"- **speed**: {TEST} decodes at {lo:.2f}–{hi:.2f}x {REF}'s tokens/sec "
                   f"across prompt lengths; TTFT speedup ranges "
                   f"{min(s for _, s, _ in summary):.2f}–{max(s for _, s, _ in summary):.2f}x."]


def _plot(plot, path: Path, REF: str, TEST: str, labels: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#fcfcfb")
    style = {REF: (BLUE, "o", labels[REF]), TEST: (ORANGE, "s", labels[TEST])}
    for ax, k, title in ((axes[0], 1, "Time to first token (s, median)"),
                         (axes[1], 2, "Decode tokens/sec (median)")):
        ax.set_facecolor("#fcfcfb")
        for e, (c, m, lab) in style.items():
            x, y = plot[e][0], plot[e][k]
            ax.plot(x, y, color=c, marker=m, markersize=7, linewidth=2, label=lab)
            if x:
                ax.annotate(f"{y[-1]:.2f}" if k == 1 else f"{y[-1]:.0f}", (x[-1], y[-1]),
                            textcoords="offset points", xytext=(6, 0), va="center",
                            fontsize=9, color="#444")
        ax.set_title(title, fontsize=11, loc="left", color="#222")
        ax.set_xlabel("prompt tokens", color="#555")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e5e5e2", linewidth=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#bbb")
        ax.tick_params(colors="#555")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report(cfg, root: Path, REF: str = "reference", TEST: str = "together") -> str:
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8")) if (root / "meta.json").exists() else {}
    models = {r["endpoint"]: r["model"] for r in meta.get("runs", [])}
    q_lines, q_v = quality_section(cfg, root, REF, TEST)
    s_lines, s_v = speed_section(cfg, root, REF, TEST)
    md = ["# Results", "",
          f"Reference (`{REF}`): `{models.get(REF, '?')}`  ",
          f"Compared (`{TEST}`): `{models.get(TEST, '?')}`", "",
          "## Verdict (auto-generated — read the tables before quoting it)", "",
          *(q_v + s_v), "", *q_lines, "", *s_lines, ""]
    out = root / ("RESULTS.md" if (REF, TEST) == ("reference", "together") else f"RESULTS_{REF}_vs_{TEST}.md")
    out.write_text("\n".join(md), encoding="utf-8")
    return "\n".join(md)
