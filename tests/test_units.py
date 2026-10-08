import numpy as np
import pytest

from parity.stats import flip_rate, paired_accuracy, ratio_of_medians
from parity.tasks import (build_program, extract_gsm8k, grade_humaneval, load_gsm8k,
                          load_humaneval, run_program)


@pytest.mark.parametrize("text,want", [
    ("... The answer is: 1,234", "1234"),
    ("The answer is **$18**.", "18"),
    ("so 3.50 dollars. The answer is: 3.5", "3.5"),
    ("x = 7, then 12", "12"),
    ("The answer is: -4.0", "-4"),
    ("no numbers here", None),
])
def test_gsm8k_extract(text, want):
    assert extract_gsm8k(text) == want


def test_gsm8k_gold_loaded():
    items = load_gsm8k(10_000, 0)
    assert len(items) == 1319
    assert items[0].gold["answer"] == "18"


def test_subset_is_deterministic():
    a = [i.id for i in load_gsm8k(50, 1234)]
    assert a == [i.id for i in load_gsm8k(50, 1234)] and len(set(a)) == 50


def test_humaneval_canonical_all_pass():
    import gzip, json
    from parity.tasks import DATA
    canon = {}
    with gzip.open(DATA / "HumanEval.jsonl.gz", "rt", encoding="utf-8") as f:
        for l in f:
            ex = json.loads(l)
            canon[ex["task_id"]] = ex["prompt"] + ex["canonical_solution"]
    items = load_humaneval(164, 0)
    fails = [it.id for it in items
             if not grade_humaneval(it, f"```python\n{canon[it.id]}```")["correct"]]
    assert fails == []


def test_humaneval_wrong_and_body_only():
    it = load_humaneval(164, 0)[0]
    assert not grade_humaneval(it, "```python\ndef has_close_elements(numbers, threshold):\n    return 'x'\n```")["correct"]
    import gzip, json
    from parity.tasks import DATA
    with gzip.open(DATA / "HumanEval.jsonl.gz", "rt", encoding="utf-8") as f:
        body = json.loads(f.readline())["canonical_solution"]
    assert grade_humaneval(it, f"```python\n{body}```")["correct"]   # body-only reply


def test_timeout():
    ok, err = run_program("while True: pass", timeout=1)
    assert not ok and err == "timeout"


def test_paired_identical_is_equivalent():
    x = np.array([1, 0] * 500)
    pa = paired_accuracy(x, x, margin=0.02, n_resamples=2000)
    assert pa.delta == 0 and pa.equivalent and not pa.different and pa.mcnemar_p == 1.0


def test_paired_detects_large_drop():
    rng = np.random.default_rng(0)
    ref = (rng.random(1000) < 0.85).astype(float)
    test = ref.copy()
    test[np.where(ref == 1)[0][:80]] = 0          # -8 pp
    pa = paired_accuracy(ref, test, margin=0.02, n_resamples=2000)
    assert pa.different and not pa.equivalent and pa.delta == pytest.approx(-0.08)
    assert "significant" in pa.verdict()


def test_paired_small_n_inconclusive():
    ref = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1], float)
    test = np.array([1, 0, 0, 1, 1, 1, 1, 0, 1, 1], float)
    assert "inconclusive" in paired_accuracy(ref, test, margin=0.02, n_resamples=2000).verdict()


def test_flip_rate():
    r1 = list(range(100)); r2 = list(range(100)); t = list(range(100))
    for i in range(10):
        t[i] = -1
    fr = flip_rate(r1, r2, t, n_resamples=2000)
    assert fr.cross == pytest.approx(0.10) and fr.noise == 0 and fr.significant


def test_ratio_of_medians():
    r, (lo, hi) = ratio_of_medians(np.full(20, 2.0), np.full(20, 1.0), n_resamples=500)
    assert r == 2.0 and lo == hi == 2.0


def test_zero_discordance_small_n_not_equivalent():
    x = np.ones(40)
    assert not paired_accuracy(x, x, margin=0.02, n_resamples=500).equivalent
    y = np.ones(200)
    assert paired_accuracy(y, y, margin=0.02, n_resamples=500).equivalent
