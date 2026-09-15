#!/usr/bin/env python
"""Separate recognition quality from reading-order / segmentation quality.

Plain CER (see score.py) is a global sequence edit distance: a page whose words
are all correct but whose articles are in the wrong order is penalised almost as
hard as one that misread every word. To tell those apart we report, per model:

  word_acc  = 1 - WER            order-SENSITIVE word accuracy (moved word = 2 errs)
  bow_R     = bag-of-words recall     order-FREE; = coverage (share of gold words present)
  bow_P     = bag-of-words precision  order-FREE; share of emitted words that are real
  bow_F1    = harmonic mean of the two
  order_gap = bow_F1 - word_acc   how much of the word loss is pure (re)ordering

A large order_gap = "words are right, sequence is jumbled" (a layout/reading-order
problem). A low bow_R with small gap = "text was dropped" (a coverage problem).

Runs against the same completed pages across every model in ocr-results/.
Needs rapidfuzz (falls back to a slow pure-python edit distance).
"""
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "txt"
RESULTS = HERE / "ocr-results"

_tok = re.compile(r"[0-9a-z]+")
words = lambda s: _tok.findall(s.lower())

try:
    from rapidfuzz.distance import Levenshtein as _Lev
    _sdist = _Lev.distance
except ImportError:
    def _sdist(a, b):
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i] + [0] * len(b)
            for j, y in enumerate(b, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
            prev = cur
        return prev[-1]


def wer(ref_w, hyp_w):
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    # map each distinct word to one char so the C edit-distance can run on words
    vocab = {w: chr(i) for i, w in enumerate(set(ref_w) | set(hyp_w))}
    d = _sdist("".join(vocab[w] for w in ref_w), "".join(vocab[w] for w in hyp_w))
    return d / len(ref_w)


def bow(ref_w, hyp_w):
    cr, ch = Counter(ref_w), Counter(hyp_w)
    overlap = sum((cr & ch).values())
    P = overlap / len(hyp_w) if hyp_w else 0.0
    R = overlap / len(ref_w) if ref_w else 1.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F


def main():
    models = sorted(p for p in RESULTS.glob("*") if p.is_dir())
    if not models:
        print("No model folders in ocr-results/.")
        return
    # compare on pages every model has produced
    common = None
    for m in models:
        names = {p.name for p in m.glob("*.txt")}
        common = names if common is None else (common & names)
    common = sorted(common or [])
    print(f"Comparing on {len(common)} pages common to all {len(models)} models\n")
    print(f"{'model':22s} {'word_acc':>8s} {'bow_R':>6s} {'bow_P':>6s} "
          f"{'bow_F1':>6s} {'order_gap':>9s}")
    for m in models:
        wa = rr = pp = ff = 0.0
        for name in common:
            rw = words((REF / name).read_text(errors="ignore"))
            hw = words((m / name).read_text(errors="ignore"))
            wa += max(0.0, 1 - wer(rw, hw))
            P, R, F = bow(rw, hw)
            rr += R; pp += P; ff += F
        n = len(common) or 1
        wa, rr, pp, ff = wa / n, rr / n, pp / n, ff / n
        print(f"{m.name:22s} {wa:8.3f} {rr:6.3f} {pp:6.3f} {ff:6.3f} {ff - wa:9.3f}")


if __name__ == "__main__":
    main()
