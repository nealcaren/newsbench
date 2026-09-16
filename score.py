#!/usr/bin/env python
"""Score newsbench/ocr-results/<model>/ against the gold transcriptions.

Three complementary numbers, because on multi-column newspapers a page can be
read perfectly yet scored badly for putting the articles in the wrong order:

  overall  = 1 - CER    order-SENSITIVE. Global char edit distance on lowercased,
                        alphanumeric-only text. Mirrors InkBench's headline metric;
                        a jumbled-but-correct page is penalised hard here.
  chrF     = character n-gram F-score (n=1..6, beta=2). Local n-grams are immune to
                        article/paragraph reordering but still character-sensitive.
  bowF1    = bag-of-words F1. Treats each page as a multiset of word tokens, so order
                        is ignored entirely. This is the "pure recognition" number,
                        robust to how each engine chunks the page.

Read `gap = bowF1 - overall` as the reading-order / segmentation penalty: a large gap
means the words are mostly right but the sequence is wrong.

A stricter, character-level order-removed score (block-aligned CER) is available with
`--aligned`; it greedily matches each gold paragraph to its best-fitting span in the
hypothesis regardless of position, so it tolerates reordering AND text reflow.

    python newsbench/score.py [--aligned]

Uses rapidfuzz + scipy when available; falls back to slower pure-python otherwise.
"""
import csv, re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "txt"
RESULTS = HERE / "ocr-results"
MANIFEST = HERE / "newsbench.csv"

# alnum-only, lowercase — for CER (matches InkBench)
norm = lambda s: re.sub(r"[^0-9a-z]", "", s.lower())
# lowercase, alnum + single spaces — for chrF (keeps word boundaries)
norm_sp = lambda s: re.sub(r"\s+", " ", re.sub(r"[^0-9a-z ]", " ", s.lower())).strip()

try:
    from rapidfuzz.distance import Levenshtein as _Lev
    _dist = _Lev.distance
except ImportError:  # slow pure-python fallback
    def _dist(r, h):
        prev = list(range(len(h) + 1))
        for i, rc in enumerate(r, 1):
            cur = [i] + [0] * len(h)
            for j, hc in enumerate(h, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
            prev = cur
        return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return _dist(r, h) / len(r)


# ---- chrF (order-robust, character n-gram F-score) --------------------------
def _ngrams(s, n):
    return Counter(s[i:i + n] for i in range(len(s) - n + 1)) if len(s) >= n else Counter()


def chrf(ref: str, hyp: str, nmax: int = 6, beta: float = 2.0) -> float:
    r, h = norm_sp(ref), norm_sp(hyp)
    if not r and not h:
        return 1.0
    ps, rs = [], []
    for n in range(1, nmax + 1):
        cr, ch = _ngrams(r, n), _ngrams(h, n)
        match = sum((cr & ch).values())
        tot_h, tot_r = sum(ch.values()), sum(cr.values())
        if tot_h:
            ps.append(match / tot_h)
        if tot_r:
            rs.append(match / tot_r)
    if not ps or not rs:
        return 0.0
    P, R = sum(ps) / len(ps), sum(rs) / len(rs)
    b2 = beta * beta
    return (1 + b2) * P * R / (b2 * P + R) if (b2 * P + R) else 0.0


# ---- bag-of-words F1 (order-free recognition) -------------------------------
_tok = re.compile(r"[0-9a-z]+")


def bow_f1(ref: str, hyp: str) -> float:
    rw, hw = _tok.findall(ref.lower()), _tok.findall(hyp.lower())
    if not rw:
        return 1.0 if not hw else 0.0
    overlap = sum((Counter(rw) & Counter(hw)).values())
    P = overlap / len(hw) if hw else 0.0
    R = overlap / len(rw)
    return 2 * P * R / (P + R) if (P + R) else 0.0


# ---- block-aligned CER (opt-in; character-level, order + reflow tolerant) ----
def aligned_cer(ref: str, hyp: str) -> float:
    """For each gold paragraph, greedily claim its best-matching span anywhere in
    the (order-free) hypothesis and charge the edit distance of that match; any
    hypothesis text left unclaimed is charged as insertions. Granularity-robust
    because the hypothesis is never pre-chunked."""
    from rapidfuzz.fuzz import partial_ratio_alignment
    paras = [norm_sp(p) for p in re.split(r"\n\s*\n", ref)]
    paras = [p for p in paras if p] or [norm_sp(ref)]
    total_ref = sum(len(p) for p in paras)
    if total_ref == 0:
        return 0.0
    remaining = norm_sp(hyp)
    cost = 0.0
    for p in sorted(paras, key=len, reverse=True):
        if not remaining:
            cost += len(p)
            continue
        a = partial_ratio_alignment(p, remaining)
        span = remaining[a.dest_start:a.dest_end]
        cost += _dist(p, span)
        remaining = remaining[:a.dest_start] + remaining[a.dest_end:]  # consume it
    cost += len(remaining)  # unclaimed hypothesis text = insertions
    return cost / total_ref


def load_tiers():
    tiers = {}
    if MANIFEST.exists():
        for row in csv.DictReader(open(MANIFEST)):
            tiers[Path(row["image_name"]).stem] = row["tier"]
    return tiers


def load_excludes():
    """Stems flagged `exclude` in the manifest (e.g. partial/faulty gold) — skipped
    for every model so the comparison stays fair."""
    ex = set()
    if MANIFEST.exists():
        for row in csv.DictReader(open(MANIFEST)):
            if (row.get("exclude") or "").strip():
                ex.add(Path(row["image_name"]).stem)
    return ex


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", action="store_true",
                    help="add block-aligned CER column (slower, char-level order-removed)")
    args = ap.parse_args()

    tiers = load_tiers()
    excludes = load_excludes()
    models = sorted(p for p in RESULTS.glob("*") if p.is_dir()) if RESULTS.exists() else []
    if not models:
        print("No model folders in newsbench/ocr-results/. Run run_newspaper_ocr.py first.")
        return
    tier_names = sorted(set(tiers.values())) or ["all"]
    extra = f" {'aln':>6s}" if args.aligned else ""
    hdr = f"{'model':26s} {'overall':>8s} " + " ".join(f"{t:>11s}" for t in tier_names)
    print(hdr + f" {'chrF':>6s} {'bowF1':>6s} {'gap':>6s}{extra}  n")
    for m in models:
        by = {t: [] for t in tier_names}
        allc, fs, bows, alns = [], [], [], []
        for ref_file in sorted(REF.glob("*.txt")):
            if ref_file.stem in excludes:
                continue
            hyp_file = m / ref_file.name
            if not hyp_file.exists():
                continue
            rtext = ref_file.read_text(encoding="utf-8", errors="ignore")
            htext = hyp_file.read_text(encoding="utf-8", errors="ignore")
            allc.append(cer(rtext, htext))
            t = tiers.get(ref_file.stem, tier_names[0])
            by.setdefault(t, []).append(allc[-1])
            fs.append(chrf(rtext, htext))
            bows.append(bow_f1(rtext, htext))
            if args.aligned:
                alns.append(1 - aligned_cer(rtext, htext))
        acc = lambda xs: (1 - sum(xs) / len(xs)) if xs else float("nan")
        mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
        ov, bf = acc(allc), mean(bows)
        tiercols = " ".join(f"{acc(by[t]):11.3f}" for t in tier_names)
        tail = f" {mean(alns):6.3f}" if args.aligned else ""
        print(f"{m.name:26s} {ov:8.3f} {tiercols} {mean(fs):6.3f} {bf:6.3f} "
              f"{bf - ov:6.3f}{tail}  {len(allc)}")


if __name__ == "__main__":
    main()
