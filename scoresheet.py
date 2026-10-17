#!/usr/bin/env python
"""Combined score sheet: OCR quality + cost + speed, one row per ocr-results/ folder.

Merges score.py's recognition/reading-order metrics with the per-run `_cost.json`
that run_openrouter_bench.py writes (real OpenRouter spend, tokens, wall-clock), so
LLM backends can be compared on accuracy AND price at once.

    python scoresheet.py            # markdown table to stdout
    python scoresheet.py --csv      # also write scoresheet.csv + scoresheet.md

Quality columns mirror score.py (order-sensitive `overall`=1-CER, order-robust
`chrF`/`bowF1`, `gap`=bowF1-overall). `cased`=1-CER keeping case AND punctuation,
which `overall` drops — the honest gap for engines that get case/punctuation right.
Cost columns are blank for local baselines
with no `_cost.json`. Scores are over the non-excluded pages (see newsbench.csv).
"""
import argparse, csv as csvmod, json, re
from pathlib import Path
from score import (cer, cer_cased, chrf, bow_f1, word_error_kinds, load_text,
                   load_tiers, load_excludes, REF, RESULTS)


def config_of(name: str) -> str:
    if name.endswith("-wholepage"):
        return "wholepage"
    if name.endswith("-paddlex"):
        return "paddlex-region"
    return "local/other"


def harness_of(name: str) -> str:
    """Which OCR pipeline produced the folder."""
    if name.startswith("progmag"):
        return "progmag-ocr"          # external progressive-magazines-ocr
    if name.startswith("ensemble"):
        return "MinerU+npo"           # MinerU base + newspaper-ocr hole patching
    return "newspaper-ocr"


# folders whose detector is AS-YOLO rather than PP-DocLayout
AS_YOLO = {"tesseract-default", "news_combo_fast", "glm-ocr-mlx"}


def region_of(name: str) -> str:
    """Detector / segmentation used: none (full page), PaddleX, DocLayout, or AS-YOLO."""
    if name.startswith("ensemble"):
        return "Ensemble"             # MinerU layout ∪ DocLayout holes
    if name.endswith("-wholepage") or "wholepage" in name:
        return "none"
    if "doclayout" in name:
        return "DocLayout"
    if name in AS_YOLO or "asyolo" in name or "as_yolo" in name:
        return "AS-YOLO"
    return "PaddleX"                   # -paddlex runs + paddlex-*/lib-*/progmag-ppdoc baselines


def display_name(name: str) -> str:
    """Strip the config suffix; the Region column now carries that info."""
    for suf in ("-wholepage", "-paddlex", "-doclayout"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def score_folder(folder: Path, tiers: dict, excludes: set) -> dict | None:
    allc, fs, bows, casedc = [], [], [], []
    gw = mw = dw = nw = 0
    by = {}
    for ref_file in sorted(REF.glob("*.txt")):
        if ref_file.stem in excludes:
            continue
        hyp = folder / ref_file.name
        if not hyp.exists():
            continue
        r = load_text(ref_file)
        h = load_text(hyp)
        c = cer(r, h)
        allc.append(c)
        casedc.append(cer_cased(r, h))
        by.setdefault(tiers.get(ref_file.stem, "all"), []).append(c)
        fs.append(chrf(r, h))
        bows.append(bow_f1(r, h))
        g, m, d, n = word_error_kinds(r, h)
        gw += g; mw += m; dw += d; nw += n
    if not allc:
        return None
    acc = lambda xs: 1 - sum(xs) / len(xs)
    mean = lambda xs: sum(xs) / len(xs)
    ov, bf = acc(allc), mean(bows)
    row = {
        "model": display_name(folder.name),
        "harness": harness_of(folder.name),
        "region": region_of(folder.name),
        "config": config_of(folder.name),
        "n": len(allc),
        "overall": ov,
        "cased": acc(casedc),
        "broadsheet": acc(by["broadsheet"]) if by.get("broadsheet") else None,
        "page": acc(by["page"]) if by.get("page") else None,
        "chrF": mean(fs),
        "bowF1": bf,
        "gap": bf - ov,
        # Order-free word-multiset error kinds, as a fraction of gold words.
        "missing": mw / gw if gw else None,
        "duplicate": dw / gw if gw else None,
        "novel": nw / gw if gw else None,
    }
    # Local models have no API cost — show $0 rather than blank. Hosted runs
    # overwrite these from their _cost.json.
    row["cost"] = 0.0
    row["cost_per_100"] = 0.0
    cost_file = folder / "_cost.json"
    if cost_file.exists():
        c = json.loads(cost_file.read_text())
        pages = c.get("pages") or len(allc)
        row["cost"] = c.get("reported_cost")
        row["cost_per_100"] = (c["reported_cost"] / pages * 100) if pages else None
        row["tok_out"] = int(c.get("completion_tokens") or 0)
        row["sec_per_page"] = (c["elapsed_s"] / pages) if c.get("elapsed_s") and pages else None
    return row


def fmt(v, spec):
    return "" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true", help="also write scoresheet.csv + scoresheet.md")
    args = ap.parse_args()

    tiers, excludes = load_tiers(), load_excludes()
    rows = [r for f in sorted(RESULTS.glob("*"))
            if f.is_dir() and (r := score_folder(f, tiers, excludes))]
    rows.sort(key=lambda r: r["overall"], reverse=True)

    cols = ["model", "harness", "region", "overall", "cased", "chrF", "bowF1", "gap",
            "missing", "duplicate", "novel", "cost_per_100", "n"]
    hdr = ["model", "harness", "region", "overall", "cased", "chrF", "bowF1", "gap",
           "miss", "dup", "novel", "$/100pg", "n"]
    specs = {"overall": ".3f", "cased": ".3f", "broadsheet": ".3f", "page": ".3f", "chrF": ".3f",
             "bowF1": ".3f", "gap": ".3f", "missing": ".1%", "duplicate": ".1%", "novel": ".1%",
             "cost": ".4f", "cost_per_100": ".2f", "tok_out": "d", "n": "d"}

    def render_row(r):
        out = []
        for c in cols:
            if c in ("model", "harness", "region"):
                out.append(str(r[c]))
            else:
                out.append(fmt(r.get(c), specs[c]))
        return out

    md = ["| " + " | ".join(hdr) + " |",
          "|" + "|".join(["---"] * len(hdr)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(render_row(r)) + " |")
    table = "\n".join(md)
    print(table)

    if args.csv:
        Path("scoresheet.md").write_text("# NewsBench score sheet\n\n" + table + "\n")
        with open("scoresheet.csv", "w", newline="") as fh:
            w = csvmod.writer(fh)
            w.writerow(cols)
            for r in rows:
                w.writerow([r.get(c) for c in cols])
        print("\nwrote scoresheet.md + scoresheet.csv")


if __name__ == "__main__":
    main()
