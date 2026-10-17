# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NewsBench is a benchmark for transcribing dense, multi-column **full** historical
newspaper pages. Its distinguishing goal is measuring **layout understanding / reading
order**, not just character recognition — the metrics exist to separate "misread the
words" from "read the words but stitched the columns in the wrong order." Corpus: 19
complete LoC *By the People* pages, each shipping a volunteer gold transcription
(`images/<id>.jpg` ↔ `txt/<id>.txt`, matched by stem). Read `README.md` and `MEMO.md`
before making corpus or scoring changes — they document provenance caveats and known bugs.

## Commands

```bash
pip install -r requirements.txt   # rapidfuzz (fast), scipy (only for --aligned)

python score.py            # main table (auto-discovers every ocr-results/<model>/ folder)
python score.py --aligned  # + block-aligned CER (slower, char-level, order-removed)
python score_order.py      # word-level recall/precision + ordering-gap breakdown
```

Scoring is stdlib-only (rapidfuzz/scipy just make it faster/richer). The `run_*ocr.py`
scripts instead drive the external [`newspaper-ocr`](https://github.com/nealcaren/newspaper-ocr)
pipeline and **must be run with that project's interpreter**, not this repo's — e.g.
`/path/to/newspaper-ocr/.venv/bin/python run_newspaper_ocr.py --name <model>`.

`run_newspaper_ocr.py` now tracks the upstream library (which as of 2026 has moved past the
pip release): the detector defaults to `auto` (= PP-DocLayout/`paddlex` when installed,
the recommended detector for dense pages), and the runner exposes the library's recovery
ladder — `--fallback <recognizer>` (backup region OCR for failed regions), `--chunk-tall-regions`,
and `--region-repair` (the post-recognition `RegionRepair` pass: lossless dedup +
container-split with strip re-OCR + fragmented-ad merge; needs a region-level recognizer
like `glm-ocr`). Common runs:

```bash
NPO=/path/to/newspaper-ocr/.venv/bin/python
$NPO run_newspaper_ocr.py --name paddlex-tesseract                          # library default
$NPO run_newspaper_ocr.py --name paddlex-glm --recognizer glm-ocr           # PP-DocLayout + GLM-OCR VLM
$NPO run_newspaper_ocr.py --name paddlex-glm-repair --recognizer glm-ocr \
    --region-repair --chunk-tall-regions --fallback paddleocr-vl            # + recovery ladder
```

GLM-OCR runs in `api` mode against a local Apple-Silicon MLX server:
`python -m mlx_vlm.server --model mlx-community/GLM-OCR-bf16 --port 8080`. The server can
crash under sustained batch load — the library now has per-region timeout handling, but for
long batch runs prefer page-by-page with a health-check/restart (see `run_progmag_robust.py`).

The `ocr-results/progmag-*` folders came from an *external* pipeline (progressive-magazines-ocr)
that was used before upstream `newspaper-ocr` had PP-DocLayout + region-repair; those two
capabilities are now native, so that detour is reproducible through `run_newspaper_ocr.py`
itself (the `paddlex-glm-repair` invocation above).

## How scoring works (the core architecture)

`score.py` compares each `ocr-results/<model>/<id>.txt` against `txt/<id>.txt` and reports
**three complementary numbers** because plain CER conflates two different failures:

- `overall` = 1 − CER — global char edit distance, **order-sensitive** (the headline metric).
- `chrF` — character n-gram F-score, robust to reordering.
- `bowF1` — bag-of-words F1, **order-free** ("pure recognition").
- `gap = bowF1 − overall` — the reading-order / segmentation penalty. A large gap means
  the words are right but the sequence is wrong.

Text is normalized to lowercase alphanumerics before comparison (see the `norm` /
`norm_sp` lambdas). Tiers (`broadsheet` vs `page`) come from `newsbench.csv`.

## To score a new system

Write one plain-text file per page to `ocr-results/<your-model-name>/<id>.txt` (matching
the image stem), then run `python score.py`. Folders are auto-discovered — no registration.

## newsbench.csv is the source of truth for the corpus

`newsbench.csv` (columns: `image_name, collection, words, tier, exclude, newspaper_note`)
is the manifest. The `exclude` column is load-bearing: any row with a non-empty `exclude`
is skipped by `score.py` for **every** model (via `load_excludes()`), so the comparison
stays fair. Excluded pages are kept in `images/`+`txt/` for provenance. Four pages are
excluded as `partial_gold` (volunteer transcribed only part of the page), so scores are
over **15** pages. The reusable screen for partial-gold pages: compare a page's gold word
count against the independent OCR consensus — a page where every backend emits far more
coherent text than the gold is a suspect (see MEMO.md §1).

## The gold is corrected — edit it carefully

The scored pages' volunteer text was corrected against the scans (269 edits, Oct 2026).
Every edit is in `docs/gold-errata.csv`; every candidate checked, including the ones where
the gold was right, is in `docs/gold-audit.csv`. Rules for any further change:

- **Diplomatic standard**: the gold follows the print — period spellings (*defence*),
  dialect (*aout*), printer's misprints (*hnman*, *wiliing*) and printed numerals. Don't
  "fix" these; `gold-audit.csv` lists misprints already confirmed on the scan.
- Change the gold only after checking the scan, and add a row to `gold-errata.csv`.
- The `txt/` files use **CRLF** line endings; read/write with `newline=""` to keep them.
- Scoring joins line-end hyphens (`load_text` in `score.py`), so a line-end dash in the
  gold must be an em dash (`throb—`), not a hyphen, or it merges with the next word.

## Scorer notes

- The old `overall` hang on the ~55k-char broadsheets (MEMO.md §2) was the pure-python
  O(n·m) Levenshtein fallback, used when rapidfuzz isn't installed. It is now an exact
  bit-parallel (Myers/Hyyrö) implementation: same numbers as rapidfuzz, ~1 s per big page.
- Folders under `ocr-results/` starting with `_` are skipped by `score.py`,
  `scoresheet.py` and `score_order.py` — use that prefix for smoke tests.
- `ocr-results/human-volunteer-v1/` is the pre-correction volunteer gold (v1), scored as
  a reference row. Don't edit it; it's what `docs/gold-errata.csv` was applied to.
