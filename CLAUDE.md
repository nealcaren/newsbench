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
stays fair. Excluded pages are kept in `images/`+`txt/` for provenance. Current exclusion:
`mss3413201856-40` (partial gold — volunteer transcribed only ~half the page). Scores are
therefore over **18** pages, not 19. The reusable screen for partial-gold pages: compare a
page's gold word count against the independent OCR consensus — a page where every backend
emits far more coherent text than the gold is a suspect (see MEMO.md §1).

## Known issue

`overall` (global Levenshtein CER) is O(n·m) and can hang on the largest ~9,000-word
broadsheets (~55k normalized chars). `bowF1`/`chrF` are unaffected. If `score.py` prints
the header then appears to hang, this is why. MEMO.md §2 details fix options (length-guarded
/ chunked CER using the existing `aligned_cer` machinery, or a max-distance cutoff).
