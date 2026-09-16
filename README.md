# NewsBench

A small, focused benchmark for **transcribing dense, multi-column historical
newspaper pages** — masthead, several columns, mixed article/advertisement layout,
small degraded type. It measures not just character recognition but **layout
understanding and reading order**, the part that generic OCR benchmarks miss.

NewsBench began as a spinoff of [InkBench](https://github.com/nealcaren/InkBench)
(a broader historical-document OCR benchmark) and reuses its transcription and scoring
conventions, but it is self-contained: everything needed to score a model lives here.

> How well can an OCR system or vision LLM read a full newspaper page — every column,
> in the right order — not just the one article a reader cares about?

## Corpus

Nineteen **complete, long, original newspaper pages** drawn from Library of Congress
[By the People](https://crowd.loc.gov/) crowd-transcribed collections, so every page
ships with a volunteer gold-standard transcription. None appear in the InkBench
400-image benchmark, so the two measures are independent.

| Property | Value |
|:---|:---|
| Pages | 19 |
| Publications | ~13 distinct papers (The Liberator, Boston Daily Advertiser, Hartford Daily Courant, Des Moines Register & Leader, Providence Sunday Journal, Anti-Suffrage Notes, Maine Anti-Suffragist, Industrial Equality, an Oberlin college paper, a 1898 Indianapolis paper, and others) |
| Eras | 1850s abolitionist broadsheets through 1919 |
| Words per page | 1,892 – 8,966 (median ≈ 3,300) |
| Total gold words | ~76,600 |
| Every page | a complete edge-to-edge printed page, not a clipping or excerpt |

**Selection.** Each page was verified *by eye* to be a true full newspaper page — a
complete printed sheet with columns running edge to edge — and required to carry a long
gold transcription (≥ ~1,900 words). The length floor is deliberate: NewsBench targets
models that can sustain a full-page, multi-column transcription, and it doubles as a
filter against *partial* gold (volunteers who transcribed only part of a dense page).

> **Note on sourcing — read this before citing coverage.** These pages were mined from
> personal-papers collections (mostly the NAWSA suffrage records), which are dominated by
> *clippings*, not full sheets. Genuine full newspaper pages are rare there, and the
> Library's *By the People* program has **no dedicated newspaper campaign** to draw from —
> every published transcription dataset is manuscripts, letters, or diaries. So while the
> 19 pages span ~13 real newspapers and seven decades, they were all captured inside
> suffrage-era collections and skew abolitionist/suffrage in subject; several are pages of
> the same *Liberator* issues. This is the realistic ceiling for *gold-transcribed* full
> newspaper pages in the LoC crowd archive, not a balanced sample of American newspapers.

## Files

```
images/            page scans (<id>.jpg), original LoC filenames
txt/               matching volunteer gold transcriptions (<id>.txt)
newsbench.csv      manifest: image_name, collection, words, tier, newspaper_note
ocr-results/       one folder per system: <model-name>/<id>.txt
run_newspaper_ocr.py   runner for the newspaper-ocr pipeline
score.py           scorer (recognition + reading-order metrics)
score_order.py     detailed order-sensitive vs order-free word breakdown
```

## Scoring

```bash
pip install -r requirements.txt   # rapidfuzz (+ scipy for --aligned); optional but fast

python score.py            # main table
python score.py --aligned  # add block-aligned CER (slower, char-level order-removed)
python score_order.py      # word-level recall/precision + ordering gap
```

`score.py` reports, per model in `ocr-results/`:

| Column | Meaning | Order? |
|:---|:---|:---|
| `overall` | 1 − CER (alphanumeric, lowercase); InkBench's headline metric | **sensitive** |
| `full_page` / `article` | the same, split by tier | sensitive |
| `chrF` | character n-gram F-score (n=1..6, β=2) | robust |
| `bowF1` | bag-of-words F1 (page as a multiset of tokens) | **free** |
| `gap` | `bowF1 − overall` = the reading-order / segmentation penalty | — |

**Why more than one number.** Plain CER is a global sequence edit distance: a page whose
words are all correct but whose articles are in the wrong order is penalised almost as
hard as one that misread every word. `chrF` (local n-grams) and `bowF1` (order-free) stay
high when only the *order* is wrong, so a large `gap` isolates reading-order failures
from recognition failures. `score_order.py` decomposes further into bag-of-words recall
(coverage) and precision (spurious text).

## Testing the `newspaper-ocr` library

`run_newspaper_ocr.py` drives the [`newspaper-ocr`](https://github.com/nealcaren/newspaper-ocr)
pipeline over `images/` (run it with that project's interpreter):

```bash
NPO=/path/to/newspaper-ocr/.venv/bin/python

$NPO run_newspaper_ocr.py --name tesseract-default                        # AS YOLO + Tesseract
$NPO run_newspaper_ocr.py --name news_combo_fast --model news_combo_fast  # bundled fine-tuned
$NPO run_newspaper_ocr.py --name glm-ocr-mlx --recognizer glm-ocr         # GLM-OCR VLM via MLX server
```

The GLM-OCR VLM runs in the pipeline's default `api` mode against a local
[MLX](https://github.com/ml-explore/mlx) server (Apple-Silicon-native):

```bash
pip install mlx-vlm
python -m mlx_vlm.server --model mlx-community/GLM-OCR-bf16 --port 8080
```

On Apple Silicon this is the practical VLM path — GLM-OCR is ~1.1B params, and MLX uses the
Metal GPU. The PyTorch VLM backends do **not** work well here: LightOnOCR (mistral3-based)
hard-aborts on MPS and is ~6× slower on CPU, and GLM-OCR's own `local` mode falls back to
CPU (it disables MPS due to vision-position-id bugs). vLLM does not run on macOS.

### Headline results (`newspaper-ocr`, n = 15)

Higher is better. `overall` = 1 − CER (lowercased, alnum-only, order-sensitive);
`cased` = 1 − CER keeping **case and punctuation** (the honest character score);
`bowF1` = order-free bag-of-words F1. `$/100pg` is real API spend per 100 pages
(**$0** for local models).

| Detector | OCR | newspaper-ocr | overall | cased | bowF1 | $/100pg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| PaddleX | GLM-OCR | 0.7.0 | **0.937** | **0.922** | **0.980** | $0.00 |
| PaddleX | Gemini-flash-lite | 0.7.0 | 0.936 | 0.915 | 0.944 | $4.51 |
| PaddleX | GLM-OCR | 0.6.0 | 0.919 | 0.905 | 0.961 | $0.00 |
| PaddleX | Tesseract | 0.7.0 | 0.899 | 0.874 | 0.891 | $0.00 |
| none (whole page) | Gemini-flash-lite |  | 0.820 | 0.803 | 0.867 | $2.88 |
| none (whole page) | Tesseract |  | 0.677 | 0.662 | 0.844 | $0.00 |

The **whole-page rows use no detector** — the recognizer is handed the raw page,
so newspaper-ocr isn't doing any layout work; these are the naive baseline (blank
version). The `PaddleX` rows are the library's job. newspaper-ocr **0.7.0** turns
the residual second pass on by default for region-level recognizers (GLM-OCR,
Gemini); **0.6.0** is the prior behavior — the version only changes those rows.

**Reading it.**
- **The detector dominates.** Adding PaddleX layout detection is worth +0.22 for
  Tesseract (0.677 → 0.899) and +0.12 for Gemini (0.820 → 0.936) — far more than
  the choice of recognizer. On these dense multi-column pages, *segmenting the page
  is the hard part.*
- **PaddleX + GLM-OCR + residual is the recommended stack** and tops every column,
  with the widest `bowF1` lead (0.980) — it recovers and orders the most words.
- **Upgrading [newspaper-ocr](https://github.com/nealcaren/newspaper-ocr) 0.6.0 →
  0.7.0** (residual second pass on by default for region recognizers) lifts
  GLM-OCR 0.919 → 0.937 and Gemini the same way — it recovers whole columns the
  detector missed.
- **`cased` keeps the ranking** and widens the gap to Tesseract (its weakness is
  punctuation, which the default `overall` metric ignores).

Regenerate the full table (all backends, tiers, tokens) with `python scoresheet.py`.

**Partial-gold exclusions.** Volunteer transcriptions are sometimes *partial* on
dense ad-heavy pages, which unfairly punishes any OCR tool that reads the whole
page. Such pages are flagged `exclude=partial_gold` in `newsbench.csv` and skipped
by `score.py`/`scoresheet.py` for every model (kept in `images/`+`txt/` for
provenance). A reliable screen: a strong backend emitting far more text than the
gold flags a likely partial gold. Currently excluded (→ **n = 15**):
`mss3413201856-40`, `mss3413201856-23`, `mss3413201776-3`, `mss83434402-9` — on each,
OCR systems recover 1.3–1.8× the gold's word count of real, coherent text.

## Scoring another system

Write one plain-text file per page to `ocr-results/<your-model-name>/<id>.txt` (matching
the image stem), then run `python score.py`. It auto-discovers every folder under
`ocr-results/`.

## License / provenance

The code in this repository (scoring scripts, harness, configuration) is released
under the [MIT License](LICENSE).

The page scans in `images/` and the gold transcriptions in `txt/` are U.S. Library
of Congress *By the People* materials — the scans are public domain and the
volunteer transcriptions are released into the public domain (CC0). No additional
copyright is asserted over these files here; please credit the Library of Congress
*By the People* program ([crowd.loc.gov](https://crowd.loc.gov/)) when reusing the
corpus.
