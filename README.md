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

### Baseline results (`newspaper-ocr`, 19 pages)

Higher is better. `overall` = 1 − CER (order-sensitive); `chrF` and `bowF1` are order-robust;
`gap = bowF1 − overall` is the reading-order / segmentation penalty. `broadsheet` = pages
≥ 4000 gold words, `page` = the rest.

| Backend | overall | broadsheet | page | chrF | bowF1 | gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `tesseract` default (AS YOLO + Tesseract) | **0.615** | 0.579 | 0.646 | 0.692 | 0.704 | 0.089 |
| `glm-ocr-mlx` (GLM-OCR 1.1B VLM) | 0.614 | 0.576 | 0.648 | 0.689 | **0.760** | 0.146 |
| `news_combo_fast` (fine-tuned Tesseract) | 0.606 | 0.576 | 0.632 | 0.666 | 0.695 | 0.089 |

**Reading it.** On the headline CER (`overall`) all three tie at ~0.61 — by that metric the
VLM does *not* beat Tesseract. But the split reveals why: on order-free `bowF1`, **GLM-OCR
leads by ~5–6 points (0.760 vs ~0.70)** — it genuinely recognizes the words better. It gives
that advantage back to reading order — its `gap` is 0.146 vs Tesseract's 0.089 — because the
layout stage stitches the VLM's regions into the wrong sequence more often. So on these dense
multi-column pages the VLM's bottleneck is **layout / reading order, not character
recognition**, which is exactly the distinction the `chrF`/`bowF1`/`gap` columns exist to
surface. Dense broadsheets (≥4000 words) cost every backend ~7 points versus shorter pages.

_(A LightOnOCR-2-1B run was attempted but is CPU-only and ~6× slower here; not included.)_
The gap of ~0.12–0.17 everywhere confirms that on multi-column newspapers, **sequencing
is a bigger error source than character recognition** — which is the whole point of this
benchmark.

Both Tesseract backends sit far below the 0.85–0.93 top vision-LLMs reach on the broader
InkBench mix: dense newspaper layout is genuinely harder.

**Notes.**
- Page `2019713453-2578` (a *New York Weekly* page) is a near-total **layout-detection**
  failure across all backends (they share the AS-YOLO detector), not a recognition
  failure. It is kept as a real failure case.
- Volunteer transcriptions are occasionally *partial* on dense ad-heavy pages, which
  unfairly punishes any OCR tool; one such page was removed during construction. A quick
  screen: a strong backend emitting far more text than the gold flags a likely partial
  gold rather than a bad transcription.

## Scoring another system

Write one plain-text file per page to `ocr-results/<your-model-name>/<id>.txt` (matching
the image stem), then run `python score.py`. It auto-discovers every folder under
`ocr-results/`.

## License / provenance

Images and transcriptions are U.S. Library of Congress *By the People* materials
(public domain in the United States). See each source collection on
[crowd.loc.gov](https://crowd.loc.gov/).
