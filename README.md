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

Forty genuine newspaper pages drawn from Library of Congress
[By the People](https://crowd.loc.gov/) crowd-transcribed collections, so every page
ships with a volunteer gold-standard transcription. Pages were selected to be **clean,
full newspaper pages** — no scrapbook/collage pages of pasted clippings (whose ambiguous
reading order makes CER unfair) and no short snippets. None appear in the InkBench
400-image benchmark, so the two measures are independent.

| Property | Value |
|:---|:---|
| Pages | 40 |
| Collections | 7 (NAWSA, Blackwell family, Mary Church Terrell, Carrie Chapman Catt, Anna E. Dickinson, Truly Douglass, Early Copyright) |
| Publications | ~25 distinct titles (The Liberator, New-York Daily Tribune, The Woman's Journal / Woman Citizen, Providence Journal, Des Moines Register, Christian Register, Unity, Democratic Digest, The Evening Star, Washington Afro-American, The Jewish Advocate, Morning Express, …) |
| Eras | 1850s abolitionist broadsheets through 1940s political weeklies |
| Words per page | 861 – 8,966 (median ≈ 2,090) |
| Total gold words | ~97,600 |
| Tiers | 27 `full_page` (1,500+ words), 13 `article` (600–1,499 words) |

The 4,700–9,000-word pages are 19th-century *Liberator*-era broadsheets — the hardest
tier, with very small type and six-plus columns.

> **Note on sourcing.** Roughly half the pages come from the NAWSA (suffrage) collection,
> but that collection is itself a scrapbook of many different newspapers, so the pages
> still span ~25 publications and a century of layout conventions.

## Files

```
images/            page scans (<id>.jpg), original LoC filenames
txt/               matching volunteer gold transcriptions (<id>.txt)
newsbench.csv      manifest: image_name, collection, words, tier, description
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
$NPO run_newspaper_ocr.py --name lightonocr-cpu --recognizer lightonocr --device cpu  # 1B VLM
```

`--device cpu` is required for the VLM backends on Apple Silicon: LightOnOCR (mistral3-based)
hard-aborts on MPS, and GLM-OCR needs an MLX/vLLM server (vLLM does not run on macOS).

### Baseline results (`newspaper-ocr`)

Higher is better. `overall` = 1 − CER; see the table above for the rest.

| Backend | overall | full_page | article | chrF | bowF1 | gap |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `news_combo_fast` (fine-tuned Tesseract) | **0.564** | 0.551 | 0.590 | 0.659 | 0.682 | 0.119 |
| `tesseract` default (AS YOLO + Tesseract) | 0.557 | 0.556 | 0.559 | 0.676 | 0.679 | 0.123 |
| `lightonocr-cpu` (LightOnOCR-2-1B VLM) | 0.481\* | 0.493\* | 0.460\* | 0.597\* | 0.655\* | 0.174\* |

\* VLM row is a partial run at time of writing; it will be refreshed on the full 40 pages.

**Reading it.** All three land near ~0.66 on order-free `bowF1` — recognition quality is
close. They separate mostly on the `gap` (reading order) and on a coverage/precision
trade-off: the VLM has the highest word precision (cleanest text) but the lowest recall
(it drops whole regions on the densest pages, likely per-region token-limit truncation).
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
