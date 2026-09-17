# NewsBench — methodology & details

Detailed corpus, scoring, and harness documentation for
[NewsBench](../README.md). The README carries the summary and headline results;
this page has everything else.

## Corpus

Nineteen **complete, long, original newspaper pages** drawn from Library of Congress
[By the People](https://crowd.loc.gov/) crowd-transcribed collections, so every page
ships with a volunteer gold-standard transcription. None appear in the
[InkBench](https://github.com/nealcaren/InkBench) 400-image benchmark, so the two
measures are independent.

| Property | Value |
|:---|:---|
| Pages | 19 (15 scored; 4 excluded for partial gold — see below) |
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

### Partial-gold exclusions

Volunteer transcriptions are sometimes *partial* on dense ad-heavy pages, which
unfairly punishes any OCR tool that reads the whole page. Such pages are flagged
`exclude=partial_gold` in `newsbench.csv` and skipped by `score.py` / `scoresheet.py`
for every model (kept in `images/` + `txt/` for provenance). A reliable screen: a
strong backend emitting far more text than the gold flags a likely partial gold.
Currently excluded (→ **n = 15**): `mss3413201856-40`, `mss3413201856-23`,
`mss3413201776-3`, `mss83434402-9` — on each, OCR systems recover 1.3–1.8× the gold's
word count of real, coherent text.

## Files

```
images/            page scans (<id>.jpg), original LoC filenames
txt/               matching volunteer gold transcriptions (<id>.txt)
newsbench.csv      manifest: image_name, collection, words, tier, exclude, newspaper_note
ocr-results/       one folder per system: <model-name>/<id>.txt
run_newspaper_ocr.py   runner for the newspaper-ocr pipeline
score.py           scorer (recognition + reading-order metrics)
scoresheet.py      combined score sheet (quality + cost + speed), one row per folder
score_order.py     detailed order-sensitive vs order-free word breakdown
```

## Scoring

```bash
pip install -r requirements.txt   # rapidfuzz (+ scipy for --aligned); optional but fast

python score.py            # main table
python score.py --aligned  # add block-aligned CER (slower, char-level order-removed)
python scoresheet.py       # combined quality + cost + speed sheet (writes scoresheet.md with --csv)
python score_order.py      # word-level recall/precision + ordering gap
```

Metrics reported per model in `ocr-results/`:

| Column | Meaning | Order? |
|:---|:---|:---|
| `overall` | 1 − CER (alphanumeric, lowercase); InkBench's headline metric | **sensitive** |
| `cased` | 1 − CER keeping **case and punctuation** (the honest character score) | sensitive |
| `full_page` / `article` | `overall` split by tier | sensitive |
| `chrF` | character n-gram F-score (n=1..6, β=2) | robust |
| `bowF1` | bag-of-words F1 (page as a multiset of tokens) | **free** |
| `gap` | `bowF1 − overall` = the reading-order / segmentation penalty | — |

**Why more than one number.** Plain CER is a global sequence edit distance: a page whose
words are all correct but whose articles are in the wrong order is penalised almost as
hard as one that misread every word. `chrF` (local n-grams) and `bowF1` (order-free) stay
high when only the *order* is wrong, so a large `gap` isolates reading-order failures
from recognition failures. `score_order.py` decomposes further into bag-of-words recall
(coverage) and precision (spurious text). `cased` keeps case and punctuation, which the
default `overall` metric strips — it doesn't change the ranking but widens the gap to
engines weak at punctuation (e.g. Tesseract).

## Testing the `newspaper-ocr` library

`run_newspaper_ocr.py` drives the [`newspaper-ocr`](https://github.com/nealcaren/newspaper-ocr)
pipeline over `images/` (run it with that project's interpreter):

```bash
NPO=/path/to/newspaper-ocr/.venv/bin/python

$NPO run_newspaper_ocr.py --name doclayout-tess                         # DocLayout-YOLO + Tesseract
$NPO run_newspaper_ocr.py --name glm-ocr-mlx --recognizer glm-ocr       # GLM-OCR VLM via MLX server
```

The detector defaults to `auto` (DocLayout-YOLO when installed, else PaddleX, else
AS-YOLO). Install the recommended detector with `newspaper-ocr[doclayout]`.

### GLM-OCR on Apple Silicon (MLX)

GLM-OCR runs in the pipeline's default `api` mode against a local
[MLX](https://github.com/ml-explore/mlx) server (Apple-Silicon-native):

```bash
pip install mlx-vlm
python -m mlx_vlm.server --model mlx-community/GLM-OCR-bf16 --port 8080
```

On Apple Silicon this is the practical VLM path — GLM-OCR is ~1.3B params, and MLX uses the
Metal GPU. The PyTorch VLM backends do **not** work well on macOS: LightOnOCR (mistral3-based)
hard-aborts on MPS and is ~6× slower on CPU, and GLM-OCR's `local` mode falls back to
CPU. vLLM does not run on macOS.

### VLM recognizers on GPU (Linux/CUDA)

On a CUDA GPU the region VLMs run in local (transformers) mode directly:

```bash
$NPO run_newspaper_ocr.py --recognizer glm-ocr --name doclayout-glm      # needs newspaper-ocr >= 0.8.1
$NPO run_newspaper_ocr.py --recognizer paddleocr-vl --name doclayout-pvl
```

Requires newspaper-ocr **≥ 0.8.1** (earlier versions returned empty output on GPU —
they loaded with `device_map="auto"`, which needs `accelerate`, and used a per-region
timeout too short for large column crops). GLM-OCR ≈ 90 s/page on an L40S;
PaddleOCR-VL is ~5× slower (~430 s/page) but slightly more accurate.

## License / provenance

The code in this repository (scoring scripts, harness, configuration) is released
under the [MIT License](../LICENSE).

The page scans in `images/` and the gold transcriptions in `txt/` are U.S. Library
of Congress *By the People* materials — the scans are public domain and the
volunteer transcriptions are released into the public domain (CC0). No additional
copyright is asserted over these files here; please credit the Library of Congress
*By the People* program ([crowd.loc.gov](https://crowd.loc.gov/)) when reusing the
corpus.
