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
| Pages | 19, all scored (four were excluded until Oct 2026 — see below) |
| Publications | ~13 distinct papers (The Liberator, Boston Daily Advertiser, Hartford Daily Courant, Des Moines Register & Leader, Providence Sunday Journal, Anti-Suffrage Notes, Maine Anti-Suffragist, Industrial Equality, an Oberlin college paper, a 1898 Indianapolis paper, and others) |
| Eras | 1850s abolitionist broadsheets through 1919 |
| Words per page | 1,892 – 8,966 (median ≈ 3,300) |
| Total gold words | ~82,000 |
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

### Restored pages (formerly excluded)

Until Oct 2026 four pages were flagged `exclude=partial_gold` and scored at n = 15:
`mss3413201856-40`, `mss3413201856-23`, `mss3413201776-3`, `mss83434402-9`. The screen
that flagged them (systems emit far more text than the gold) turned out to be half
right. Checked against the scans:

- **`mss3413201856-40`** really was partial: columns 1, 3 and 4 and the top of column 5
  (railroad news, the Commercial Record markets, imports/exports, the New York market
  telegraph) had never been transcribed. About 4,700 words were added from scan crops
  (errata R301–R304); three illegible figures are marked `(?)`.
- **`mss3413201856-23`** was missing most of its far-right column (Dartmouth and
  Worcester Free Institute commencements, ~690 words), which the scan crops at the edge;
  it is transcribed as far as each line is visible (R101–R102).
- **`mss3413201776-3`** and **`mss83434402-9`** were complete all along. Their excess
  "missing" text was DocLayout region systems reading overlapping boxes twice; whole-page
  and PaddleX systems, which don't double-read, show ~1% extra, like every scored page.
  One volunteer note not in the print was removed (R201).

So the partial-gold screen must use systems that don't duplicate (whole-page or
non-overlapping region runs), or count duplicate text separately; a median over
DocLayout runs flags complete pages. The `exclude` column and `load_excludes()` remain
for future cases. The restored text was spot-checked line by line against the scans by
an independent pass (see the PR for results).

### Gold corrections (2026-10)

The volunteer transcriptions are good but not perfect. On the original 15 scored pages, 269
spots were corrected against the scans (191 replacements, 46 deletions, 32 insertions;
every one listed with its context and evidence in
[`gold-errata.csv`](gold-errata.csv)). Two pages hold half of them: `mss3413201417-29`
(75) and `mss3413201628-41` (60, a dialect-humour page).

**How they were found.** Eight independent systems (MinerU, gpt-5.6-luna,
PaddleOCR-VL, GLM-OCR, gemini, mistral, deepseek, Tesseract) were word-aligned to the
gold after reordering each output to the gold's order. A spot became a candidate when
at least five of the eight agreed on the same reading against the gold, when the gold
had words no system produced, or when at least three systems produced text the gold
lacked. All 411 candidates were then checked against crops of the page scan (each
verdict, with its evidence, is in [`gold-audit.csv`](gold-audit.csv)); 126 turned
out to be gold-correct (mostly text the systems read twice, or printer's misprints the
gold had kept), and 4 were unreadable and left alone. Only spots confirmed on the scan
were changed, so a system gains nothing just for agreeing with the others.

**Standard: diplomatic.** The gold now follows the print: period spellings
(*defence*, *practise*, *counsellor*, *pedlar*), dialect spellings (*aout*, *minits*),
printer's misprints (*hnman*, *belieing*, *wiliing*) and the printed numerals. What was
fixed: volunteer typos and misreadings (*press* for *persons*, *chair* for *chain*,
*priority* for *propriety*, *woolen* for *women*), modernized spellings, words the
volunteer added (*by the Virginia planter* for *by a*), passages typed twice, printed
lines the volunteer skipped, missing headlines and datelines, and line-end dashes typed
as hyphens (*throb-* for *throb—*), which the scorer would otherwise join with the next
word. The masthead year on `mss3413201283-24` is kept as 1913: the last digit is broken
(OCR splits 1919/1913/1918), but the article calls Wilson "a historian" rather than
President, never mentions the World War, and a clipping pasted on the page, about the
then-new Woman Suffrage Party of Providence, is hand-dated "Fri. Jan 10, 1913".

Effect: every system's `overall` rises by about 0.003 and `bowF1` by about 0.004; the
ranking does not change. The original volunteer text (gold v1) is kept as a scored row,
`ocr-results/human-volunteer-v1/`: on the original 15 pages it gets `overall` 0.997 and
`bowF1` 0.995 against the corrected gold, the size of the correction. Over all 19 it
drops to 0.971, because v1 lacks the text later restored on two pages.

## Files

```
images/            page scans (<id>.jpg), original LoC filenames
txt/               matching gold transcriptions (<id>.txt): volunteer text, corrected per gold-errata.csv
newsbench.csv      manifest: image_name, collection, words, tier, exclude, newspaper_note
ocr-results/       one folder per system: <model-name>/<id>.txt
run_newspaper_ocr.py   runner for the newspaper-ocr pipeline
score.py           scorer (recognition + reading-order metrics)
scoresheet.py      combined score sheet (quality + cost + speed), one row per folder
score_order.py     detailed order-sensitive vs order-free word breakdown
```

## Scoring

```bash
pip install -r requirements.txt   # rapidfuzz (+ scipy for --aligned); optional

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

Without rapidfuzz, `score.py` falls back to an exact bit-parallel (Myers/Hyyrö) edit
distance in pure Python, so the numbers are identical and the largest broadsheets still
finish in about a second each. Folders under `ocr-results/` whose name starts with `_`
(smoke tests, scratch runs) are skipped by every scorer.

**Unicode.** Text is NFKC-normalized on load, so a VLM's `47½` and the gold's ASCII
`47 1/2` compare equal (otherwise the `½` is dropped and the read counts as an error);
this matters mainly on the market-report page `mss3413201856-40`.

**Line-break hyphens.** Before any metric, a hyphen at the end of a line is joined to
the next line's word in both the gold and the output (`load_text` in `score.py`). Many
VLMs keep the printed hyphenation (`mus-` / `tered`) while the gold joins the word, and
without this each hyphenated line end counted as one missing word plus two novel ones,
about 0.035 bowF1 for the hosted VLMs. The alphanumeric `overall` CER was never
affected.

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
