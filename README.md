# NewsBench

A small, focused benchmark for **transcribing dense, multi-column historical
newspaper pages** — masthead, several columns, mixed article/advertisement layout,
small degraded type. It measures not just character recognition but **layout
understanding and reading order**, the part that generic OCR benchmarks miss.

> How well can an OCR system or vision LLM read a full newspaper page — every column,
> in the right order — not just the one article a reader cares about?

Fully self-contained: the corpus, the gold transcriptions, and the scoring tools
all live here. Bring your own OCR output and score it.

## The corpus

Nineteen **complete, long, original newspaper pages** from Library of Congress
[By the People](https://crowd.loc.gov/) crowd transcriptions, each shipping with a
volunteer gold-standard transcription.

| Property | Value |
|:---|:---|
| Pages | 19 (15 scored; 4 excluded for partial gold) |
| Publications | ~13 distinct papers (The Liberator, Boston Daily Advertiser, Hartford Daily Courant, Des Moines Register & Leader, and others) |
| Eras | 1850s abolitionist broadsheets through 1919 |
| Words per page | 1,892 – 8,966 (median ≈ 3,300) |
| Total gold words | ~76,600 |
| Every page | a complete edge-to-edge printed page, not a clipping or excerpt |

Pages were verified by eye to be true full newspaper sheets and required to carry a
long gold transcription (≥ ~1,900 words). Four pages whose volunteer gold is only
*partial* are flagged `exclude=partial_gold` in `newsbench.csv` and skipped by the
scorers. See **[docs/methodology.md](docs/methodology.md)** for corpus selection,
sourcing caveats, the exclusions, and provenance.

## Files

```
images/            page scans (<id>.jpg), original LoC filenames
txt/               matching volunteer gold transcriptions (<id>.txt)
newsbench.csv      manifest: image_name, collection, words, tier, exclude, newspaper_note
ocr-results/       one folder per system: <model-name>/<id>.txt
```

## Scoring tools

```bash
pip install -r requirements.txt   # rapidfuzz (fast); scipy for score.py --aligned
```

| Script | What it does |
|:---|:---|
| `score.py` | Per-model quality table: `overall` (1 − CER), `cased`, `chrF`, `bowF1`, and the reading-order `gap`. `--aligned` adds block-aligned CER. |
| `scoresheet.py` | Combined sheet — quality + cost + speed, one row per `ocr-results/` folder. `--csv` writes `scoresheet.md` / `scoresheet.csv`. |
| `score_order.py` | Word-level recall/precision + ordering-gap breakdown. |
| `run_newspaper_ocr.py` | Runner that drives the [`newspaper-ocr`](https://github.com/nealcaren/newspaper-ocr) pipeline over `images/`. |

Metrics are designed to separate **recognition** from **reading order**: plain CER
punishes a correctly-read-but-mis-ordered page almost as hard as a misread one, so
`chrF` (local n-grams) and `bowF1` (order-free) stay high when only the order is
wrong, and `gap = bowF1 − overall` isolates the segmentation/ordering penalty. Full
details in [docs/methodology.md](docs/methodology.md).

## Score your own system

Write one plain-text file per page to `ocr-results/<your-model-name>/<id>.txt`
(matching the image stem in `images/`), then:

```bash
python score.py        # auto-discovers every folder under ocr-results/
python scoresheet.py   # combined quality/cost/speed sheet
```

## License

Code: [MIT](LICENSE). Page scans and gold transcriptions are U.S. Library of Congress
*By the People* public-domain / CC0 materials — please credit
[crowd.loc.gov](https://crowd.loc.gov/). Details in
[docs/methodology.md](docs/methodology.md).
