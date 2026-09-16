# Memo: scorer fix + partial-gold page removal

_2026-09-15. Findings from running the `progressive-magazines-ocr` pipeline
(PP-DocLayout_plus-L + GLM-OCR) through NewsBench and cross-checking the gold._

## 1. Remove the bum page — `mss3413201856-40` (partial gold)

The gold transcription for `mss3413201856-40` (Boston Daily Advertiser, "full
page") covers only about **half the page** and **ends mid-advertisement**
(`"...inquire of Auctioneers, No. 9"` — no address, no close). It lists 5,253
gold words for a full broadsheet, while its sibling full page
(`mss3413201856-23`) has 8,966.

Every OCR backend independently recovers **~1.9× the gold's word count** of real,
coherent text (election results, financial notices, auction ads) that the
volunteer never transcribed — so the page unfairly punishes precision/CER for
*correct* reads.

**How it was found (reusable screen):** compare the gold word count against the
**independent OCR consensus**. A page where every backend emits far more text
than the gold — or where the gold ends mid-sentence — is a partial-gold suspect.
Running this over all 19 pages flagged **only** `mss3413201856-40` (best OCR
1.89× gold; next-highest page just 1.07×). Note the opposite case,
`mss3413201309-4`, where OCR recovers *less* than gold and 9% of gold words
appear in no OCR output — that's a genuinely hard page with **correct** gold,
not a bad one. Keep it.

**Action taken:**
- `newsbench.csv` — added an `exclude` column; `mss3413201856-40` is flagged
  `partial_gold` with the reason inline.
- `score.py` — added `load_excludes()`; flagged pages are skipped for **every**
  model, so the comparison stays fair. Scores are now over **18** pages.
- `README.md` — documented the exclusion + the cross-check method.
- The image and gold file are **kept** in `images/`+`txt/` for provenance; only
  scoring skips them.

Impact: dropping it raised the strongest backend from bowF1 0.950 (19pp) to
**0.965** (18pp) — i.e. the page was masking real quality with a gold artifact.

## 2. Fix the scorer — `overall` (CER) hangs on the largest pages

`score.py`'s headline metric `overall = 1 − CER` uses a global Levenshtein edit
distance. That's **O(n·m)**, and on a full ~9,000-word broadsheet (~55k
normalized chars) a single page doesn't finish in a reasonable time. Symptom:
`score.py` prints the header row and then appears to hang — it's stuck computing
CER on the biggest page(s). `bowF1` and `chrF` are unaffected (both ran fine over
all 18 pages).

This is why the full-corpus `overall` column couldn't be produced in this
session; the order-free `bowF1` table was used as the headline instead.

**Options to fix (pick one):**
- **Length guard / chunked CER.** Above some char threshold, compute CER on
  aligned blocks (the machinery already exists in `aligned_cer`) or on fixed
  windows, instead of one global edit distance. Fastest to implement.
- **Cheaper backend.** Confirm `rapidfuzz.distance.Levenshtein.distance` is
  actually being used (it is when importable) — but even bit-parallel it's too
  slow at 55k chars. A banded/Myers implementation with a max-distance cutoff
  (`score_cutoff`) would bound the work; CER only needs the raw distance, and a
  cutoff at, say, `len(ref)` avoids the worst case.
- **Skip-and-note.** If `overall` on giant pages isn't worth the cost, cap it and
  report `overall` only up to a length, flagging the capped pages — but that
  weakens the headline metric, so prefer a real fix.

Recommended: add a `--fast`/length-guarded CER path so `overall` stays the
headline metric without hanging on the densest (and most interesting) pages.

## 3. Context: what the run showed (for the record)

- Our pipeline's advantage over the `glm-ocr-mlx` backend is **layout coverage**,
  not the recognizer: both use GLM-OCR, but `glm-ocr-mlx` uses newspaper-ocr's
  **AS-YOLO** detector, which under-segments dense pages and drops whole
  regions (recovering ~50–70% of gold words vs our ~95–105%). PP-DocLayout boxes
  the full page. So the ~0.20 bowF1 gap is real and about detection, not OCR.
- Our remaining weakness is small: PP-DocLayout occasionally leaves **uncovered
  bands** (a top masthead strip, thin inter-block text) that never reach the
  recognizer — see `mss3413201323-27` (~566 real words missed). An "OCR the
  leftover strips" pass would recover most of it.
- Benchmarking gotcha unrelated to the scorer: the **local MLX GLM-OCR server can
  crash under sustained batch load** (connection-refused → whole pages drop to
  ~0). Run page-by-page with a health-check/restart (see
  `run_progmag_robust.py`), or use an in-process recognizer for batch runs.
