# NewsBench

A small, focused benchmark for **transcribing dense, multi-column historical
newspaper pages** — masthead, several columns, mixed article/advertisement layout,
small degraded type. It measures not just character recognition but **layout
understanding and reading order**, the part that generic OCR benchmarks miss.

> How well can an OCR system or vision LLM read a full newspaper page — every column,
> in the right order — not just the one article a reader cares about?

**Corpus:** 19 complete, long, original newspaper pages from Library of Congress
[By the People](https://crowd.loc.gov/) crowd transcriptions (1850s–1919; 15 scored,
4 excluded for partial gold). Every page ships with a volunteer gold transcription.
Fully self-contained — everything needed to score a model lives here.

See **[docs/methodology.md](docs/methodology.md)** for the corpus details, scoring
metrics, the `newspaper-ocr` harness, and provenance.

## Results

[`newspaper-ocr`](https://github.com/nealcaren/newspaper-ocr) v0.8.1, every combination
of the two strong detectors × three recognizers, scored on the same GPU (n = 15).
`overall` = 1 − CER (order-sensitive); `cased` keeps case + punctuation; `bowF1` is
order-free bag-of-words F1. Higher is better; all local (no API cost).

| Detector | Recognizer | overall | cased | bowF1 |
|:---|:---|:---:|:---:|:---:|
| **DocLayout-YOLO** | PaddleOCR-VL | **0.970** | **0.953** | **0.985** |
| **DocLayout-YOLO** | GLM-OCR | 0.959 | 0.943 | 0.985 |
| **DocLayout-YOLO** | Tesseract | 0.919 | 0.889 | 0.910 |
| AS-YOLO | PaddleOCR-VL | 0.816 | 0.801 | 0.937 |
| AS-YOLO | GLM-OCR | 0.803 | 0.788 | 0.942 |
| AS-YOLO | Tesseract | 0.620 | 0.602 | 0.706 |

**The detector is the whole game.** Holding the recognizer fixed and only swapping the
detector moves the score more than anything else: +0.156 for GLM-OCR, +0.154 for
PaddleOCR-VL, and **+0.30** for Tesseract. On dense multi-column pages, *segmenting the
page correctly is harder than reading the type.*

- **DocLayout-YOLO wins** — it proposes finer, more complete regions, so even Tesseract
  reaches 0.919, and the two VLMs are near-perfect peers (0.970 / 0.959).
- **AS-YOLO caps the VLMs at ~0.81** despite bowF1 ~0.94 — the words are right, the
  reading order isn't. That gap *is* the segmentation penalty.
- **DocLayout-YOLO + Tesseract (0.919)** is a strong fully-local, free, no-VLM option.

Hosted models (Gemini, GPT, etc.), cost/speed, and every historical run are in the full
score sheet — regenerate with `python scoresheet.py` (see
[docs/methodology.md](docs/methodology.md)).

## Score your own system

Write one plain-text file per page to `ocr-results/<your-model-name>/<id>.txt` (matching
the image stem in `images/`), then:

```bash
pip install -r requirements.txt   # rapidfuzz for fast scoring
python score.py                   # auto-discovers every folder under ocr-results/
```

## License

Code: [MIT](LICENSE). Page scans and gold transcriptions are U.S. Library of Congress
*By the People* public-domain / CC0 materials — please credit
[crowd.loc.gov](https://crowd.loc.gov/). Details in
[docs/methodology.md](docs/methodology.md).
