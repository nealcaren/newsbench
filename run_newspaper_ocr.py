#!/usr/bin/env python
"""Run the newspaper-ocr library over the NewsBench pages.

Must be run with the newspaper-ocr project's interpreter, e.g.:

    NPO=/Users/nealcaren/Documents/GitHub/newspaper-ocr/.venv/bin/python

    # Library default: PP-DocLayout (paddlex) + Tesseract
    $NPO run_newspaper_ocr.py --name paddlex-tesseract

    # PP-DocLayout + GLM-OCR VLM, with the region-repair + recovery ladder
    $NPO run_newspaper_ocr.py --name paddlex-glm --recognizer glm-ocr \
        --region-repair --chunk-tall-regions --fallback paddleocr-vl

Writes one <id>.txt per page into newsbench/ocr-results/<name>/.

The detector defaults to `auto` (= paddlex/PP-DocLayout when installed, else
as_yolo) to match the library default; PP-DocLayout is the recommended detector
for dense newspaper pages. `--region-repair` runs the post-recognition
`RegionRepair` pass (lossless dedup, container-split with strip re-OCR,
fragmented-ad merge) — it needs a region-level recognizer (e.g. glm-ocr).
"""
import argparse, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"

# Region-level VLM recognizers (as opposed to line-level tesseract/kraken/trocr).
REGION_VLMS = ("glm-ocr", "lightonocr", "paddleocr-vl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="output folder name under ocr-results/")
    ap.add_argument("--detector", default="auto",
                    help="layout detector: auto (default; paddlex if installed) | "
                         "paddlex (PP-DocLayout) | as_yolo | doclayout_yolo")
    ap.add_argument("--recognizer", default="tesseract",
                    help="tesseract (default) | glm-ocr | paddleocr-vl | lightonocr | kraken | ...")
    ap.add_argument("--model", default=None, help="recognizer_model, e.g. news_combo_fast")
    ap.add_argument("--spell-check", action="store_true")
    ap.add_argument("--device", default=None,
                    help="force device for VLM backends: cpu/mps/cuda (mps can crash)")
    # Recovery ladder (region-level recognizers only) — see newspaper-ocr README.
    ap.add_argument("--fallback", default=None,
                    help="backup region recognizer for failed regions, e.g. paddleocr-vl")
    ap.add_argument("--chunk-tall-regions", action="store_true",
                    help="split tall regions that time out into bands and re-OCR each")
    ap.add_argument("--region-repair", action="store_true",
                    help="post-recognition RegionRepair pass (dedup + container-split + "
                         "ad-merge); needs a region-level recognizer")
    ap.add_argument("--limit", type=int, default=0, help="only first N images (debug)")
    args = ap.parse_args()

    from newspaper_ocr import Pipeline

    kw = {
        "detector": args.detector,
        "spell_check": args.spell_check,
        "chunk_tall_regions": args.chunk_tall_regions,
    }
    if args.fallback:
        kw["fallback"] = args.fallback

    # VLM backends are device-sensitive; build the recognizer object when a
    # device is pinned so we don't inherit an auto default (mps) that aborts.
    if args.device and args.recognizer in REGION_VLMS:
        if args.recognizer == "lightonocr":
            from newspaper_ocr.recognizers.lightonocr import LightOnOcrRecognizer
            kw["recognizer"] = LightOnOcrRecognizer(device=args.device)
        elif args.recognizer == "paddleocr-vl":
            from newspaper_ocr.recognizers.paddleocr_vl import PaddleOcrVlRecognizer
            kw["recognizer"] = PaddleOcrVlRecognizer(device=args.device)
        else:
            from newspaper_ocr.recognizers.glm_ocr import GlmOcrRecognizer
            kw["recognizer"] = GlmOcrRecognizer(device=args.device)
    else:
        kw["recognizer"] = args.recognizer
        if args.model:
            kw["recognizer_model"] = args.model

    pipe = Pipeline(**kw)

    # RegionRepair is a caller-invoked post pass, not a Pipeline kwarg: analyze
    # (detect + recognize) -> repair -> format. It re-OCRs the strips it opens up,
    # so it needs a region-level recognizer.
    repair = None
    if args.region_repair:
        if not isinstance(pipe.recognizer, __import__(
                "newspaper_ocr.recognizers.base", fromlist=["RegionRecognizer"]
        ).RegionRecognizer):
            ap.error("--region-repair needs a region-level recognizer "
                     f"(e.g. --recognizer glm-ocr); got {args.recognizer}")
        from newspaper_ocr.region_repair import RegionRepair
        repair = RegionRepair(recognizer=pipe.recognizer)

    def transcribe(img_path):
        if repair is None:
            return pipe.ocr(img_path)
        layout = pipe.analyze(img_path)
        layout = repair.repair(layout)
        return pipe.formatter.format(layout)

    out = HERE / "ocr-results" / args.name
    out.mkdir(parents=True, exist_ok=True)

    imgs = sorted(IMAGES.glob("*.jpg"))
    if args.limit:
        imgs = imgs[: args.limit]

    t0 = time.time()
    for i, img in enumerate(imgs, 1):
        dst = out / (img.stem + ".txt")
        if dst.exists():
            print(f"[{i}/{len(imgs)}] skip {img.name}", flush=True)
            continue
        t = time.time()
        try:
            text = transcribe(str(img))
        except Exception as e:  # keep going; record the failure
            text = ""
            print(f"[{i}/{len(imgs)}] ERROR {img.name}: {e}", flush=True)
        dst.write_text(text, encoding="utf-8")
        print(f"[{i}/{len(imgs)}] {img.name} {time.time()-t:.1f}s {len(text)}c", flush=True)

    print(f"done {len(imgs)} pages in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
