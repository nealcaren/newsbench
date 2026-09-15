#!/usr/bin/env python
"""Run the newspaper-ocr library over the NewsBench pages.

Must be run with the newspaper-ocr project's interpreter, e.g.:

    /Users/nealcaren/Documents/GitHub/newspaper-ocr/.venv/bin/python \
        newsbench/run_newspaper_ocr.py --name tesseract-default

Writes one <id>.txt per page into newsbench/ocr-results/<name>/.
"""
import argparse, time, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="output folder name under ocr-results/")
    ap.add_argument("--recognizer", default="tesseract")
    ap.add_argument("--model", default=None, help="recognizer_model, e.g. news_combo_fast")
    ap.add_argument("--spell-check", action="store_true")
    ap.add_argument("--device", default=None,
                    help="force device for VLM backends: cpu/mps/cuda (mps can crash)")
    ap.add_argument("--limit", type=int, default=0, help="only first N images (debug)")
    args = ap.parse_args()

    from newspaper_ocr import Pipeline

    # VLM backends can be device-sensitive; build the recognizer object when a
    # device is pinned so we don't inherit the auto (mps) default that aborts.
    if args.device and args.recognizer in ("lightonocr", "paddleocr-vl", "glm-ocr"):
        if args.recognizer == "lightonocr":
            from newspaper_ocr.recognizers.lightonocr import LightOnOcrRecognizer
            rec = LightOnOcrRecognizer(device=args.device)
        elif args.recognizer == "paddleocr-vl":
            from newspaper_ocr.recognizers.paddleocr_vl import PaddleOcrVlRecognizer
            rec = PaddleOcrVlRecognizer(device=args.device)
        else:
            from newspaper_ocr.recognizers.glm_ocr import GlmOcrRecognizer
            rec = GlmOcrRecognizer(device=args.device)
        pipe = Pipeline(recognizer=rec, spell_check=args.spell_check)
    else:
        kw = {"recognizer": args.recognizer, "spell_check": args.spell_check}
        if args.model:
            kw["recognizer_model"] = args.model
        pipe = Pipeline(**kw)

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
            text = pipe.ocr(str(img))
        except Exception as e:  # keep going; record the failure
            text = ""
            print(f"[{i}/{len(imgs)}] ERROR {img.name}: {e}", flush=True)
        dst.write_text(text, encoding="utf-8")
        print(f"[{i}/{len(imgs)}] {img.name} {time.time()-t:.1f}s {len(text)}c", flush=True)

    print(f"done {len(imgs)} pages in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
