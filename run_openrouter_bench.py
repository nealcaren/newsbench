#!/usr/bin/env python
"""Benchmark an OpenRouter VLM over the NewsBench pages, two layout configs.

Run with the newspaper-ocr interpreter (has httpx + paddlex):

    NPO=/Users/nealcaren/Documents/GitHub/newspaper-ocr/.venv/bin/python
    export OPENROUTER_API_KEY=sk-or-...

    # (1) no segmentation — whole page image in one call
    $NPO run_openrouter_bench.py --model deepseek/deepseek-v4.1-flash --config wholepage

    # (2) PaddleX / PP-DocLayout — one call per detected region
    $NPO run_openrouter_bench.py --model deepseek/deepseek-v4.1-flash --config paddlex

Output folder name defaults to a sanitized "<model-tail>-<config>" under
ocr-results/. Writes one <id>.txt per page plus a `_cost.json` with the real
OpenRouter spend (usage.include) and token totals for the run.
"""
import argparse, json, time, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"

# Whole-page multi-column pages need reading-order guidance; region crops don't.
WHOLEPAGE_PROMPT = (
    "This is a scan of a full historical newspaper page with multiple columns. "
    "Transcribe ALL of the text exactly as printed, proceeding column by column "
    "in natural reading order (each column top-to-bottom, left column first). "
    "Include every article, headline, and advertisement. Output only the "
    "transcription text, no commentary."
)


def jpeg_openrouter(**kw):
    """OpenRouterRecognizer that uploads JPEG, not PNG.

    The stock recognizer re-encodes every crop as lossless PNG. For a whole
    newspaper page that is ~19MB (25MB base64) — over Anthropic's 5MB image
    limit, so claude-* silently 400s on every page. The source scans are already
    JPEG, so JPEG q92 is the same pixels at ~1/5 the bytes; this benefits every
    provider (far less upload) and unblocks the size-capped ones.
    """
    import io, base64
    from newspaper_ocr.recognizers.openai_compat import OpenRouterRecognizer

    import threading

    class JpegOpenRouter(OpenRouterRecognizer):
        JPEG_QUALITY = 92
        _usage_lock = threading.Lock()

        def _record_usage(self, usage):
            # pages run concurrently -> serialize the += into shared totals
            with self._usage_lock:
                super()._record_usage(usage)

        def _recognize_api(self, image):
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="JPEG", quality=self.JPEG_QUALITY)
            b64 = base64.b64encode(buf.getvalue()).decode()

            def _post():
                body = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": self.prompt},
                    ]}],
                    self._token_param: self.max_tokens,
                    **self.extra_body,
                }
                return self._client.post(self.endpoint, headers=self._headers(), json=body)

            resp = _post()
            if (resp.status_code == 400 and not self._token_param_locked
                    and "max_completion_tokens" in resp.text):
                self._token_param = ("max_completion_tokens"
                                     if self._token_param == "max_tokens" else "max_tokens")
                self._token_param_locked = True
                resp = _post()
            resp.raise_for_status()
            data = resp.json()
            self._record_usage(data.get("usage"))
            return (data["choices"][0]["message"].get("content") or "").strip()

    return JpegOpenRouter(**kw)


def whole_page_detector():
    """A Detector that returns the whole image as a single region (no segmentation)."""
    from newspaper_ocr.detectors.base import Detector
    from newspaper_ocr.models import PageLayout, Region, BBox

    class WholePageDetector(Detector):
        def detect(self, image):
            w, h = image.size
            region = Region(bbox=BBox(0, 0, w, h), image=image, label="text")
            return PageLayout(image=image, regions=[region],
                              width=w, height=h, lines_detected=False)

    return WholePageDetector()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="OpenRouter model id, e.g. deepseek/deepseek-v4.1-flash")
    ap.add_argument("--config", required=True, choices=["wholepage", "paddlex"])
    ap.add_argument("--name", default=None, help="output folder (default: <model-tail>-<config>)")
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="output token cap; whole broadsheets need ~12k+")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--limit", type=int, default=0, help="only first N pages (smoke test)")
    ap.add_argument("--pages", default=None, help="comma-separated stems to run (smoke test)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="pages processed in parallel (network-bound; big win for API backends)")
    args = ap.parse_args()

    from newspaper_ocr import Pipeline

    rec_kw = dict(
        model=args.model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        extra_body={"usage": {"include": True}},   # ask OpenRouter to bill us back the real cost
        extra_headers={"X-Title": "newsbench"},
    )
    # Whole-page needs reading-order guidance; paddlex region crops keep the default prompt.
    if args.config == "wholepage":
        rec_kw["prompt"] = WHOLEPAGE_PROMPT
    rec = jpeg_openrouter(**rec_kw)
    if args.config == "wholepage":
        pipe = Pipeline(detector=whole_page_detector(), recognizer=rec,
                        layout_processing=False, text_cleaning=False)
    else:
        pipe = Pipeline(detector="paddlex", recognizer=rec)

    tail = re.sub(r"[^0-9a-zA-Z._-]", "-", args.model.split("/")[-1])
    name = args.name or f"{tail}-{args.config}"
    out = HERE / "ocr-results" / name
    out.mkdir(parents=True, exist_ok=True)

    imgs = sorted(IMAGES.glob("*.jpg"))
    if args.pages:
        want = set(args.pages.split(","))
        imgs = [p for p in imgs if p.stem in want]
    if args.limit:
        imgs = imgs[: args.limit]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    print_lock = threading.Lock()
    done = [0]
    todo = [img for img in imgs if not (out / (img.stem + ".txt")).exists()]
    print(f"{len(imgs)} pages ({len(imgs)-len(todo)} already done), "
          f"concurrency={args.concurrency}", flush=True)

    def work(img):
        t = time.time()
        try:
            text = pipe.ocr(str(img))
        except Exception as e:
            text = ""
            with print_lock:
                print(f"ERROR {img.name}: {e}", flush=True)
        (out / (img.stem + ".txt")).write_text(text, encoding="utf-8")
        with print_lock:
            done[0] += 1
            print(f"[{done[0]}/{len(todo)}] {img.name} {time.time()-t:.1f}s {len(text)}c  "
                  f"cum ${rec.usage_totals.get('reported_cost', 0.0):.4f}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        list(as_completed([ex.submit(work, img) for img in todo]))

    # Persist the money + token accounting for this run.
    summary = {
        "model": args.model,
        "config": args.config,
        "pages": len(imgs),
        "elapsed_s": round(time.time() - t0, 1),
        **rec.usage_totals,
    }
    (out / "_cost.json").write_text(json.dumps(summary, indent=2))
    print(f"done {len(imgs)} pages in {summary['elapsed_s']}s  "
          f"reported_cost=${summary['reported_cost']:.4f}  "
          f"tokens in/out={int(summary['prompt_tokens'])}/{int(summary['completion_tokens'])} "
          f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
