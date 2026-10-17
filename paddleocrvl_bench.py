"""Run the full PaddleOCR-VL pipeline (its own layout + reading order) whole-page."""
import argparse, time, re
from pathlib import Path
HERE = Path(__file__).resolve().parent

def md_text(res):
    md = getattr(res, "markdown", None)
    if isinstance(md, dict):
        t = md.get("markdown_texts")
        if isinstance(t, list): return "\n\n".join(str(x) for x in t)
        if t: return str(t)
    return ""

def strip_md(s):
    s = re.sub(r"`{3}.*?`{3}", " ", s, flags=re.S)
    s = re.sub(r"[#*_>|]+", " ", s)
    return re.sub(r"[ \t]+", " ", s).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from paddleocr import PaddleOCRVL
    print(f"loading PaddleOCR-VL ({args.device}) ...", flush=True)
    pipe = PaddleOCRVL(device=args.device)
    out = HERE / "ocr-results" / args.name; out.mkdir(parents=True, exist_ok=True)
    imgs = sorted((HERE / "images").glob("*.jpg"))
    if args.limit: imgs = imgs[:args.limit]
    for i, img in enumerate(imgs, 1):
        dst = out / (img.stem + ".txt")
        if dst.exists(): print(f"[{i}] skip {img.name}", flush=True); continue
        t = time.time()
        try:
            parts = [md_text(r) for r in pipe.predict(str(img))]
            txt = strip_md("\n\n".join(p for p in parts if p))
        except Exception as e:
            txt = ""; print(f"ERROR {img.name}: {type(e).__name__}: {e}", flush=True)
        dst.write_text(txt, encoding="utf-8")
        print(f"[{i}/{len(imgs)}] {img.name} {time.time()-t:.0f}s {len(txt)}c "
              f"head={txt[:70]!r}", flush=True)
main()
