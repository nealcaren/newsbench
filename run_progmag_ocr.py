#!/usr/bin/env python3
"""Score the progressive-magazines-ocr pipeline (PP-DocLayout_plus-L layout +
GLM-OCR recognition + column reading-order + region-repair) on NewsBench.

Our pipeline ingests PDFs, so each NewsBench JPG is losslessly wrapped in a
1-page PDF (img2pdf embeds the original JPEG bytes; the pipeline extracts the
largest embedded image back at native resolution). Per-page region text is then
concatenated in reading order and written to ocr-results/<name>/<id>.txt.

Run from the progressive-magazines-ocr repo's uv environment, with the MLX
GLM-OCR server already up (localhost:8080). Example:
  uv run --with img2pdf python /path/to/newsbench/run_progmag_ocr.py \
      --repo /path/to/progressive-magazines-ocr --name progmag-ppdoc-glm-repair
"""
import argparse, glob, json, os, subprocess, sys, tempfile
from pathlib import Path

BENCH = Path(__file__).parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to progressive-magazines-ocr")
    ap.add_argument("--name", default="progmag-ppdoc-glm-repair")
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--work", default=None, help="scratch dir (default: temp)")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="nb_progmag_"))
    pdfs, site = work / "pdfs", work / "site"
    pdfs.mkdir(parents=True, exist_ok=True); site.mkdir(parents=True, exist_ok=True)

    # 1. wrap JPGs -> 1-page PDFs (lossless)
    import img2pdf
    imgs = sorted(glob.glob(str(BENCH / "images" / "*.jpg")))
    for f in imgs:
        stem = Path(f).stem
        out = pdfs / f"{stem}.pdf"
        if not out.exists():
            out.write_bytes(img2pdf.convert(f))
    print(f"wrapped {len(imgs)} images -> {pdfs}", flush=True)

    # 2. run our pipeline (via `uv run` so its PEP-723 inline deps resolve)
    cmd = ["uv", "run", str(repo / "ocr_newspapers.py"),
           "--input-dir", str(pdfs), "--output-dir", str(site)]
    if not args.no_repair:
        cmd.append("--repair")
    print("RUN:", " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONUTF8="1")
    subprocess.run(cmd, cwd=str(repo), env=env, check=True)

    # 3. extract per-page reading-order text -> ocr-results/<name>/<id>.txt
    outdir = BENCH / "ocr-results" / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for ft in sorted(glob.glob(str(site / "*" / "full_text.json"))):
        stem = Path(ft).parent.name
        d = json.load(open(ft))
        parts = []
        for p in d.get("pages", []):
            for r in p.get("regions", []):
                t = (r.get("text") or "").strip()
                if t:
                    parts.append(t)
        (outdir / f"{stem}.txt").write_text("\n".join(parts))
        n += 1
    print(f"wrote {n} transcriptions -> {outdir}", flush=True)

if __name__ == "__main__":
    main()
