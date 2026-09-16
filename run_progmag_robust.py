#!/usr/bin/env python3
"""Robust page-by-page NewsBench runner for the progressive-magazines-ocr
pipeline. The local MLX GLM-OCR server can crash under sustained batch load;
once it's down, every region errors with 'connection refused' and whole pages
drop to ~0. This driver runs ONE page per pipeline invocation, verifies the MLX
server is alive before each (restarting it if not), and retries a page once if
too many regions errored — so a single server death can't corrupt the run.

Usage (from the progressive-magazines-ocr repo, so `uv run` resolves its deps):
  uv run --with img2pdf --with httpx python /path/newsbench/run_progmag_robust.py \
     --repo /path/progressive-magazines-ocr --name progmag-ppdoc-glm-repair
"""
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path
import httpx

BENCH = Path(__file__).parent
MLX_URL = "http://localhost:8080/v1/models"
MODEL = "mlx-community/GLM-OCR-bf16"

def server_up():
    try:
        return "GLM-OCR" in httpx.get(MLX_URL, timeout=3).text
    except Exception:
        return False

def ensure_server(repo):
    if server_up():
        return
    print("  [MLX server down -> restarting]", flush=True)
    subprocess.Popen(
        ["uv", "run", "--with", "mlx-vlm", "python", "-m", "mlx_vlm.server",
         "--model", MODEL, "--port", "8080"],
        cwd=str(repo), stdout=open("/tmp/mlx_server.log", "a"),
        stderr=subprocess.STDOUT)
    for _ in range(60):
        time.sleep(5)
        if server_up():
            print("  [MLX server back up]", flush=True); return
    raise RuntimeError("MLX server did not come back up")

def err_fraction(site, stem):
    pj = site / stem / "page_01.json"
    if not pj.exists():
        return 1.0
    R = json.load(open(pj))["regions"]
    if not R:
        return 1.0
    return sum(1 for r in R if r.get("status") == "error") / len(R)

def run_page(repo, pdf, site):
    subprocess.run(["uv", "run", str(repo / "ocr_newspapers.py"),
                    "--input-dir", str(pdf.parent), "--output-dir", str(site),
                    "--force", "--repair"],
                   cwd=str(repo), env=dict(os.environ, PYTHONUTF8="1"), check=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--name", default="progmag-ppdoc-glm-repair")
    ap.add_argument("--work", required=True)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    work = Path(args.work); work.mkdir(parents=True, exist_ok=True)

    import img2pdf
    imgs = sorted(glob.glob(str(BENCH / "images" / "*.jpg")))
    outdir = BENCH / "ocr-results" / args.name
    outdir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(imgs, 1):
        stem = Path(img).stem
        one = work / stem; one.mkdir(exist_ok=True)
        pdf = one / f"{stem}.pdf"
        if not pdf.exists():
            pdf.write_bytes(img2pdf.convert(img))
        site = one / "site"
        print(f"[{i}/{len(imgs)}] {stem}", flush=True)
        for attempt in (1, 2):
            ensure_server(repo)
            run_page(repo, pdf, site)
            ef = err_fraction(site, stem)
            if ef <= 0.15:
                break
            print(f"  [error fraction {ef:.0%} on attempt {attempt} -> restart+retry]", flush=True)
            time.sleep(3)
        # extract reading-order text
        ft = site / stem / "full_text.json"
        parts = []
        if ft.exists():
            d = json.load(open(ft))
            parts = [(r.get("text") or "").strip() for p in d.get("pages", [])
                     for r in p.get("regions", []) if (r.get("text") or "").strip()]
        (outdir / f"{stem}.txt").write_text("\n".join(parts))
        print(f"  -> {len(' '.join(parts).split())} words (err {err_fraction(site, stem):.0%})", flush=True)
    print(f"done -> {outdir}", flush=True)

if __name__ == "__main__":
    main()
