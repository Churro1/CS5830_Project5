#!/usr/bin/env python3
"""
Extract dominant colors, a text-color guess, contrast, and detail/frequency
features from book cover images listed in a CSV.

Usage:
    python cover_features.py books.csv out.csv --isbn-col ISBN --url-col Image-URL-M
    python cover_features.py books.csv out.csv --limit 200      # quick trial run

Re-running resumes: ISBNs already present in the output are skipped.
Downloaded images are cached in ./cover_cache so re-runs don't re-download.

pip install pandas numpy pillow requests scikit-learn scikit-image scipy
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # we parallelise ourselves

import argparse
import csv
import hashlib
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy import ndimage
from skimage import color
from sklearn.cluster import KMeans

WORK_SIZE = (100, 150)  # every cover is resized to this so features are comparable


# --------------------------------------------------------------------- loading
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def make_session(referer=None):
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    s.headers.update({
        "User-Agent": BROWSER_UA,
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    if referer:
        s.headers["Referer"] = referer
    retry = Retry(total=3, backoff_factor=1.5, respect_retry_after_header=True,
                  status_forcelist=[429, 500, 502, 503, 504])
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def load_image(url, cache_dir, session):
    p = cache_dir / (hashlib.md5(url.encode()).hexdigest() + ".img")
    if p.exists():
        data = p.read_bytes()
    else:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        data = r.content
        p.write_bytes(data)
    img = Image.open(io.BytesIO(data))
    img.load()
    # Amazon-style "no cover" responses are often a 1x1 gif
    if img.width < 20 or img.height < 20:
        raise ValueError(f"placeholder image ({img.width}x{img.height})")
    return img.convert("RGB")


# -------------------------------------------------------------------- features
def lab_to_hex(lab):
    rgb = color.lab2rgb(np.asarray(lab, dtype=float).reshape(1, 1, 3)).reshape(3)
    r, g, b = (np.clip(rgb, 0, 1) * 255).round().astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def analyze(img, k=6, n_colors=3, merge_de=12.0):
    img = img.resize(WORK_SIZE, Image.LANCZOS)
    rgb = np.asarray(img, dtype=float) / 255.0
    h, w, _ = rgb.shape
    lab = color.rgb2lab(rgb)
    L = lab[..., 0]
    gray = L / 100.0

    out = {}

    # ---- dominant colors: k-means in Lab, then merge perceptually close clusters
    flat = lab.reshape(-1, 3)
    km = KMeans(k, n_init=4, random_state=0).fit(flat)
    labels = km.labels_
    centers = km.cluster_centers_
    frac = np.bincount(labels, minlength=k) / len(labels)

    merged = []  # [center, weight]
    for i in np.argsort(-frac):
        for m in merged:
            if np.linalg.norm(m[0] - centers[i]) < merge_de:
                m[0] = (m[0] * m[1] + centers[i] * frac[i]) / (m[1] + frac[i])
                m[1] += frac[i]
                break
        else:
            merged.append([centers[i].copy(), frac[i]])
    merged.sort(key=lambda m: -m[1])
    for n in range(n_colors):
        if n < len(merged):
            out[f"color{n+1}_hex"] = lab_to_hex(merged[n][0])
            out[f"color{n+1}_pct"] = round(float(merged[n][1]), 4)
        else:
            out[f"color{n+1}_hex"], out[f"color{n+1}_pct"] = "", 0.0

    return out


# ------------------------------------------------------------------------ main
_local = threading.local()


def process(row_id, url, cache_dir, referer, delay):
    if not hasattr(_local, "session"):
        _local.session = make_session(referer)
    session = _local.session
    try:
        if delay:
            time.sleep(delay)
        feats = analyze(load_image(url, cache_dir, session))
        return {"isbn": row_id, "error": "", **feats}
    except Exception as e:
        return {"isbn": row_id, "error": f"{type(e).__name__}: {e}"[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--isbn-col", default="ISBN")
    ap.add_argument("--url-col", default="Image-URL-M",
                    help="Medium/large is better than small for text + detail features")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--referer", default=None,
                    help="e.g. https://www.amazon.com/ if the host does hotlink checks")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds each worker sleeps before a request (throttle)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache-dir", default="cover_cache")
    a = ap.parse_args()

    cache = Path(a.cache_dir)
    cache.mkdir(exist_ok=True)

    df = pd.read_csv(a.input_csv, dtype=str, encoding_errors="replace", sep=';', encoding='latin1', escapechar='\\')  # dtype=str keeps ISBN leading zeros
    df = df.dropna(subset=[a.isbn_col, a.url_col]).drop_duplicates(a.isbn_col)

    out_path = Path(a.output_csv)
    need_header = not out_path.exists()
    done = set()
    if out_path.exists():
        old = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        ok = old[old["error"] == ""]
        if len(ok) < len(old):  # rewrite file without failed rows so they get retried
            ok.to_csv(out_path, index=False)
        done = set(ok["isbn"])
        df = df[~df[a.isbn_col].isin(done)]
    if a.limit:
        df = df.head(a.limit)
    print(f"{len(df)} to process ({len(done)} already done)")

    fields = None
    f = open(out_path, "a", newline="", encoding="utf-8")
    writer = None
    n_done = n_err = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(process, i, u, cache, a.referer, a.delay)
                for i, u in zip(df[a.isbn_col], df[a.url_col])]
        for fut in as_completed(futs):
            row = fut.result()
            if writer is None:
                # fixed column order, even if the first row is an error
                fields = ["isbn", "error", "color1_hex", "color1_pct", "color2_hex",
                          "color2_pct", "color3_hex", "color3_pct"]
                writer = csv.DictWriter(f, fields, restval="")
                if need_header:
                    writer.writeheader()
            writer.writerow(row)
            if row["error"] and n_err < 3:
                print("  sample error:", row["isbn"], row["error"])
            n_done += 1
            n_err += bool(row["error"])
            if n_done % 500 == 0:
                f.flush()
                print(f"{n_done}/{len(df)}  errors={n_err}")
    f.close()
    print(f"finished: {n_done} rows, {n_err} errors -> {out_path}")


if __name__ == "__main__":
    main()