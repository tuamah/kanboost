"""
Downloads the ARGO dataset (open access, no credentialing, CC BY-NC-SA
4.0) from PhysioNet: signal (.dat/.hea) and annotation files for all
1962 records, plus the top-level metadata files. Skips the 3D
electroanatomical mesh files and the MATLAB duplicate (not needed for
per-segment EGM/ECG classification).

First attempt at 16 concurrent workers via bare urllib triggered SSL
"WRONG_VERSION_NUMBER" errors -- PhysioNet resetting connections under
too much concurrency, not a real protocol issue. Fixed with a shared
requests.Session (connection pooling/keep-alive), a small worker count,
and retry-with-backoff per file.
"""
import concurrent.futures
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://physionet.org/files/argo/1.0.0"
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "cache" / "argo_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SUFFIXES = [".dat", ".hea", ".annotation_ann1", ".annotation_ann2", ".annotation_ann3", ".annotation_consensus"]
TOP_LEVEL_FILES = ["Additional_subject_data.csv", "README.txt", "ANNOTATORS", "LICENSE.txt", "RECORDS"]
MAX_WORKERS = 6


def make_session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS, max_retries=retry)
    s.mount("https://", adapter)
    return s


def fetch(session, url, out_path):
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            return True
        except Exception as exc:
            if attempt == 3:
                print(f"FAILED {url}: {exc}")
                return False
            time.sleep(1.5 * (attempt + 1))
    return False


def main():
    session = make_session()
    for fname in TOP_LEVEL_FILES:
        fetch(session, f"{BASE_URL}/{fname}", DATA_DIR / fname)

    records = (DATA_DIR / "RECORDS").read_text().strip().splitlines()
    print(f"{len(records)} records to fetch, {len(records) * len(SUFFIXES)} files total")

    jobs = []
    for rec in records:
        for suf in SUFFIXES:
            url = f"{BASE_URL}/{rec}{suf}"
            out_path = DATA_DIR / f"{rec}{suf}"
            jobs.append((url, out_path))

    ok, failed = 0, []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch, session, url, out): (url, out) for url, out in jobs}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            url, out = futures[fut]
            if fut.result():
                ok += 1
            else:
                failed.append(url)
            if i % 500 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  {i}/{len(jobs)} done ({ok} ok, {len(failed)} failed) -- {elapsed:.0f}s elapsed, {i/elapsed:.1f} files/s")

    print(f"\ndone: {ok}/{len(jobs)} ok, {len(failed)} failed, {time.perf_counter()-t0:.0f}s total")
    if failed:
        print("first 10 failures:", failed[:10])


if __name__ == "__main__":
    main()
