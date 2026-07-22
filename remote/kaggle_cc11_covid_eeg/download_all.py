import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "cache" / "eeg_data"
DATASET_ID = "ds007823"

participants = pd.read_csv(HERE / "participants.tsv", sep="\t")
subjects = participants["participant_id"].tolist()

total_bytes = 0
ok, failed = 0, []
for pid in subjects:
    dest = DATA_DIR / pid / "eeg"
    dest.mkdir(parents=True, exist_ok=True)
    fname = f"{pid}_task-COVID_eeg.edf"
    out = dest / fname
    if out.exists() and out.stat().st_size > 0:
        total_bytes += out.stat().st_size
        ok += 1
        continue
    url = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}/{pid}/eeg/{fname}"
    t0 = time.perf_counter()
    try:
        urllib.request.urlretrieve(url, out)
        sz = out.stat().st_size
        total_bytes += sz
        ok += 1
        print(f"{pid}: {sz/1e6:.1f}MB in {time.perf_counter()-t0:.1f}s")
    except urllib.error.HTTPError as exc:
        print(f"{pid}: FAILED {exc}")
        failed.append(pid)

print(f"\ndone: {ok}/{len(subjects)} ok, {len(failed)} failed, total {total_bytes/1e9:.2f} GB")
if failed:
    print("failed subjects:", failed)
