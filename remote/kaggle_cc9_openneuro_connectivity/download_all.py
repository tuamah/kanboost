import time
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).with_name("cache") / "openneuro_data"
DATASET_ID = "ds004504"

subjects = [f"sub-{i:03d}" for i in range(1, 66)]

total_bytes = 0
ok, failed = 0, []
for pid in subjects:
    sub = pid.replace("sub-", "")
    dest = DATA_DIR / f"sub-{sub}" / "eeg"
    dest.mkdir(parents=True, exist_ok=True)
    fname = f"sub-{sub}_task-eyesclosed_eeg.set"
    out = dest / fname
    if out.exists() and out.stat().st_size > 0:
        total_bytes += out.stat().st_size
        ok += 1
        continue
    url = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}/sub-{sub}/eeg/{fname}"
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
