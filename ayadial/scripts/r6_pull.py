"""Mac side: unpack a round 6 results zip that came back through the Drive connector.   python scripts/r6_pull.py <connector result file or .zip>
Extracts results/round6/* and specs6/* only; never overwrites the documents that live on the Mac."""
import base64, io, json, sys, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
KEEP_LOCAL = {"results/round6/journal.md", "results/round6/decisions.md", "results/round6/STATE.md", "results/round6/summary.md", "results/round6/summary_prose.md", "results/round6/verification.md"}
raw = Path(sys.argv[1]).read_bytes()
if not sys.argv[1].endswith(".zip"):
    d = json.loads(raw)
    if isinstance(d, list):
        d = json.loads(d[0]["text"])
    raw = base64.b64decode(d["content"])
n = skipped = 0
for info in zipfile.ZipFile(io.BytesIO(raw)).infolist():
    name = info.filename
    p = (ROOT / name).resolve()
    ok = (name.startswith("results/round6/") or name.startswith("specs6/")) and str(p).startswith(str(ROOT)) and ".." not in name
    if info.is_dir():
        continue
    if not ok or name in KEEP_LOCAL:
        skipped += 1
        continue
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(zipfile.ZipFile(io.BytesIO(raw)).read(info))
    n += 1
print(f"extracted {n} files, skipped {skipped}")
