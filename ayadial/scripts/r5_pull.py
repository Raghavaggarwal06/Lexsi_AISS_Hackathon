"""Mac side: unpack a results zip that came back from Drive through the Drive connector.
  python scripts/r5_pull.py <file with the connector's JSON result, or a .zip>
The connector returns {"content": <base64>, "title": ...}. The zip is extracted into build/, except the documents that live on the Mac
(journal.md, decisions.md, STATE.md, summary.md) and anything outside results/round5, scripts/ and specs5/. Zip paths are checked before extraction."""
import base64, io, json, sys, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
KEEP_LOCAL = {"results/round5/journal.md", "results/round5/decisions.md", "results/round5/STATE.md", "results/round5/summary.md", "results/round5/summary_prose.md", "results/round5/verification.md"}
src = Path(sys.argv[1])
raw = src.read_bytes()
if src.suffix != ".zip":
    d = json.loads(raw)
    if isinstance(d, list):  # some harnesses wrap the result as [{"type": "text", "text": "<json>"}]
        d = json.loads(d[0]["text"])
    raw = base64.b64decode(d["content"])
z = zipfile.ZipFile(io.BytesIO(raw))
n = skipped = 0
for info in z.infolist():
    name = info.filename
    p = (ROOT / name).resolve()
    ok = (name.startswith("results/round5/") or (name.startswith("scripts/") and "/r5" in "/" + name.split("/", 1)[1]) or name.startswith("specs5/")) and str(p).startswith(str(ROOT)) and ".." not in name
    if info.is_dir():
        continue
    if not ok or name in KEEP_LOCAL or name.startswith("scripts/"):
        skipped += 1
        continue
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(z.read(info))
    n += 1
print(f"extracted {n} files, skipped {skipped} (Mac-side documents, scripts and anything outside results/round5 and specs5)")
