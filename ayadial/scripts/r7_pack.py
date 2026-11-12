"""Pack round 7's results (ids, labels, numbers; no text) into a zip on Drive after the leak check:   python scripts/r7_pack.py
Writes results/round7/leak_check_last.txt and MyDrive/lexsi-knob/round7/r7_results_<HHMM>.zip (small; safe to pull to the Mac). Replies are not
packed: they stay in runs_private/round7, which r7lib.sync mirrors to MyDrive/lexsi-knob/round7/runs_private/."""
import subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7lib as L7

ROOT, D = L7.ROOT, L7.DRIVE
tag = time.strftime("%H%M", time.gmtime())
chk = subprocess.run("python scripts/pipeline/check_no_harmful_text.py results/round7 scripts/r7lib.py scripts/r7gpu.py scripts/r7_safetune.py scripts/r7_pack.py scripts/r7_boot.sh scripts/r7_queue.sh",
                     shell=True, capture_output=True, text=True, cwd=ROOT)
last = chk.stdout.strip().splitlines()[-1] if chk.stdout.strip() else chk.stderr[-300:]
print("leak check:", last)
(L7.OUT / "leak_check_last.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC ", time.gmtime()) + last + "\n")
if chk.returncode != 0:
    print(chk.stdout[-1500:])
    sys.exit("leak check failed: results not packed")
z = Path(f"/content/r7_results_{tag}.zip")
subprocess.run(f"rm -f {z}; zip -q -r {z} results/round7 scripts/r7lib.py scripts/r7gpu.py scripts/r7_safetune.py scripts/r7_pack.py scripts/r7_boot.sh scripts/r7_queue.sh && cp {z} {D}/", shell=True, cwd=ROOT)
print("results zip", (D / z.name).stat().st_size if (D / z.name).exists() else "not copied", z.name)
L7.sync()
print("R7_PACK DONE")
