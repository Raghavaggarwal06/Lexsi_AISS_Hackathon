"""Pack the round 6 results (ids, labels, numbers; no text) into a zip on Drive, after the leak check:   python scripts/r6_pack.py [all]
'all' also packs runs_private/round6 without the cells copied from round 5 (only files newer than the round 6 boot)."""
import subprocess, sys, time
from pathlib import Path
ROOT, D = Path("/content/build"), Path("/content/drive/MyDrive/lexsi-knob/round6")
tag = time.strftime("%H%M", time.gmtime())
chk = subprocess.run("python scripts/pipeline/check_no_harmful_text.py results/round6 specs6 scripts/r6*.py scripts/wp_*.py", shell=True, capture_output=True, text=True, cwd=ROOT)
last = chk.stdout.strip().splitlines()[-1] if chk.stdout.strip() else chk.stderr[-300:]
print("leak check:", last)
(ROOT / "results/round6/leak_check_last.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC ", time.gmtime()) + last + "\n")
if chk.returncode != 0:
    print(chk.stdout[-1500:]); sys.exit("leak check failed: results not packed")
subprocess.run(f"cp queue6.log logs6_*.txt runs_private/round6/ 2>/dev/null; rm -f /content/r6_results_{tag}.zip; "
               f"zip -q -r /content/r6_results_{tag}.zip results/round6 specs6 scripts/r6*.py scripts/wp_*.py && cp /content/r6_results_{tag}.zip {D}/", shell=True, cwd=ROOT)
print("results zip", (D / f"r6_results_{tag}.zip").stat().st_size, f"r6_results_{tag}.zip")
if len(sys.argv) > 1 and sys.argv[1] == "all":
    subprocess.run(f"rm -f /content/r6_private_{tag}.zip; find runs_private/round6 -type f -newer results/round6/sets_manifest.json | zip -q /content/r6_private_{tag}.zip -@ && zip -q /content/r6_private_{tag}.zip cache_r6/vec_*.npz cache_r6/wp_flag.npz 2>/dev/null; cp /content/r6_private_{tag}.zip {D}/", shell=True, cwd=ROOT)
    print("private zip", (D / f"r6_private_{tag}.zip").stat().st_size, f"r6_private_{tag}.zip")
