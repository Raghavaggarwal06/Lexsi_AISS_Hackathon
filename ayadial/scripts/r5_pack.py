"""Pack the round 5 results (ids, labels, numbers; no text) and, separately, the private runs, into zips on Drive.
  python scripts/r5_pack.py results     -> MyDrive/lexsi-knob/round5/r5_results_<HHMM>.zip   (small; safe to pull to the Mac at any time)
  python scripts/r5_pack.py all         -> also r5_private_<HHMM>.zip (prompts, replies, adapters, logs; private)
Runs the leak check on results/round5 first and refuses to pack results if it reports a hit."""
import subprocess, sys, time
from pathlib import Path
ROOT = Path("/content/build")
D = Path("/content/drive/MyDrive/lexsi-knob/round5")
tag = time.strftime("%H%M", time.gmtime())
chk = subprocess.run("python scripts/pipeline/check_no_harmful_text.py results/round5 specs5 scripts/r5*.py scripts/r5*.sh scripts/pipeline/r5*.py", shell=True, capture_output=True, text=True, cwd=ROOT)
last = chk.stdout.strip().splitlines()[-1] if chk.stdout.strip() else chk.stderr[-300:]
print("leak check:", last)
(ROOT / "results/round5/leak_check_last.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S UTC ", time.gmtime()) + last + "\n")
if chk.returncode != 0:
    print(chk.stdout[-1500:])
    sys.exit("leak check failed: results not packed")
subprocess.run(f"cp queue5.log logs5_*.txt runs_private/round5/ 2>/dev/null; rm -f /content/r5_results_{tag}.zip; zip -q -r /content/r5_results_{tag}.zip results/round5 scripts/r5*.py scripts/r5*.sh scripts/pipeline/r5*.py specs5 && cp /content/r5_results_{tag}.zip {D}/", shell=True, cwd=ROOT)
print("results zip", (D / f"r5_results_{tag}.zip").stat().st_size, f"r5_results_{tag}.zip")
if len(sys.argv) > 1 and sys.argv[1] == "all":
    subprocess.run(f"rm -f /content/r5_private_{tag}.zip; zip -q -r /content/r5_private_{tag}.zip runs_private/round5 cache_r5/vec_*.npz -x 'runs_private/round5/r4_reuse/*' && cp /content/r5_private_{tag}.zip {D}/", shell=True, cwd=ROOT)
    print("private zip", (D / f"r5_private_{tag}.zip").stat().st_size, f"r5_private_{tag}.zip")
