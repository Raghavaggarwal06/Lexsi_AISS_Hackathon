"""Run a round 5 script with the folders of round 6:   python scripts/r6_exec.py r5_judge2.py wg <tag> <set regex> <cell> ...
r6lib is imported first, so the script's paths (through r5lib) point at results/round6 and runs_private/round6."""
import runpy, sys
from pathlib import Path
import r6lib  # noqa: F401
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(str(Path(__file__).resolve().parent / script), run_name="__main__")
