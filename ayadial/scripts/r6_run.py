"""Round 6 spec runner: round 5's runner (scripts/r5_run.py) with the folders of round 6.   python scripts/r6_run.py specs6/<spec>.json
r6lib is imported first, so every path the runner uses points at results/round6 and runs_private/round6, and the closed suite frz5 is refused."""
import runpy, sys
from pathlib import Path
import r6lib  # noqa: F401  (repoints r4lib and r5lib before r5_run imports them)
runpy.run_path(str(Path(__file__).resolve().parent / "r5_run.py"), run_name="__main__")
