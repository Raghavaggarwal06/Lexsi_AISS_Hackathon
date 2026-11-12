"""Declare the frozen test before any frozen prompt is generated: python scripts/r5_mkfrozen.py specs5/frozen_cells.json
Input: {"cells": [...cell specs as for r5_run.py...], "roles": {"<cell name>": "<why it is here>"}, "rule": "..."}.
Refuses if a frozen cell already exists. Writes results/round5/frozen_declaration.json (cells, roles, the frozen hash, time) and specs5/frozen.json."""
import sys
from r5lib import *

src = json.loads(Path(sys.argv[1]).read_text())
man = json.loads((OUT / "sets_manifest.json").read_text())
rows = suite_rows("frz5")
keys = sorted(rk(r) for r in rows)
assert hashlib.sha256("\n".join(keys).encode()).hexdigest() == man["frozen_ids_sha256"], "the frozen rows on the runtime do not match the recorded hash"
frozen_started = (RAW / "gen" / "base_frz5.jsonl").exists()
assert not frozen_started, "the frozen base already exists: the declaration must come first"
limits = {"tightening candidates beyond v1 (at most 2)": [c["name"] for c in src["cells"] if src["roles"].get(c["name"], "").startswith("new tightening")],
          "loosening settings (at most 1)": [c["name"] for c in src["cells"] if src["roles"].get(c["name"], "").startswith("loosening")],
          "combined settings (at most 1)": [c["name"] for c in src["cells"] if src["roles"].get(c["name"], "").startswith("combined")]}
assert len(limits["tightening candidates beyond v1 (at most 2)"]) <= 2 and len(limits["loosening settings (at most 1)"]) <= 1 and len(limits["combined settings (at most 1)"]) <= 1, limits
decl = {"written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "frozen_n": len(keys), "frozen_ids_sha256": man["frozen_ids_sha256"], "rule": src["rule"],
        "cells": [{"name": c["name"], "role": src["roles"][c["name"]], "spec": {k: v for k, v in c.items() if k != "name"}} for c in src["cells"]], "limits": limits,
        "vector_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(CACHE.glob("vec_*.npz"))}}
(OUT / "frozen_declaration.json").write_text(json.dumps(decl, indent=1))
(ROOT / "specs5" / "frozen.json").write_text(json.dumps({"suite": "frz5", "cells": src["cells"]}, indent=1))
print(json.dumps(decl, indent=1))
sync()
