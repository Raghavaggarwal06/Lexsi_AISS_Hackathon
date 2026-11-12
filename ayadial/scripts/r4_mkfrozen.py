
"""Fix the frozen-test operating points from the development scoreboard and write the spec. No frozen data is read here.
  python r4_mkfrozen.py main            -> specs/frozen1.json (base, ungated at its budget-2 and budget-5 picks, foldcap at its two picks)
  python r4_mkfrozen.py lora <method>   -> specs/frozen2.json (the trained LoRA at its budget-5 pick, if it has one)"""
import sys
from r4lib import *
sbd = json.loads((OUT / "scoreboard_dev.json").read_text())
decl_p = OUT / "frozen_declaration.json"
decl = json.loads(decl_p.read_text()) if decl_p.exists() else {}
E = "frozen ids sha256 " + json.loads((OUT / "sets_manifest.json").read_text())["frozen_ids_sha256"]
if sys.argv[1] == "main":
    b = sbd["boards"]["foldcap"]
    picks = {"foldcap_budget2": b["2"]["method_pick"]["setting"], "foldcap_budget5": b["5"]["method_pick"]["setting"], "ungated_budget2": b["2"]["control_pick"]["setting"], "ungated_budget5": b["5"]["control_pick"]["setting"]}
    cells = []
    for s in sorted({picks["ungated_budget2"], picks["ungated_budget5"]} - {None}):
        cells.append({"name": f"ungated@{s:g}", "f": "ungated", "scale": s, "pos": "all"})
    for s in sorted({picks["foldcap_budget2"], picks["foldcap_budget5"]} - {None}):
        cells.append({"name": f"foldcap@{s:g}", "f": "fold", "scale": s, "gain": 1.0, "cap_beta": 4.0})
    decl["main"] = {"written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "rule": "settings are the development scoreboard picks at budgets 2 and 5 (all development prompts)", "picks": picks, "cells": [c["name"] for c in cells], "frozen": E,
                    "dev_ungated_grid": sorted(float(n.split("@")[1]) for n in sbd["rates"] if n.startswith("ungated@")), "dev_foldcap_grid": sorted(float(n.split("@")[1]) for n in sbd["rates"] if n.startswith("foldcap@"))}
    (ROOT / "specs/frozen1.json").write_text(json.dumps({"rows": "frozen", "base": "base_frozen", "cells": cells, "boards": []}, indent=1))
else:
    m = sys.argv[2]
    s = sbd["boards"][m]["5"]["method_pick"]["setting"]
    decl["lora"] = {"written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "method": m, "rule": "budget-5 development pick; not opened if no setting is inside the budget", "pick": s}
    cells = [] if s is None else [{"name": f"{m}@{s:g}", "f": "lora", "scale": s}]
    (ROOT / "specs/frozen2.json").write_text(json.dumps({"rows": "frozen", "base": "base_frozen", "lora": sys.argv[3], "cells": cells, "boards": []}, indent=1))
decl_p.write_text(json.dumps(decl, indent=1))
print(json.dumps(decl, indent=1))
sync()
