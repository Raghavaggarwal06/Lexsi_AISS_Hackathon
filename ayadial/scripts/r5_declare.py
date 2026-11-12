"""Write specs5/frozen_cells.json from results/round5/picks.json, so that the frozen settings come from the rules and not from a hand.
  python scripts/r5_declare.py          then: python scripts/r5_mkfrozen.py specs5/frozen_cells.json
Cells: the control at its two strict-side settings and at its lax-side pick; v1 at its two declared settings; v2 and v3g at their frozen settings;
the lax setting; the combined setting (v3g strict + lax). A side whose rule returns no setting is left out, and the reason is recorded."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
P = json.loads((ROOT / "results/round5/picks.json").read_text())
T, Lx = P["tightening"], P.get("loosening")
cells, roles, notes = [], {}, []


def add(c, role):
    cells.append(c); roles[c["name"]] = role


add({"name": "ungated@0.02", "f": "ungated", "scale": 0.02}, "control, strict side: its round 4 pick at the 2-point budget")
add({"name": "ungated@0.1", "f": "ungated", "scale": 0.1}, "control, strict side: its round 4 pick at the 5-point budget")
add({"name": "foldcap@1", "f": "v1", "s": 1}, "v1 (round 4 adapter files, unchanged), declared setting 1")
add({"name": "foldcap@1.25", "f": "v1", "s": 1.25}, "v1 (round 4 adapter files, unchanged), declared setting 2")
for method, vec, what in (("v2cap", "v2", "v2 = v1's recipe with GCG and PAIR pool artifacts and suffix transplants added to the read; v1's guard"),
                          ("v3gcap", "v3g", "v3g = v2's additions plus the 'a' half of the in-the-wild, MultiJail and Aya development slices in the read; guard refit on the same additions")):
    s = T.get(method, {}).get("frozen_setting")
    if s is None:
        notes.append(f"{method}: no setting passed the development rule; not in the frozen test")
        continue
    add({"name": f"{method}@{s:g}", "f": "strict", "vec": vec, "s": s}, f"new tightening candidate: {what}; setting {s:g} from the development rule (picks.json)")
s3 = T.get("v3gcap", {}).get("frozen_setting")
if Lx and Lx.get("lax_frozen_setting") is not None:
    t = Lx["lax_frozen_setting"]
    add({"name": f"lax3@{t:g}", "f": "lax", "vec": "v3g", "t": t, "theta": "v3g:h01"}, f"loosening: lax units on the v3g read (fires below the 1st percentile of harmful fit prompts), scale {t:g} from the mirror budget rule")
    if s3 is not None:
        add({"name": f"both3@{s3:g}", "f": "both", "svec": "v3g", "s": s3, "lvec": "v3g", "t": t, "theta": "v3g:h01"}, f"combined: v3g strict at {s3:g} and the lax units at {t:g}, installed together")
else:
    notes.append("loosening: no lax setting passed the mirror budget rule on development data; no lax cell and no combined cell")
if Lx and Lx.get("control_frozen_setting") is not None:
    c = Lx["control_frozen_setting"]
    add({"name": f"ctrlneg@{c:g}", "f": "ungated", "scale": c}, "control, lax side: the ungated push at negative scale, its pick under the mirror budget rule")
else:
    add({"name": "ctrlneg@-0.05", "f": "ungated", "scale": -0.05}, "control, lax side: the ungated push at -0.05 (no negative setting passed the mirror rule; this is the mildest setting run on every development set)")
# most informative cells first, so that a runtime lost half way leaves the headline comparisons complete
ORDER = ["v3gcap@", "ungated@0.1", "foldcap@1", "lax3@", "both3@", "v2cap@", "ctrlneg@", "foldcap@1.25", "ungated@0.02"]
rank = lambda n: next((i for i, p_ in enumerate(ORDER) if n == p_ or (p_.endswith("@") and n.startswith(p_))), len(ORDER))
cells.sort(key=lambda c: rank(c["name"]))
out = {"rule": "settings fixed on development data by the rules of journal J12 (tightening) and the charter's mirror budget rule (loosening), computed by scripts/r5_picks.py; base is generated with the same batching; every cell is compared with base and with the control of its side",
       "cells": cells, "roles": roles, "notes": notes}
(ROOT / "specs5/frozen_cells.json").write_text(json.dumps(out, indent=1))
print(json.dumps({"cells": [c["name"] for c in cells], "notes": notes}, indent=1))
