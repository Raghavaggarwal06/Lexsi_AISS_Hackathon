"""Build every table of the round from the label files and picks.json, in one go. Pure CPU; runs on the runtime or on the Mac.
  python scripts/r5_tables.py [dev] [frozen]         (default: both; a table whose cells do not exist yet is skipped)
Writes results/round5/tables/*.json and *.md through r5_report.py."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r5_report as R

OUT = R.OUT
P = json.loads((OUT / "picks.json").read_text()) if (OUT / "picks.json").exists() else {"tightening": {}}
T = P["tightening"]
fs = lambda m: T.get(m, {}).get("frozen_setting")
V2 = f"v2cap@{fs('v2cap'):g}" if fs("v2cap") is not None else None
V3 = f"v3gcap@{fs('v3gcap'):g}" if fs("v3gcap") is not None else None
Lx = P.get("loosening") or {}
LAX = f"lax3@{Lx['lax_frozen_setting']:g}" if Lx.get("lax_frozen_setting") is not None else None
NEG = f"ctrlneg@{Lx['control_frozen_setting']:g}" if Lx.get("control_frozen_setting") is not None else "ctrlneg@-0.05"
BOTH = f"both3@{fs('v3gcap'):g}" if (LAX and fs("v3gcap") is not None) else None
CTRL = "ungated@0.1"
have = set(R.load_labels())
want = [a for a in sys.argv[1:] if not a.startswith("--")] or ["dev", "frozen"]


def tab(name, side, cells, control, **kw):
    cells = [c for c in cells if c and c in have]
    if not cells:
        print(f"skip {name}: no cells")
        return None
    print(f"--- {name}")
    return R.table(name, side, cells, control=control if control in have else None, **kw)


if "dev" in want:
    tab("dev_v1", "dev", [CTRL, "ungated@0.02", "foldcap@1", "foldcap@1.25"], CTRL)
    tight = [CTRL, "foldcap@1", "foldcap@1.25", V2, V3]
    if tab("dev_main", "dev", tight, CTRL):
        tab("dev_main_b", "dev", tight, CTRL, half="b")
        for c in ("foldcap@1", "foldcap@1.25", V2, V3):
            if c and c in have:
                R.claims("dev_main_b", c)
                R.claims("dev_v1", c) if c.startswith("foldcap") else None
    for m, n in (("foldcap", "board_v1"), ("v2cap", "board_v2"), ("v3gcap", "board_v3g")):
        if any(k.startswith(m + "@") for k in have):
            R.board(n, m, "ungated")
    lax_cells = [NEG, "laxh@1", "laxh3@1", "lax3@1", "lax3@2", BOTH]
    if tab("dev_lax", "dev", lax_cells, NEG):
        tab("dev_lax_b", "dev", lax_cells, NEG, half="b")
    if any(k.startswith("lax3@") for k in have) and any(k.startswith("ctrlneg@") for k in have):
        try:
            R.board_lax("laxboard", "lax3", "ctrlneg")
        except Exception as e:
            print("laxboard:", type(e).__name__, e)
if "frozen" in want and "base_frz5" in have:
    tight = [CTRL, "ungated@0.02", "foldcap@1", "foldcap@1.25", V2, V3]
    for judge, suf in ((None, ""), ("wg", "_wg")):
        if judge and not (OUT / "labels_wg" / "base_frz5.json").exists():  # second-judge label files live in labels_wg/
            continue
        if tab(f"frz_tight{suf}", "frz", tight, CTRL, judge=judge):
            for c in ("foldcap@1", "foldcap@1.25", V2, V3):
                if c:
                    try:
                        R.claims(f"frz_tight{suf}", c)
                    except Exception as e:
                        print("claims", c, type(e).__name__, e)
        tab(f"frz_lax{suf}", "frz", [NEG, LAX, BOTH], NEG, judge=judge)
        if BOTH and tab(f"frz_both{suf}", "frz", [CTRL, V3, BOTH], CTRL, judge=judge):  # the combined setting on the tightening rule
            try:
                R.claims(f"frz_both{suf}", BOTH)
            except Exception as e:
                print("claims both", type(e).__name__, e)
    if (OUT / "labels_q8" / "base_frz5.json").exists():  # Qwen3Guard-Gen-8B on the rows in other languages (MultiJail, Aya, translated SORRY-Bench styles)
        if tab("frz_tight_q8", "frz", [CTRL, "foldcap@1", V3, BOTH], CTRL, judge="q8"):
            for c in ("foldcap@1", V3, BOTH):
                try:
                    R.claims("frz_tight_q8", c)
                except Exception as e:
                    print("claims q8", c, type(e).__name__, e)
        tab("frz_lax_q8", "frz", [LAX, BOTH], None, judge="q8")
json.dump({"control": CTRL, "v2": V2, "v3g": V3, "lax": LAX, "control_negative": NEG, "both": BOTH}, open(OUT / "tables" / "cells_used.json", "w"), indent=1)
print("cells:", CTRL, V2, V3, LAX, NEG, BOTH)
