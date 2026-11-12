"""Round 7: every development-grid setting of the SafeTune methods on the 1910 WildJailbreak development rows (CPU, labels only).
  python scripts/r7_dev_grid.py   ->  results/round7/dev_grid.json
Per setting: harmful (1000 rows) refused / answered without harm / unsafe, harmless (910 rows) refusal, degenerate share over all rows; shares with the
report's rule (r7lib.shares_from). SafeTune was calibrated on the 7555 requests AyaDial's direction was computed from (journal J0, J2, J3).
RefusalDirection: the every-layer grid of J2 only (the layer-22 cells are void). Also Base from the same run, and any AyaDial-WSP / AyaDial sweep cell."""
import re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7lib as L7
rows = L7.suite_rows("wjx")
H = [L7.rk(r) for r in rows if r["set"] in L7.WJ_H]
Bk = [L7.rk(r) for r in rows if r["set"] in L7.WJ_B]
P = L7.J(L7.OUT / "safetune_pick.json")


def one(lab):
    h, b = L7.shares_from(lab, H), L7.shares_from(lab, Bk)
    return {"harmful": {"refused": h["refused_only"], "answered": h["answered"], "unsafe": h["unsafe"]}, "harmless": {"refusal": b["refusal"]},
            "degenerate": float(np.mean([lab[k][2] for k in H + Bk]) * 100)}


out = {"n_harmful": len(H), "n_harmless": len(Bk), "base": one(L7.J(L7.OUT / "labels" / "base_wjx.json")), "methods": {}}
grids = {"rd": L7.GRID["rd"], "adasteer": L7.GRID["adasteer"], "cast": L7.GRID["cast"], "wsp": [0, 0.5, 0.75, 1, 1.25, 1.5], "refv3g": [0.5, 0.75, 1, 1.25, 1.5]}
names = {**L7.METHODS, "wsp": "AyaDial-WSP", "refv3g": "AyaDial (written as an edit)"}
for arm, grid in grids.items():
    for s in grid:
        p = L7.OUT / "labels" / f"wjx.st_{arm}@{s:g}.json"
        if p.exists():
            lab = L7.J(p)
            if all(k in lab for k in H + Bk):
                out["methods"].setdefault(names[arm], {"picked": P["methods"].get(arm, {}).get("setting"), "settings": {}})["settings"][f"{s:g}"] = one(lab)
L7.write_json(L7.OUT / "dev_grid.json", out)
print("base", {k: round(v, 1) for k, v in out["base"]["harmful"].items()}, round(out["base"]["harmless"]["refusal"], 1))
for m, d in out["methods"].items():
    print(m, "picked", d["picked"], {s: (round(x["harmful"]["unsafe"], 1), round(x["harmless"]["refusal"], 1), round(x["degenerate"], 1)) for s, x in d["settings"].items()})
