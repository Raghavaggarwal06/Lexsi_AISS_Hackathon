"""The knob's cells on the 300 rows of the SafeTune screen, from existing label files, so that the baselines and the knob can be read in one table.
  python scripts/r5_screen_ref.py          (pure CPU; runtime or Mac)
The screen rows are a fixed sample of the 1910 WildJailbreak development rows (ids = the keys of labels/base_stscreen.json). Each cell's rates are
computed on those ids from its own label file; the knob's cells were generated with the wjx batching and are shown next to base_wjx on the same ids,
the SafeTune arms next to the screen's own base. Writes results/round5/safetune/knob_on_screen_rows.json (ids and rates only)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/round5"
WJ_H = ("wj_adversarial_harmful", "dev_ho_h", "dev_in_h")
lab = lambda n: json.loads((OUT / "labels" / f"{n}.json").read_text()) if (OUT / "labels" / f"{n}.json").exists() else None
scr = lab("base_stscreen")
if scr is None:
    sys.exit("no screen yet")
keys = sorted(scr)
H, B = [k for k in keys if k.split("|")[0] in WJ_H], [k for k in keys if k.split("|")[0] not in WJ_H]
rate = lambda L, ks, c: round(float(np.mean([L[k][c] for k in ks]) * 100), 1)
out = {"n": [len(H), len(B)], "note": "rates on the 300 SafeTune screen rows, from each cell's own label file; knob cells share the batching of base_wjx, not of the screen's base", "cells": {}}
for c in ("base_wjx", "ungated@0.1", "foldcap@1", "foldcap@1.25", "v2cap@1", "v3gcap@1", "lax3@2", "both3@1"):
    L = lab(c)
    if L is None or not all(k in L for k in keys):
        continue
    out["cells"][c] = {"harmful_refusal": rate(L, H, 0), "harmful_unsafe": rate(L, H, 1), "benign_refusal": rate(L, B, 0), "degenerate": rate(L, keys, 2)}
(OUT / "safetune").mkdir(exist_ok=True)
(OUT / "safetune" / "knob_on_screen_rows.json").write_text(json.dumps(out, indent=1))
for c, e in out["cells"].items():
    print(f"{c:14s} harmful refusal {e['harmful_refusal']:5.1f} unsafe {e['harmful_unsafe']:5.1f} | benign refusal {e['benign_refusal']:5.1f}")
