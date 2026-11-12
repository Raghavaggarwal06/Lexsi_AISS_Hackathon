"""Are SafeTune's CAST and AdaSteer cells the same replies? They print the same three rates on the 300-prompt screen. Pure CPU, on the runtime.
  python scripts/r5_cast_check.py
Compares label files and generated replies (text is read, only counts are written). Writes results/round5/safetune/cast_vs_adasteer.json."""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "runs_private/round5", ROOT / "results/round5"
lab = lambda n: json.loads((OUT / "labels" / f"{n}.json").read_text())
gen = lambda n: {f"{r['set']}|{r['id']}": r["response"] for r in (json.loads(l) for l in (RAW / "gen" / f"{n}.jsonl").read_text().split("\n") if l.strip())}
names = {"adasteer": "stscreen.st_adasteer@4", "adasteer_8": "stscreen.st_adasteer@8", "cast_0.5": "stscreen8.st_cast@0.5", "cast_1": "stscreen8.st_cast@1", "cast_2": "stscreen8.st_cast@2", "base_screen": "base_stscreen", "base_screen8": "base_stscreen8"}
L, G = {k: lab(v) for k, v in names.items()}, {k: gen(v) for k, v in names.items()}
ks = sorted(L["adasteer"])
pair = lambda a, b: {"n": len(ks), "refusal_labels_agree_pct": round(100 * float(np.mean([L[a][k][0] == L[b][k][0] for k in ks])), 1), "unsafe_labels_agree_pct": round(100 * float(np.mean([L[a][k][1] == L[b][k][1] for k in ks])), 1),
                     "identical_replies_pct": round(100 * float(np.mean([G[a][k] == G[b][k] for k in ks])), 1)}
out = {"what": "pairwise comparison of cells on the 300 SafeTune screen prompts: share of equal refusal labels, equal unsafe labels and character-identical replies",
       "pairs": {"adasteer vs cast_0.5": pair("adasteer", "cast_0.5"), "adasteer vs adasteer_8": pair("adasteer", "adasteer_8"), "cast_0.5 vs cast_1": pair("cast_0.5", "cast_1"), "cast_0.5 vs base_screen8": pair("cast_0.5", "base_screen8"),
                 "cast_2 vs base_screen8": pair("cast_2", "base_screen8"), "base_screen vs base_screen8": pair("base_screen", "base_screen8")}}
(OUT / "safetune" / "cast_vs_adasteer.json").write_text(json.dumps(out, indent=1))
for k, v in out["pairs"].items():
    print(f"{k:30s} {v}")
