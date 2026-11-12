"""Round 7: paired differences AyaDial minus each SafeTune method on every test row of the report (CPU, labels only).
  python scripts/r7_vs_ayadial.py
AyaDial = cell v3gcap@1 of rounds 5 and 6; SafeTune cells = round 7's picked settings (safetune_pick.json). Same rows, cluster bootstrap of the
report's tables (r7_safetune.paired: 2000 resamples, seed 0, clusters of r5_report / r6_report). Negative unsafe = AyaDial lets fewer unsafe replies
through; negative refusal = AyaDial refuses less. Writes results/round7/ayadial_vs_safetune.json. Also includes results/round7 AyaDial-WSP cells
(<suite>.st_wsp@1) when they exist."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7lib as L7
import r7_safetune as S7
import r5_report as R5
c5 = R5.cluster_of
import r6_report as R6r
c6 = R6r.cluster_of

P = S7.load_pick()
arms = dict(P["methods"])
if (L7.OUT / "labels" / "frz5.st_wsp@1.json").exists():
    L7.METHODS["wsp"] = "AyaDial-WSP"
    arms["wsp"] = {"setting": 1}
out = {"reference": "AyaDial (v3gcap@1)", "difference": "AyaDial minus the method, percentage points, 95% cluster-bootstrap interval", "first": {}, "second": {}}
for test in ("first", "second"):
    Lr = L7.round_labels(L7.TABLES[test][0])
    aya = Lr["v3gcap@1"]
    R5.cluster_of = c5 if test == "first" else c6
    for arm, m in arms.items():
        mc, _ = S7.method_cells(arm, m)
        lab = L7.r7_labels(mc[test])
        for row in L7.table_rows(test).values():
            keys = sorted(k for k in Lr["base"] if k.split("|")[0] in row["sets"])
            if not keys or not all(k in lab and k in aya for k in keys):
                continue
            d = S7.paired(aya, lab, keys, [R5.cluster_of(k) for k in keys], R5)
            out[test].setdefault(L7.METHODS[arm], {})[row["row"]] = {"n": len(keys), "kind": row["kind"], **d}
L7.write_json(L7.OUT / "ayadial_vs_safetune.json", out)
show = ["adversarial_harmful (fresh WildJailbreak)", "in-the-wild template + harmful question"]
for test in ("first", "second"):
    for name, rows in out[test].items():
        print(test, name, len(rows), "rows")
