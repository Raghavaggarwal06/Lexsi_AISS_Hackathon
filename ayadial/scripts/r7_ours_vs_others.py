"""Round 7: paired differences of each of our two adapters against every other method, on every test row of the report (CPU, labels only).
  python scripts/r7_ours_vs_others.py   ->  results/round7/ours_vs_others.json
Ours: AyaDial (v3gcap@1, rounds 5 and 6) and AyaDial-WSP (round 7, st_wsp@1). Others: Control (ungated@0.1, rounds 5 and 6) and SafeTune
RefusalDirection, AdaSteer, CAST at their picked settings (round 7). Difference = ours minus other, percentage points; same rows; the tables'
cluster bootstrap (r7_safetune.paired: 2000 resamples, seed 0). Negative unsafe = ours lets fewer unsafe replies through."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7lib as L7
import r7_safetune as S7
import r5_report as R5
c5 = R5.cluster_of
import r6_report as R6r
c6 = R6r.cluster_of

P = S7.load_pick()
L7.METHODS["wsp"] = "AyaDial-WSP"
out = {"difference": "ours minus the other method, percentage points, 95% cluster-bootstrap interval", "first": {}, "second": {}}
for test in ("first", "second"):
    Lr = L7.round_labels(L7.TABLES[test][0])
    R5.cluster_of = c5 if test == "first" else c6
    others = {"Control": Lr[L7.CONTROL]}
    for arm, m in P["methods"].items():
        others[L7.METHODS[arm]] = L7.r7_labels(S7.method_cells(arm, m)[0][test])
    ours = {"AyaDial": Lr["v3gcap@1"], "AyaDial-WSP": L7.r7_labels(S7.method_cells("wsp", {"setting": 1})[0][test])}
    for row in L7.table_rows(test).values():
        keys = sorted(k for k in Lr["base"] if k.split("|")[0] in row["sets"])
        cl = [R5.cluster_of(k) for k in keys]
        for on, ol in ours.items():
            for tn, tl in others.items():
                if keys and all(k in ol and k in tl for k in keys):
                    out[test].setdefault(on, {}).setdefault(row["row"], {"n": len(keys), "kind": row["kind"]})[tn] = S7.paired(ol, tl, keys, cl, R5)
L7.write_json(L7.OUT / "ours_vs_others.json", out)
for test in ("first", "second"):
    for on, rows in out[test].items():
        print(test, on, len(rows), "rows", sorted({k for r in rows.values() for k in r if k not in ("n", "kind")}))
