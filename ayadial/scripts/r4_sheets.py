"""Private sheets for hand labelling. Writes to runs_private/round4/ only.
  python r4_sheets.py <base cell> <method cell> <control cell>"""
import csv, sys
from r4lib import *
base_c, meth_c, ctrl_c = sys.argv[1:4]
rng = random.Random(77)
H = ["wj_adversarial_harmful", "dev_ho_h", "dev_in_h"]
B = ["wj_adversarial_benign", "dev_ho_b", "dev_in_b"]
gen = {c: {rk(r): r for r in read_jsonl(RAW / "gen" / f"{c}.jsonl")} for c in (base_c, meth_c, ctrl_c)}
lab = {c: labels(c) for c in gen}
wg = {c: json.loads((OUT / "labels_wg" / f"{c}.json").read_text()) for c in gen if (OUT / "labels_wg" / f"{c}.json").exists()}
items, seen = [], set()


def add(cell, key, why):
    if (cell, key) in seen or key not in gen[cell]:
        return
    seen.add((cell, key))
    items.append({"cell": cell, "key": key, "why": why})


def take(pool, n, why):
    rng.shuffle(pool)
    for cell, key in pool[:n]:
        add(cell, key, why)


hk = [k for k in lab[base_c] if k.split("|")[0] in H]
bk = [k for k in lab[base_c] if k.split("|")[0] in B]
# 1. judge disagreements (Qwen3Guard vs WildGuard) on the unsafe label, harmful prompts
take([(c, k) for c in wg for k in hk if k in wg[c] and k in lab[c] and wg[c][k][1] != lab[c][k][1]], 30, "judges disagree on unsafe")
take([(c, k) for c in wg for k in hk + bk if k in wg[c] and k in lab[c] and wg[c][k][0] != lab[c][k][0]], 10, "judges disagree on refusal")
# 2. the cells the headline rests on
take([(meth_c, k) for k in hk if lab[base_c][k][1] == 1 and lab[meth_c].get(k, [0, 0, 0])[1] == 0], 15, "base unsafe, method not unsafe")
take([(base_c, k) for k in hk if lab[base_c][k][1] == 1], 10, "base judged unsafe")
take([(meth_c, k) for k in hk if lab[meth_c].get(k, [0, 0, 0])[1] == 1], 10, "method still judged unsafe")
take([(ctrl_c, k) for k in hk if lab[ctrl_c].get(k, [0, 0, 0])[1] == 1], 5, "control still judged unsafe")
take([(meth_c, k) for k in bk if lab[base_c][k][0] == 0 and lab[meth_c].get(k, [0, 0, 0])[0] == 1], 10, "benign newly refused by the method")
take([(base_c, k) for k in hk if lab[base_c][k][0] == 0 and lab[base_c][k][1] == 0], 10, "base answered, judged safe (harmful prompt)")
take([(c, k) for c in gen for k in hk + bk], max(0, 100 - len(items)), "random fill")
rng.shuffle(items)
with open(RAW / "human_sheet_blinded.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item", "prompt", "response", "human_refused (y/n)", "human_harmful_content (y/n)", "notes"])
    for i, it in enumerate(items):
        r = gen[it["cell"]][it["key"]]
        w.writerow([f"item-{i:03d}", r["prompt"], r["response"], "", "", ""])
with open(RAW / "human_sheet_KEY.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item", "cell", "set|id", "why_sampled", "qwen3guard_refused", "qwen3guard_unsafe", "wildguard_refused", "wildguard_harmful"])
    for i, it in enumerate(items):
        q = lab[it["cell"]][it["key"]]
        g = wg.get(it["cell"], {}).get(it["key"], ["", "", ""])
        w.writerow([f"item-{i:03d}", it["cell"], it["key"], it["why"], q[0], q[1], g[0], g[1]])
# 30 adversarial_benign prompts the base refuses: are they truly benign?
ref_b = [k for k in bk if lab[base_c][k][0] == 1]
rng.shuffle(ref_b)
with open(RAW / "benign_refused_check.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["set|id", "prompt", "base_response", "truly_benign (y/n)", "notes"])
    for k in ref_b[:30]:
        w.writerow([k, gen[base_c][k]["prompt"], gen[base_c][k]["response"], "", ""])
from collections import Counter
(OUT / "human_sheets.json").write_text(json.dumps({"n_items": len(items), "why": dict(Counter(it["why"] for it in items)), "cells": dict(Counter(it["cell"] for it in items)), "benign_refused_listed": min(30, len(ref_b)), "benign_refused_total": len(ref_b),
                                                   "files": ["runs_private/round4/human_sheet_blinded.csv", "runs_private/round4/human_sheet_KEY.csv", "runs_private/round4/benign_refused_check.csv"]}, indent=1))
sync()
print(json.loads((OUT / "human_sheets.json").read_text()))
