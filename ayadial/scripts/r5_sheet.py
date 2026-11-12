"""Build one blinded sheet of frozen replies for a person to label (the round's recommended next step). Pure CPU, on the runtime.
  python scripts/r5_sheet.py
Strata (prompt, cells): in-the-wild template + harmful question (20 prompts; base and v3g replies), in-the-wild template + harmless instruction
(15; base and v3g), Swahili and Javanese MultiJail (15 each; base and v3g), and fresh OR-Bench prompts the base refused and the lax setting answered
(20; the lax reply). Where a second judge has scored the rows, half of a stratum is drawn from prompts on which the two judges disagree.
Writes runs_private/round5/human_sheet_r5_blinded.csv (prompt and reply text: private), human_sheet_r5_KEY.csv (cell and judge labels: do not open
before labelling) and results/round5/human_sheet_r5.json (ids and counts only). Nobody has filled it in; the script does not."""
import csv, json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "runs_private/round5", ROOT / "results/round5"
rng = random.Random(55)


def gen(name):
    p = RAW / "gen" / f"{name}.jsonl"
    return {f"{r['set']}|{r['id']}": r for r in (json.loads(l) for l in p.read_text().split("\n") if l.strip())}


def lab(folder, name):
    p = OUT / folder / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


G = {c: gen(c) for c in ("base_frz5", "v3gcap@1", "lax3@2")}
M = {c: lab("labels", c) for c in G}
S2 = {c: {**lab("labels_wg", c), **lab("labels_q8", c)} for c in G}
STRATA = [("itw_h_frz", 20, ["base_frz5", "v3gcap@1"]), ("itw_b_frz", 15, ["base_frz5", "v3gcap@1"]), ("mj_sw_frz", 15, ["base_frz5", "v3gcap@1"]), ("mj_jv_frz", 15, ["base_frz5", "v3gcap@1"])]
rows, meta, from_dis = [], {}, set()
for set_, n, cells in STRATA:
    keys = sorted(k for k in G["base_frz5"] if k.split("|")[0] == set_)
    dis = [k for k in keys if any(k in S2[c] and S2[c][k][:2] != M[c][k][:2] for c in cells)]
    pick = rng.sample(dis, min(n // 2, len(dis)))
    from_dis.update(pick)  # the disagreement half: over-sampled on purpose, so rates on this sheet are not population estimates
    pick += rng.sample([k for k in keys if k not in pick], n - len(pick))
    meta[set_] = {"prompts": len(pick), "drawn_from_judge_disagreements": len([k for k in pick if k in dis]), "cells": cells}
    rows += [(k, c) for k in pick for c in cells]
orb = sorted(k for k in G["base_frz5"] if k.split("|")[0] == "orb5_frz" and M["base_frz5"][k][0] == 1 and M["lax3@2"][k][0] == 0)
pick = rng.sample(orb, min(20, len(orb)))
meta["orb5_frz"] = {"prompts": len(pick), "pool_base_refused_lax_answered": len(orb), "cells": ["lax3@2"]}
rows += [(k, "lax3@2") for k in pick]
rng.shuffle(rows)
with open(RAW / "human_sheet_r5_blinded.csv", "w", newline="") as fb, open(RAW / "human_sheet_r5_KEY.csv", "w", newline="") as fk:
    wb, wk = csv.writer(fb), csv.writer(fk)
    wb.writerow(["row", "prompt", "response", "human_refused (y/n)", "human_harmful_content (y/n)", "notes"])
    wk.writerow(["row", "set", "id", "cell", "main_refused", "main_unsafe", "second_judge_refused", "second_judge_unsafe", "drawn_from_disagreement_half"])
    for i, (k, c) in enumerate(rows, 1):
        r = G[c][k]
        wb.writerow([i, r["prompt"], r["response"], "", "", ""])
        s2 = S2[c].get(k)
        wk.writerow([i, *k.split("|", 1), c, M[c][k][0], M[c][k][1], "" if s2 is None else s2[0], "" if s2 is None else s2[1], int(k in from_dis)])
(OUT / "human_sheet_r5.json").write_text(json.dumps({"rows": len(rows), "strata": meta, "files": "runs_private/round5/human_sheet_r5_blinded.csv and human_sheet_r5_KEY.csv (private)",
                                                     "ids": [f"{k}|{c}" for k, c in rows], "filled_in": False,
                                                     "note": "half of each of the first four strata is drawn from prompts on which the two judges disagree (marked in the KEY file), so agreement on this sheet is not a population estimate; the other half is a random draw"}, indent=1))
print(len(rows), json.dumps(meta))
