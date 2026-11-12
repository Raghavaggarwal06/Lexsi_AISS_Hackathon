"""Published attack artifacts (JailbreakBench PAIR black-box and GCG white-box, vicuna-13b-v1.5 files, as in round 3 block 5),
matched to our eval-half JBB behaviours. Transfer attacks, not adaptive to the fold. Mine (not in the charter's list).
  python r4_jbb.py"""
import urllib.request
from r4lib import *
from r4_fold import Fold
SHA = json.loads((ROOT / "results/round3/block5/sets.json").read_text())["jbb_artifacts_sha"]


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "eval-client"}), timeout=60).read()


goals = {r["prompt"].strip().lower(): r["id"] for r in load_set("harmful") if r["id"] in set(load_splits()["harmful"]["eval"]) and r["id"].startswith("jbb-")}
rows = []
for attack, path in (("PAIR", "attack-artifacts/PAIR/black_box/vicuna-13b-v1.5.json"), ("GCG", "attack-artifacts/GCG/white_box/vicuna-13b-v1.5.json")):
    data = json.loads(fetch(f"https://raw.githubusercontent.com/JailbreakBench/artifacts/{SHA}/{path}"))
    for j in data.get("jailbreaks", []):
        gid = goals.get(str(j.get("goal", "")).strip().lower())
        if gid and j.get("prompt"):
            rows.append({"id": f"{gid}::{attack}", "set": f"jbb_{attack}", "prompt": j["prompt"]})
model, tok = load()
out = {"artifacts_sha": SHA, "n": {s: sum(r["set"] == s for r in rows) for s in ("jbb_PAIR", "jbb_GCG")}, "rates": {}}
from r4lib import Steer
old = np.load(ROOT / "results/day1/directions.npz")
cells = [("base_jbb", None), ("ungated@0.1", Steer(model, old["r_raw"][23], 0.1, f="ungated", pos="all")), ("foldcap@1", Fold(model, 1.0, gain=1.0, cap_beta=4.0)), ("foldcap@1.25", Fold(model, 1.25, gain=1.0, cap_beta=4.0))]
for name, ctx in cells:
    lab = run_cell(model, tok, name, rows, ctx)
    out["rates"][name] = {s: {"n": int(sum(r["set"] == s for r in rows)), "refusal": round(float(np.mean([lab[rk(r)][0] for r in rows if r["set"] == s]) * 100), 1), "unsafe": round(float(np.mean([lab[rk(r)][1] for r in rows if r["set"] == s]) * 100), 1),
                              "degenerate": round(float(np.mean([lab[rk(r)][2] for r in rows if r["set"] == s]) * 100), 1)} for s in ("jbb_PAIR", "jbb_GCG")}
    log(f"{name} {out['rates'][name]}")
(OUT / "controls").mkdir(exist_ok=True)
(OUT / "controls/jbb_artifacts.json").write_text(json.dumps(out, indent=1))
sync()
log("R4_JBB DONE")
