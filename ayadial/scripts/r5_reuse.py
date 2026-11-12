"""Can last night's development cells be reused tonight? Re-generate the base on the same rows with the same batching and compare.
Also runs v1 at s = 1 from the adapter files on the same rows and compares it with last night's foldcap@1 (the check that v1 is v1).
Threshold written in the journal before the run: at least 99% identical refusal labels on the base, else nothing is reused.
  python scripts/r5_reuse.py
Writes results/round5/reuse_check.json. If the check passes, last night's cells (wj and own rows only) are copied into runs_private/round5/gen."""
import shutil
from r5lib import *

model, tok = load()
out = {"threshold": "base refusal labels at least 99% identical", "suites": {}}
ok = True
for suite, bname in (("wjx", "base_wjx"), ("own", "base_own")):
    rows = suite_rows(suite)
    old = {rk(r): r for r in read_jsonl(RAW / "r4_reuse" / f"{bname}.jsonl")}
    assert all(rk(r) in old for r in rows), f"{suite}: last night's base does not cover the rows"
    new_lab = run(model, tok, suite, f"chk_{bname}", None, rows)
    new = {rk(r): r for r in read_jsonl(RAW / "gen" / f"{cname('chk_' + bname)}.jsonl")}
    judge([old[rk(r)] for r in rows])
    keys = [rk(r) for r in rows]
    old_lab = {k: label(old[k]) for k in keys}
    same_text = float(np.mean([old[k]["response"] == new[k]["response"] for k in keys]))
    agree = [float(np.mean([old_lab[k][c] == new_lab[k][c] for k in keys])) for c in range(3)]
    e = {"n": len(keys), "identical_responses": same_text, "label_agreement": {"refusal": agree[0], "unsafe": agree[1], "degenerate": agree[2]},
         "rates_last_night": [round(float(np.mean([old_lab[k][c] for k in keys]) * 100), 2) for c in range(3)], "rates_tonight": [round(float(np.mean([new_lab[k][c] for k in keys]) * 100), 2) for c in range(3)]}
    e["pass"] = bool(agree[0] >= 0.99)
    ok &= e["pass"]
    out["suites"][suite] = e
    log(f"{suite}: {e}")
# v1 from the files against last night's foldcap@1, on the wjx rows
rows = suite_rows("wjx")
old = {rk(r): r for r in read_jsonl(RAW / "r4_reuse" / "foldcap@1.jsonl")}
new_lab = run(model, tok, "wjx", "chk_foldcap@1", Units(model, v1_units(vec("v1"), 1.0)), rows)
new = {rk(r): r for r in read_jsonl(RAW / "gen" / f"{cname('chk_foldcap@1')}.jsonl")}
keys = [rk(r) for r in rows]
judge([old[k] for k in keys])
old_lab = {k: label(old[k]) for k in keys}
fo, fn = np.array([old[k].get("fire") or 0.0 for k in keys]), np.array([new[k].get("fire") or 0.0 for k in keys])
out["v1_from_files_vs_last_night_foldcap@1"] = {"n": len(keys), "identical_responses": float(np.mean([old[k]["response"] == new[k]["response"] for k in keys])),
                                               "label_agreement": {"refusal": float(np.mean([old_lab[k][0] == new_lab[k][0] for k in keys])), "unsafe": float(np.mean([old_lab[k][1] == new_lab[k][1] for k in keys]))},
                                               "fire_max_abs_diff": float(np.abs(fo - fn).max()), "fire_corr": float(np.corrcoef(fo, fn)[0, 1]),
                                               "rates_last_night": [round(float(np.mean([old_lab[k][c] for k in keys]) * 100), 2) for c in range(3)], "rates_tonight": [round(float(np.mean([new_lab[k][c] for k in keys]) * 100), 2) for c in range(3)]}
log(f"v1: {out['v1_from_files_vs_last_night_foldcap@1']}")
out["reuse"] = bool(ok)
if ok and not os.environ.get("R5_SMOKE"):
    copied = []
    for f in sorted((RAW / "r4_reuse").glob("*.jsonl")):
        dst = RAW / "gen" / f.name
        if not dst.exists():
            shutil.copy(f, dst)
            copied.append(f.stem)
    for name in copied:  # labels for the reused cells (no generation: every row is already there)
        rows_ = read_jsonl(RAW / "gen" / f"{name}.jsonl")
        judge(rows_)
        (OUT / "labels" / f"{name}.json").write_text(json.dumps({rk(r): label(r) for r in rows_}))
    out["copied_cells"] = copied
    # tonight's degeneracy rule must not change last night's labels on these rows
    diff = {}
    for name in copied:
        old_l, new_l = json.loads((R4OUT / "labels" / f"{name}.json").read_text()), json.loads((OUT / "labels" / f"{name}.json").read_text())
        diff[name] = sum(old_l[k] != v for k, v in new_l.items())
    out["labels_changed_vs_round4_files"] = {"total": int(sum(diff.values())), "by_cell": {k: v for k, v in diff.items() if v}}
else:  # tonight's own bases become the bases; every other cell on these rows is re-run tonight
    for bname in ("base_wjx", "base_own"):
        shutil.copy(RAW / "gen" / f"{cname('chk_' + bname)}.jsonl", RAW / "gen" / f"{cname(bname)}.jsonl")
        shutil.copy(OUT / "labels" / f"{cname('chk_' + bname)}.json", OUT / "labels" / f"{cname(bname)}.json")
    shutil.copy(RAW / "gen" / f"{cname('chk_foldcap@1')}.jsonl", RAW / "gen" / f"{cname('foldcap@1')}.jsonl")
    shutil.copy(OUT / "labels" / f"{cname('chk_foldcap@1')}.json", OUT / "labels" / f"{cname('foldcap@1')}.json")
(OUT / f"{cname('reuse_check')}.json").write_text(json.dumps(out, indent=1))
sync()
log(f"R5_REUSE DONE reuse={ok}")
