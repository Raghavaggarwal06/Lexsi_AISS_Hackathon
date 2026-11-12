"""Run cells on a suite: python scripts/r5_run.py specs5/<spec>.json
spec = {"suite": "dev5", "cells": [{"name": "foldcap@1", "f": "v1", "s": 1}, {"name": "ungated@0.1", "f": "ungated", "scale": 0.1}, ...]}
A custom suite is declared once with "suite_sets": [regex, ...] (rows of sets.jsonl whose set name matches); its rows are then fixed in
runs_private/round5/suites/<suite>.jsonl. Cells of a custom suite must carry the suite as a prefix ("bs.lax@1"), so that no cell file mixes
two batchings of the same row. The base cell of the suite (base_<suite>) is generated first if missing.
One line per cell goes to the log; rates per set and per group go to results/round5/run/<spec>.json."""
import collections, sys
from r5lib import *

POOLS = (("ADV_H", lambda s: s in WJ_H), ("ADV_B", lambda s: s in WJ_B),
         ("MJ_dev", lambda s: s.startswith("mj_") and s.endswith("_dev")), ("MJ_frz", lambda s: s.startswith("mj_") and s.endswith("_frz")),
         ("AYA_dev", lambda s: s.startswith("aya_") and s.endswith("_dev")), ("AYA_frz", lambda s: s.startswith("aya_") and s.endswith("_frz")),
         ("PHT_dev", lambda s: s.startswith("pht_") and s.endswith("_dev")), ("PHT_frz", lambda s: s.startswith("pht_") and s.endswith("_frz")),
         ("JBB_seen_methods", lambda s: re.fullmatch(r"jbb_(GCG|PAIR)_(dev|frz)", s) is not None), ("JBB_unseen_methods", lambda s: re.fullmatch(r"jbb_(DSN|JBC|RS)_(dev|frz)", s) is not None),
         ("JBB_pool", lambda s: s.startswith("jbb_") and s.endswith("_pool")),
         ("SORRY_dev", lambda s: s.startswith("sorry_") and s.endswith("_dev")), ("SORRY_frz", lambda s: s.startswith("sorry_") and s.endswith("_frz")))


def groups(rows):
    """Report groups: every set, plus pooled groups (upper-case first letters) by benchmark and side."""
    g = collections.OrderedDict()
    for r in rows:
        for tag, cond in POOLS:
            if cond(r["set"]):
                g.setdefault(tag, []).append(rk(r))
    for r in rows:
        g.setdefault(r["set"], []).append(rk(r))
    return g


def rates(lab, base, keys):
    a, b0 = np.array([lab[k] for k in keys], float), np.array([base[k] for k in keys], float)
    return {"n": len(keys), "ref": round(a[:, 0].mean() * 100, 1), "uns": round(a[:, 1].mean() * 100, 1), "deg": round(a[:, 2].mean() * 100, 1), "base_ref": round(b0[:, 0].mean() * 100, 1), "base_uns": round(b0[:, 1].mean() * 100, 1)}


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    suite = spec["suite"]
    if spec.get("suite_sets"):
        (RAW / "suites").mkdir(exist_ok=True)
        sp = RAW / "suites" / f"{suite}.jsonl"
        if not sp.exists():
            write_jsonl(sp, [r for r in sets() if any(re.fullmatch(p, r["set"]) for p in spec["suite_sets"])])
        assert all(c["name"].startswith(suite + ".") for c in spec["cells"]), "cells of a custom suite need the suite prefix"
    rows = suite_rows(suite)
    model, tok = load()
    bname = base_name(suite)
    t0 = time.time()
    base = run(model, tok, suite, bname, None, rows)
    log(f"base {bname} n={len(base)} ({time.time() - t0:.0f}s) refused {np.mean([v[0] for v in base.values()]) * 100:.1f} unsafe {np.mean([v[1] for v in base.values()]) * 100:.1f}")
    sync()
    (OUT / "run").mkdir(exist_ok=True)
    logp = OUT / "run" / f"{Path(sys.argv[1]).stem}.json"
    summ = json.loads(logp.read_text()) if logp.exists() else {}
    G = groups(rows)
    show = spec.get("show") or [g for g in G if g[0].isupper()] or list(G)[:8]
    summ["base"] = {"suite": suite, "n": len(rows), **{g: rates(base, base, keys) for g, keys in G.items()}}
    for c in spec["cells"]:
        t0 = time.time()
        lab = run(model, tok, suite, c["name"], context(model, c), rows)
        gen = {rk(r): r for r in read_jsonl(RAW / "gen" / f"{cname(c['name'])}.jsonl")}
        s = {"spec": {k: v for k, v in resolve(c).items() if k != "name"}, "suite": suite, "n": len(rows), "seconds": round(time.time() - t0)}
        for g, keys in G.items():
            s[g] = rates(lab, base, keys)
            fire = [gen[k].get("fire") for k in keys if k in gen and gen[k].get("fire") is not None]
            if fire:
                s[g]["fire_mean"] = round(float(np.mean(fire)), 3)
                s[g]["fire_gt_half"] = round(float(np.mean(np.abs(fire) > 0.5)), 3)
        summ[c["name"]] = s
        logp.write_text(json.dumps(summ, indent=1))
        f = lambda t: f"{t} {s[t]['base_ref']}->{s[t]['ref']}/{s[t]['base_uns']}->{s[t]['uns']}" + (f" f{s[t]['fire_gt_half']}" if "fire_gt_half" in s[t] else "")
        log(f"{c['name']} ({s['seconds']}s) | " + " | ".join(f(t) for t in show if t in s))
        sync()
    log("R5_RUN DONE")


if __name__ == "__main__":
    main()
