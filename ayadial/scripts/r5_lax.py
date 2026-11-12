"""The lax side, before any generation: where could it fire? Forward passes only (cached).
  python scripts/r5_lax.py <vec name> [--frozen]
The lax units push away from refusal where the read z is BELOW a threshold theta. For each candidate theta this reports, per set:
  the share of prompts below theta (the unit fires), and that share among prompts the base model refuses (only those can change from refusal to answer).
On harmless sets that second number is the false refusals the lax side can reach; on harmful sets it is the refusals it could release.
Thresholds: the 1st and 5th percentile of z on the harmful fit prompts (trfit_h), and two stricter fixed values.
Writes results/round5/lax/exposure_<vec>.json. No prompt text."""
import collections, sys
from r5lib import *


def main():
    vn = sys.argv[1]
    side = "frz" if "--frozen" in sys.argv else "dev"
    V = vec(vn)
    by = collections.defaultdict(list)
    for r in sets():
        by[r["set"]].append(r)
    names = [n for n in by if (n in WJ_H + WJ_B + OWN or n.endswith("_dev") or n.endswith("_pool"))] if side == "dev" else [n for n in by if n.endswith("_frz")]
    model = tok = None
    A = {}
    for n in ["trfit_h", "trfit_b", "harmful_fit", "benign_fit", "xstest_safe_fit", "orbench_hard_fit"] + names:
        if not (CACHE / f"act_{n}.npz").exists() and model is None:
            model, tok = load()
        A[n] = acts(model, tok, n)
    Z = {n: zread(V, A[n][0]) for n in A}
    base = {}
    for f in sorted((OUT / "labels").glob("base_*.json")):
        if side == "dev" and f.stem == "base_frz5":
            continue
        base.update(json.loads(f.read_text()))
    th = {"h01": float(np.quantile(Z["trfit_h"], 0.01)), "h05": float(np.quantile(Z["trfit_h"], 0.05)), "fixed0.25": 0.25, "fixed0.10": 0.10}
    out = {"vec": vn, "side": side, "thresholds": th, "theta0_strict": V["theta0"], "ramp_width": 1.0 / LAX_BETA,
           "fit": {n: {"n": int(len(Z[n])), **{f"below_{k}": float((Z[n] < v).mean()) for k, v in th.items()}} for n in ("trfit_h", "trfit_b", "harmful_fit", "benign_fit", "xstest_safe_fit", "orbench_hard_fit")}, "sets": {}}
    for n in names:
        z, ids = Z[n], A[n][2]
        have = np.array([f"{n}|{i}" in base for i in ids])
        ref = np.array([base.get(f"{n}|{i}", [0, 0, 0])[0] == 1 for i in ids])
        e = {"n": int(len(z)), "harm": by[n][0].get("harm"), "base_labels": int(have.sum()), "base_refused": int(ref.sum())}
        for k, v in th.items():
            e[f"below_{k}"] = float((z < v).mean())
            e[f"full_push_{k}"] = float((z < v - 1.0 / LAX_BETA).mean())
            e[f"refused_and_below_{k}"] = float((ref & (z < v)).mean()) if have.any() else None
            e[f"share_of_refused_below_{k}"] = float((z[ref] < v).mean()) if ref.any() else None
        out["sets"][n] = e
    (OUT / "lax").mkdir(exist_ok=True)
    (OUT / "lax" / f"exposure_{vn}{'_frozen' if side == 'frz' else ''}.json").write_text(json.dumps(out, indent=1))
    print("thresholds", {k: round(v, 3) for k, v in th.items()}, "| fit below h01:", {n: round(v["below_h01"], 3) for n, v in out["fit"].items()})
    for n, e in sorted(out["sets"].items()):
        f = lambda x: "  -  " if x is None else f"{x:.2f}"
        print(f"{n:26s} n {e['n']:4d} refused {e['base_refused']:4d} | below h01 {f(e['below_h01'])} of refused {f(e['share_of_refused_below_h01'])} | below h05 {f(e['below_h05'])} of refused {f(e['share_of_refused_below_h05'])} | below 0.25 {f(e['below_fixed0.25'])} of refused {f(e['share_of_refused_below_fixed0.25'])}")
    sync()
    log("R5_LAX DONE")


if __name__ == "__main__":
    main()
