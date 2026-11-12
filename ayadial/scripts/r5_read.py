"""Score a read on every benchmark: python scripts/r5_read.py <vec name> [<vec name> ...]      (vec_<name>.npz in cache_r5)
Forward passes only (cached). For each pair of a harmful and a harmless set: AUROC of z = w.x at the marker, overall and among the prompts
the base model answers (the base cell of the suite the row belongs to), and the share of each set above the firing thresholds theta0/s.
Writes results/round5/read/<vec>.json. No prompt text."""
import collections, sys
from r5lib import *

LG = ["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]
RES = {"high": ["zh", "it", "vi"], "medium": ["ar", "ko", "th"], "low": ["bn", "sw", "jv"]}
PLAIN_B = ["benign", "xstest_safe"]


def pairs(names, side):
    """{pair name: (harmful sets, harmless sets)} for one side ('dev' or 'frz'). Only pairs whose sets exist are kept."""
    P = collections.OrderedDict()
    if side == "dev":
        P["r4.adversarial (wj eval rows)"] = (["wj_adversarial_harmful"], ["wj_adversarial_benign"])
        P["r4.held-out clusters"] = (["dev_ho_h"], ["dev_ho_b"])
        P["r4.plain harmful vs benign"] = (["harmful"], ["benign"])
        P["r4.harmful vs orbench_hard"] = (["harmful"], ["orbench_hard"])
        P["r4.harmful_jb vs adversarial_benign"] = (["harmful_jb"], ["wj_adversarial_benign"])
        P["pht.harmful vs pht_harmless"] = (["harmful"], ["pht_harmless_dev"])
        P["pht.harmful vs pht_contro"] = (["harmful"], ["pht_contro_dev"])
    else:
        P["wj5.fresh adversarial"] = (["wj5_h_frz"], ["wj5_b_frz"])
    P[f"itw.same templates ({side})"] = ([f"itw_h_{side}"], [f"itw_b_{side}"])
    P[f"itw.harmful vs plain harmless ({side})"] = ([f"itw_h_{side}"], PLAIN_B if side == "dev" else ["orb5_frz"])
    for lg in LG:
        P[f"mj.{lg} ({side})"] = ([f"mj_{lg}_{side}"], [f"aya_{lg}_{side}"])
    for lv, ls in RES.items():
        P[f"mj.{lv}-resource ({side})"] = ([f"mj_{l}_{side}" for l in ls], [f"aya_{l}_{side}" for l in ls])
    P[f"mj.all non-English ({side})"] = ([f"mj_{l}_{side}" for l in LG[1:]], [f"aya_{l}_{side}" for l in LG[1:]])
    for m in ("GCG", "PAIR", "DSN", "JBC", "RS"):
        for pool in (["pool", "dev"] if side == "dev" else ["frz"]):
            P[f"jbb.{m} {pool} vs plain harmless"] = ([f"jbb_{m}_{pool}"], PLAIN_B if side == "dev" else [f"aya_en_{side}", "orb5_frz"])
    P[f"sfx.transplants ({side})"] = ([f"sfx_h_{side}"], [f"sfx_b_{side}"])
    P[f"jbb.GCG vs harmless+suffix ({side})"] = ([f"jbb_GCG_{side}"], [f"sfx_b_{side}"])
    P[f"jbb.DSN vs harmless+suffix ({side})"] = ([f"jbb_DSN_{side}"], [f"sfx_b_{side}"])
    styles = sorted({n[len("sorry_"):-len(f"_{side}")] for n in names if n.startswith("sorry_") and n.endswith(f"_{side}")})
    for st in styles:
        P[f"sorry.{st} ({side})"] = ([f"sorry_{st}_{side}"], [f"pht_harmless_{side}"])
    return collections.OrderedDict((k, v) for k, v in P.items() if all(n in names for n in v[0] + v[1]))


def main():
    side = "frz" if "--frozen" in sys.argv else "dev"
    vnames = [a for a in sys.argv[1:] if not a.startswith("--")]
    allr = sets()
    by = collections.defaultdict(list)
    for r in allr:
        by[r["set"]].append(r)
    P = pairs(set(by), side)
    need = sorted({n for h, b in P.values() for n in h + b})
    model = tok = None
    A = {}
    for n in need:
        if not (CACHE / f"act_{n}.npz").exists() and model is None:
            model, tok = load()
        A[n] = acts(model, tok, n)
    # base answered: from whichever base cell holds the row
    base = {}
    for f in sorted((OUT / "labels").glob("base_*.json")):
        if side == "dev" and f.stem == "base_frz5":
            continue
        base.update(json.loads(f.read_text()))
    (OUT / "read").mkdir(exist_ok=True)
    for vn in vnames:
        V = vec(vn)
        th = V["theta0"]
        Z = {n: zread(V, A[n][0]) for n in need}
        ans = {n: np.array([base.get(f"{n}|{i}", [1, 0, 0])[0] == 0 and base.get(f"{n}|{i}", [1, 0, 0])[2] == 0 for i in A[n][2]]) for n in need}
        have = {n: np.mean([f"{n}|{i}" in base for i in A[n][2]]) for n in need}
        out = {"vec": vn, "theta0": th, "side": side, "pairs": {}, "sets": {}}
        for n in need:
            z = Z[n]
            out["sets"][n] = {"n": int(len(z)), "base_label_coverage": float(have[n]), "base_answered": int(ans[n].sum()), "z_mean": float(z.mean()), "z_p10_p50_p90": [float(x) for x in np.percentile(z, [10, 50, 90])],
                              **{f"fires_s{s:g}": float((s * z > th).mean()) for s in (1, 1.25, 2)}, **{f"fires_s{s:g}_among_answered": (float((s * z[ans[n]] > th).mean()) if ans[n].any() else None) for s in (1, 1.25)},
                              "push_s1_mean": float(np.clip(BETA * (z - th), 0, 1).mean())}
        for name, (hs, bs) in P.items():
            zh, zb = np.concatenate([Z[n] for n in hs]), np.concatenate([Z[n] for n in bs])
            ah, ab = np.concatenate([ans[n] for n in hs]), np.concatenate([ans[n] for n in bs])
            e = {"n_h": int(len(zh)), "n_b": int(len(zb)), "auroc": auroc(zh, zb), "n_h_answered": int(ah.sum()), "n_b_answered": int(ab.sum()),
                 "auroc_answered": auroc(zh[ah], zb[ab]) if ah.sum() >= 5 and ab.sum() >= 5 else None,
                 "fires_h_s1": float((zh > th).mean()), "fires_b_s1": float((zb > th).mean()), "fires_h_s1_answered": float((zh[ah] > th).mean()) if ah.any() else None}
            rng = np.random.default_rng(0)
            bt = [auroc(zh[rng.integers(0, len(zh), len(zh))], zb[rng.integers(0, len(zb), len(zb))]) for _ in range(500)]
            e["auroc_ci95"] = [float(x) for x in np.percentile(bt, [2.5, 97.5])]
            if e["auroc_answered"] is not None:
                zha, zba = zh[ah], zb[ab]
                bt = [auroc(zha[rng.integers(0, len(zha), len(zha))], zba[rng.integers(0, len(zba), len(zba))]) for _ in range(500)]
                e["auroc_answered_ci95"] = [float(x) for x in np.percentile(bt, [2.5, 97.5])]
            out["pairs"][name] = e
        (OUT / "read" / f"{vn}{'_frozen' if side == 'frz' else ''}.json").write_text(json.dumps(out, indent=1))
        for name, e in out["pairs"].items():
            aa = "none" if e["auroc_answered"] is None else f"{e['auroc_answered']:.3f}"
            print(f"{vn} | {name:44s} n {e['n_h']:4d}/{e['n_b']:4d} auroc {e['auroc']:.3f} answered {aa} ({e['n_h_answered']}/{e['n_b_answered']}) fires h {e['fires_h_s1']:.2f} b {e['fires_b_s1']:.2f}")
    sync()
    log("R5_READ DONE")


if __name__ == "__main__":
    main()
