"""Phase 1: harm direction per layer, AUROC on the held-out half, GO / WEAK / NO-GO.

Layer indices here are hidden-state indices: 0 is the embedding output and k is the output of
decoder layer k-1, so a hook for index k goes on model.model.layers[k-1].

The best layer is picked using fit-half data only, because the data README keeps the eval half
for reading results. The eval-selected layer is reported next to it as an optimistic comparison.

Selection also requires separating xstest_unsafe from xstest_safe. Those two share one format,
and d never sees XSTest, so this is what stops the search landing on a layer that reads prompt
style: 03b_confound_check.py showed TF-IDF alone reaches 0.999 on harmful vs xstest_safe.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from common import DAY1, SETS, load_config, load_set, load_splits

COMPARISONS = {"benign": "harmful_vs_benign", "xstest_safe": "harmful_vs_xstest_safe", "orbench_hard": "harmful_vs_orbench_hard"}
NEGATIVES = list(COMPARISONS)
VARIANTS = ["raw", "cleaned"]
SAME_FORMAT = "xstest_unsafe_vs_xstest_safe"


def load_acts():
    """{set: {"fit": array, "eval": array}} with float32 activations."""
    splits, acts = load_splits(), {}
    for name in SETS:
        ids = [r["id"] for r in load_set(name)]
        where = {id_: i for i, id_ in enumerate(ids)}
        array = np.load(DAY1 / f"acts_{name}.npy").astype(np.float32)
        acts[name] = {h: array[[where[i] for i in splits[name][h]]] for h in ("fit", "eval")}
    return acts


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def clean(d, benign_fit, k):
    """Remove the top k principal components of the benign fit activations from d."""
    centred = benign_fit - benign_fit.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    top = vt[:k]
    return unit(d - top.T @ (top @ d))


def auroc(pos, neg):
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(labels, np.r_[pos, neg]))


def bootstrap_ci(pos, neg, seed, n=1000):
    rng = np.random.default_rng(seed)
    values = [
        auroc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)
    ]
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def constant_direction_cv(acts, layer):
    """Most near-constant direction after LayerNorm, as a proxy for the phase-3 tau input.

    Fit w = Sigma^-1 mu (ridge) on the benign-type fit halves and read std/|mean| on their eval
    halves. Uses last-token states only, so it is a proxy for "constant across tokens".
    """

    def normed(arrays):
        x = np.concatenate([a[:, layer] for a in arrays])
        return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)

    fit = normed([acts[n]["fit"] for n in NEGATIVES])
    held = normed([acts[n]["eval"] for n in NEGATIVES])
    mu = fit.mean(0)
    centred = fit - mu
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    ridge = (s**2).mean() + 1e-8  # keeps the inverse stable with far fewer rows than dimensions
    w = mu / ridge + vt.T @ ((vt @ mu) * (1 / (s**2 + ridge) - 1 / ridge))
    proj = held @ unit(w)
    return float(proj.std() / (abs(proj.mean()) + 1e-12))


def analyse_layer(acts, layer, k, tpr):
    harm_fit = acts["harmful"]["fit"][:, layer]
    benign_fit = acts["benign"]["fit"][:, layer]
    r_raw = harm_fit.mean(0) - benign_fit.mean(0)
    dirs = {"raw": unit(r_raw), "cleaned": clean(unit(r_raw), benign_fit, k)}

    out = {"r_raw": r_raw, "d": dirs}
    for variant, d in dirs.items():
        tau = float(np.quantile(harm_fit @ d, 1 - tpr))
        harm_eval = acts["harmful"]["eval"][:, layer] @ d
        row = {
            "tau": tau,
            "harmful_mean_proj": float((harm_fit @ d).mean()),
            "benign_mean_proj": float((benign_fit @ d).mean()),
            "tpr_eval_harmful": float((harm_eval > tau).mean()),
            "tpr_eval_xstest_unsafe": float((acts["xstest_unsafe"]["eval"][:, layer] @ d > tau).mean()),
            "auroc": {}, "fit_auroc": {}, "fpr": {},
        }
        for neg in NEGATIVES:
            key = COMPARISONS[neg]
            neg_eval = acts[neg]["eval"][:, layer] @ d
            row["auroc"][key] = auroc(harm_eval, neg_eval)
            row["fit_auroc"][key] = auroc(harm_fit @ d, acts[neg]["fit"][:, layer] @ d)
            row["fpr"][neg] = float((neg_eval > tau).mean())
        for half, store in (("eval", "auroc"), ("fit", "fit_auroc")):
            row[store][SAME_FORMAT] = auroc(
                acts["xstest_unsafe"][half][:, layer] @ d, acts["xstest_safe"][half][:, layer] @ d
            )
        out[variant] = row
    return out


def verdict(xstest, orbench, rules):
    if xstest >= rules["auroc_xstest_go"] and orbench >= rules["auroc_orbench_go"]:
        return "GO"
    if xstest >= rules["auroc_xstest_weak"]:
        return "WEAK"
    return "NO-GO"


def margin(aurocs, rules, same_format_bar=None):
    """How far a layer clears the tightest bar. Without a same-format bar, only the two GO bars."""
    gaps = [
        aurocs["harmful_vs_xstest_safe"] - rules["auroc_xstest_go"],
        aurocs["harmful_vs_orbench_hard"] - rules["auroc_orbench_go"],
    ]
    if same_format_bar is not None:
        gaps.append(aurocs[SAME_FORMAT] - same_format_bar)
    return min(gaps)


def plot(per_layer, path, best_layer):
    colors = {"harmful_vs_benign": "#4C72B0", "harmful_vs_xstest_safe": "#DD8452", "harmful_vs_orbench_hard": "#55A868", SAME_FORMAT: "#8172B3"}
    layers = sorted(per_layer)
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, color in colors.items():
        ax.plot(layers, [per_layer[l]["raw"]["auroc"][key] for l in layers], color=color, label=f"{key} (raw)")
        ax.plot(layers, [per_layer[l]["cleaned"]["auroc"][key] for l in layers], color=color, ls="--", label=f"{key} (cleaned)")
    ax.axvline(best_layer, color="gray", ls=":", label=f"selected layer {best_layer}")
    ax.axhline(0.9, color="black", lw=0.6)
    ax.set_xlabel("hidden-state index (0 = embeddings, k = output of decoder layer k-1)")
    ax.set_ylabel("AUROC on the eval half")
    ax.set_ylim(0.4, 1.01)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    cfg = load_config()
    acts = load_acts()
    rules = cfg["go_rules"]
    k, tpr = cfg["direction"]["clean_pcs"], cfg["direction"]["tpr_for_threshold"]
    lo, hi = cfg["layers"]["min"], cfg["layers"]["max"]

    analysis = {l: analyse_layer(acts, l, k, tpr) for l in range(37)}
    candidates = [(l, v) for l in range(lo, hi + 1) for v in VARIANTS]
    bar = cfg["selection"]["same_format_bar"]
    layer, variant = max(candidates, key=lambda c: margin(analysis[c[0]][c[1]]["fit_auroc"], rules, bar))
    oracle_layer, oracle_variant = max(candidates, key=lambda c: margin(analysis[c[0]][c[1]]["auroc"], rules, bar))
    bars_layer, bars_variant = max(candidates, key=lambda c: margin(analysis[c[0]][c[1]]["fit_auroc"], rules))
    bars_only = analysis[bars_layer][bars_variant]

    best = analysis[layer][variant]
    oracle = analysis[oracle_layer][oracle_variant]
    xs, orb = best["auroc"]["harmful_vs_xstest_safe"], best["auroc"]["harmful_vs_orbench_hard"]
    call = verdict(xs, orb, rules)
    oracle_call = verdict(oracle["auroc"]["harmful_vs_xstest_safe"], oracle["auroc"]["harmful_vs_orbench_hard"], rules)

    d = analysis[layer]["d"][variant]
    harm_eval = acts["harmful"]["eval"][:, layer] @ d
    ci = {COMPARISONS[n]: bootstrap_ci(harm_eval, acts[n]["eval"][:, layer] @ d, cfg["seed"]) for n in NEGATIVES}
    ci[SAME_FORMAT] = bootstrap_ci(acts["xstest_unsafe"]["eval"][:, layer] @ d, acts["xstest_safe"]["eval"][:, layer] @ d, cfg["seed"])
    cv ={l: constant_direction_cv(acts, l) for l in range(1, 37)}
    rule_gap = xs >= rules["auroc_xstest_go"] and call == "WEAK"

    results = {
        "selected": {"layer": layer, "decoder_layer_for_hook": layer - 1, "variant": variant, "selected_on": "fit half"},
        "verdict": call,
        "at_selected": {k_: v for k_, v in best.items()},
        "auroc_95ci_eval": ci,
        "eval_selected_comparison": {"layer": oracle_layer, "variant": oracle_variant, "verdict": oracle_call, "auroc": oracle["auroc"]},
        "go_bars_only_selection": {"layer": bars_layer, "variant": bars_variant, "auroc": bars_only["auroc"]},
        "constant_direction_cv_last_token_proxy": cv,
        "per_layer": {l: {v: analysis[l][v] for v in VARIANTS} for l in analysis},
    }
    with open(DAY1 / "auroc.json", "w") as f:
        json.dump(results, f, indent=1)

    layers = range(37)
    np.savez(
        DAY1 / "directions.npz",
        selected_layer=layer,
        selected_variant=variant,
        d_raw=np.stack([analysis[l]["d"]["raw"] for l in layers]),
        d_cleaned=np.stack([analysis[l]["d"]["cleaned"] for l in layers]),
        r_raw=np.stack([analysis[l]["r_raw"] for l in layers]),
        **{f"{name}_{v}": np.array([analysis[l][v][name] for l in layers])
           for name in ("tau", "harmful_mean_proj", "benign_mean_proj") for v in VARIANTS},
    )
    plot(analysis, DAY1 / "auroc.png", layer)

    fmt = lambda key: f"{best['auroc'][key]:.3f} (95% CI {ci[key][0]:.3f} to {ci[key][1]:.3f})"
    lines = [
        "# Phase 1 summary",
        "",
        f"**Verdict: {call}**",
        "",
        f"Selected layer: hidden-state index {layer} (hook on `model.model.layers[{layer - 1}]`), {variant} direction, chosen on the fit half by the largest margin over the two GO bars and a same-format bar (xstest_unsafe vs xstest_safe >= {bar}).",
        "",
        "AUROC on the eval half at the selected layer:",
        "",
        f"- harmful vs benign: {fmt('harmful_vs_benign')}",
        f"- harmful vs xstest_safe: {fmt('harmful_vs_xstest_safe')}",
        f"- harmful vs orbench_hard: {fmt('harmful_vs_orbench_hard')}",
        f"- xstest_unsafe vs xstest_safe (same format, not a go rule): {fmt(SAME_FORMAT)}",
        "",
        f"False positive rate on the eval half at tau = {best['tau']:.3f} (90% TPR on fit harmful):",
        "",
        *[f"- {n}: {best['fpr'][n]:.3f}" for n in NEGATIVES],
        "",
        f"True positive rate at that tau: harmful eval {best['tpr_eval_harmful']:.3f}, xstest_unsafe eval {best['tpr_eval_xstest_unsafe']:.3f}.",
        "",
        f"For comparison, picking the layer on the eval half itself gives index {oracle_layer} ({oracle_variant}): "
        f"xstest {oracle['auroc']['harmful_vs_xstest_safe']:.3f}, orbench {oracle['auroc']['harmful_vs_orbench_hard']:.3f}, verdict {oracle_call}. This is optimistic and is not the headline.",
        "",
        f"Selecting on the two GO bars alone picks index {bars_layer} ({bars_variant}), which passes them "
        f"(xstest {bars_only['auroc']['harmful_vs_xstest_safe']:.3f}, orbench {bars_only['auroc']['harmful_vs_orbench_hard']:.3f}) "
        f"but gives only {bars_only['auroc'][SAME_FORMAT]:.3f} on the same-format pair, so it was rejected.",
        "",
        "Caution: the go-rule comparisons are partly solvable from surface text alone (see `confound.json`, surface baselines), so the same-format line is the one to trust. It was added to the layer choice after the first run showed the problem.",
    ]
    if rule_gap:
        lines += ["", "Note: xstest clears 0.90 but orbench is below 0.85. The go rules do not name this case; it is reported as WEAK."]
    if oracle_call != call:
        lines += ["", "Note: the verdict depends on how the layer is chosen. Treat it as borderline."]
    with open(DAY1 / "summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
