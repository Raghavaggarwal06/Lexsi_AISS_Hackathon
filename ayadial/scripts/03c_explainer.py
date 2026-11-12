"""Phase 1 explainer: builds results/day1/go_explained.html, one self-contained page.

Every dot on the page is a held-out (eval-half) prompt. The 2D map is PCA fitted on the fit half
with no labels. The danger score rescales the raw harm projection so that 0 is the typical benign
prompt and 1 is the typical harmful prompt, the same rescaling candidate A uses.

Harmful prompt text stays out of the shareable page, and the script checks that before writing
it. --include-harmful writes a private copy to runs/, which is never published.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from common import DAY1, ROOT as ROOT_DIR, SETS, load_set, load_splits

ORDER = ["harmful", "xstest_unsafe", "orbench_hard", "xstest_safe", "benign"]
WITHHELD = {"harmful", "xstest_unsafe"}
REFERENCE_LAYER = 23
LAYERS = range(1, 37)


def load_all():
    splits, acts, rows = load_splits(), {}, {}
    for name in SETS:
        table = load_set(name)
        where = {r["id"]: i for i, r in enumerate(table)}
        array = np.load(DAY1 / f"acts_{name}.npy").astype(np.float32)
        acts[name] = {h: array[[where[i] for i in splits[name][h]]] for h in ("fit", "eval")}
        rows[name] = [table[where[i]] for i in splits[name]["eval"]]
    return acts, rows


def pca_coords(acts, layer):
    """Two label-free axes fitted on the fit half. Returns fit coords, eval coords, variance share."""
    fit = np.concatenate([acts[n]["fit"][:, layer] for n in ORDER])
    held = np.concatenate([acts[n]["eval"][:, layer] for n in ORDER])
    mean = fit.mean(0)
    _, s, vt = np.linalg.svd(fit - mean, full_matrices=False)
    fit_xy, eval_xy = (fit - mean) @ vt[:2].T, (held - mean) @ vt[:2].T
    scale = np.percentile(np.abs(fit_xy), 99)
    return fit_xy / scale, eval_xy / scale, float((s[:2] ** 2).sum() / (s**2).sum())


def align(source, target):
    """Rotation or reflection that best lines up source with target, so frames do not jump."""
    u, _, vt = np.linalg.svd(source.T @ target)
    return u @ vt


def build_map(acts, n_harmful_eval):
    """Per-layer 2D coordinates, chained outward from the reference layer for smooth animation."""
    coords, variance = {}, {}
    fit_ref, eval_ref, variance[REFERENCE_LAYER] = pca_coords(acts, REFERENCE_LAYER)
    flip = np.diag([-1.0 if eval_ref[:n_harmful_eval, 0].mean() < 0 else 1.0, 1.0])
    fit_ref, eval_ref = fit_ref @ flip, eval_ref @ flip
    coords[REFERENCE_LAYER] = eval_ref
    for direction in (1, -1):
        previous = fit_ref
        layer = REFERENCE_LAYER + direction
        while layer in LAYERS:
            fit_xy, eval_xy, variance[layer] = pca_coords(acts, layer)
            rotation = align(fit_xy, previous)
            previous = fit_xy @ rotation
            coords[layer] = eval_xy @ rotation
            layer += direction
    return coords, variance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-harmful", action="store_true", help="private copy with harmful prompt text, in runs/")
    args = ap.parse_args()
    acts, rows = load_all()
    saved = np.load(DAY1 / "directions.npz")
    auroc = json.load(open(DAY1 / "auroc.json"))
    confound = json.load(open(DAY1 / "confound.json"))
    coords, variance = build_map(acts, len(rows["harmful"]))

    held = {n: acts[n]["eval"] for n in ORDER}
    group = [i for i, n in enumerate(ORDER) for _ in rows[n]]
    text = [None if n in WITHHELD and not args.include_harmful else r["prompt"] for n in ORDER for r in rows[n]]
    data = {
        "groups": ORDER, "group": group, "text": text, "map": {}, "score": {}, "tau": {},
        "past": {}, "variance": {}, "auroc": {},
    }
    for layer in LAYERS:
        benign_mean = float(saved["benign_mean_proj_raw"][layer])
        spread = float(saved["harmful_mean_proj_raw"][layer]) - benign_mean
        d, tau = saved["d_raw"][layer], float(saved["tau_raw"][layer])
        scores = {n: (held[n][:, layer] @ d - benign_mean) / spread for n in ORDER}
        tau_scaled = (tau - benign_mean) / spread
        data["score"][layer] = [round(float(x), 3) for n in ORDER for x in scores[n]]
        data["map"][layer] = [round(float(x), 3) for x in coords[layer].ravel()]
        data["tau"][layer] = round(tau_scaled, 3)
        data["past"][layer] = [round(float((scores[n] > tau_scaled).mean()), 4) for n in ORDER]
        data["variance"][layer] = round(variance[layer], 3)
        data["auroc"][layer] = {k: round(v, 3) for k, v in auroc["per_layer"][str(layer)]["raw"]["auroc"].items()}

    expected = auroc["per_layer"][str(REFERENCE_LAYER)]["raw"]["fpr"]
    for n, value in expected.items():
        past = data["past"][REFERENCE_LAYER][ORDER.index(n)]
        assert abs(past - value) < 1e-3, f"{n}: page says {past}, auroc.json says {value}"

    layer = REFERENCE_LAYER
    tau_scaled = data["tau"][layer]
    d = saved["d_raw"][layer]
    benign_mean = float(saved["benign_mean_proj_raw"][layer])
    spread = float(saved["harmful_mean_proj_raw"][layer]) - benign_mean
    scored = {n: (held[n][:, layer] @ d - benign_mean) / spread for n in ORDER}
    data["flagged"] = {}
    for n in ("orbench_hard", "benign", "xstest_safe"):
        order = np.argsort(-scored[n])
        hits = [i for i in order if scored[n][i] > tau_scaled]
        data["flagged"][n] = {
            "count": len(hits), "total": len(rows[n]),
            "top": [[round(float(scored[n][i]), 2), rows[n][i]["prompt"]] for i in hits[:8]],
        }
    data["missed"] = {n: [int((scored[n] <= tau_scaled).sum()), len(rows[n])] for n in WITHHELD}
    data["compare"] = {v: {k: auroc["per_layer"][str(layer)][v][k] for k in ("auroc", "fpr", "tpr_eval_harmful", "tpr_eval_xstest_unsafe")} for v in ("raw", "cleaned")}
    data["tfidf"] = {k: round(v["tfidf_logreg"], 3) for k, v in confound["surface_baselines"].items()}

    template = (Path(__file__).parent / "explainer_template.html").read_text(encoding="utf8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("/*__DATA__*/null", payload)
    if not args.include_harmful:
        for name in WITHHELD:
            for row in load_set(name):
                assert row["prompt"] not in html, f"harmful text leaked: {row['id']}"
    out = (ROOT_DIR / "runs" / "go_explained_local.html") if args.include_harmful else DAY1 / "go_explained.html"
    out.write_text(html, encoding="utf8")
    print(f"wrote {out} ({len(html) / 1024:.0f} KB, {len(group)} prompts, {len(data['map'])} layers)")


if __name__ == "__main__":
    main()
