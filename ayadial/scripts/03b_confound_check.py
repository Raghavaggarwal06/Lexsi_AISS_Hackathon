"""Phase 1 check: is the harm direction reading harm, or the style of each dataset?

XSTest safe and unsafe share one format (short questions), and JBB harmful and benign are paired
by topic, so AUROC inside those pairs cannot be explained by format. Surface-text baselines show
how far each comparison can be solved without the model. Writes results/day1/confound.json.
"""
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import DAY1, SETS, load_config, load_set, load_splits

POSITIVE, NEGATIVE = ["harmful", "xstest_unsafe"], ["benign", "xstest_safe", "orbench_hard"]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def auroc(pos, neg):
    return float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))


def load():
    """{set: {half: {"acts", "text", "source"}}}"""
    splits, data = load_splits(), {}
    for name in SETS:
        rows = load_set(name)
        where = {r["id"]: i for i, r in enumerate(rows)}
        acts = np.load(DAY1 / f"acts_{name}.npy").astype(np.float32)
        data[name] = {}
        for half in ("fit", "eval"):
            idx = [where[i] for i in splits[name][half]]
            data[name][half] = {
                "acts": acts[idx],
                "text": [rows[i]["prompt"] for i in idx],
                "source": np.array([rows[i]["source"] for i in idx]),
            }
    return data


def pairs(data, layer, d):
    """Eval-half AUROC of direction d on each comparison that matters."""
    score = lambda name, half="eval": data[name][half]["acts"][:, layer] @ d
    harm, ben = score("harmful"), score("benign")
    harm_src, ben_src = data["harmful"]["eval"]["source"], data["benign"]["eval"]["source"]
    return {
        "same_format_xstest_unsafe_vs_safe": auroc(score("xstest_unsafe"), score("xstest_safe")),
        "paired_jbb_harmful_vs_jbb_benign": auroc(harm[harm_src == "jbb"], ben[ben_src == "jbb"]),
        "harmful_vs_jbb_benign": auroc(harm, ben[ben_src == "jbb"]),
        "harmful_vs_alpaca": auroc(harm, ben[ben_src == "alpaca"]),
        "harmful_vs_xstest_safe": auroc(harm, score("xstest_safe")),
        "harmful_vs_orbench_hard": auroc(harm, score("orbench_hard")),
        "xstest_unsafe_vs_orbench_hard": auroc(score("xstest_unsafe"), score("orbench_hard")),
    }


def class_gap(data, positives, negatives, layer):
    """Difference of class means on the fit half, each set weighted equally within its class."""
    pos = np.mean([data[n]["fit"]["acts"][:, layer].mean(0) for n in positives], axis=0)
    neg = np.mean([data[n]["fit"]["acts"][:, layer].mean(0) for n in negatives], axis=0)
    return unit(pos - neg)


def surface_baselines(data):
    """AUROC from prompt length alone and from TF-IDF logistic regression (no model involved)."""
    out = {}
    for pos, neg in [("harmful", "xstest_safe"), ("harmful", "orbench_hard"), ("harmful", "benign"), ("xstest_unsafe", "xstest_safe")]:
        fit_text = data[pos]["fit"]["text"] + data[neg]["fit"]["text"]
        fit_y = np.r_[np.ones(len(data[pos]["fit"]["text"])), np.zeros(len(data[neg]["fit"]["text"]))]
        ev_text = data[pos]["eval"]["text"] + data[neg]["eval"]["text"]
        ev_y = np.r_[np.ones(len(data[pos]["eval"]["text"])), np.zeros(len(data[neg]["eval"]["text"]))]
        length = roc_auc_score(ev_y, [len(t) for t in ev_text])
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        clf = LogisticRegression(max_iter=2000).fit(vec.fit_transform(fit_text), fit_y)
        tfidf = roc_auc_score(ev_y, clf.decision_function(vec.transform(ev_text)))
        out[f"{pos}_vs_{neg}"] = {"length_only": float(max(length, 1 - length)), "tfidf_logreg": float(tfidf)}
    return out


def main():
    cfg = load_config()
    data = load()
    saved = np.load(DAY1 / "directions.npz")
    selected = int(saved["selected_layer"])
    lo, hi = cfg["layers"]["min"], cfg["layers"]["max"]
    d_by_name = {
        "harm_direction_as_built": lambda l: saved["d_raw"][l],
        "within_xstest_only": lambda l: class_gap(data, ["xstest_unsafe"], ["xstest_safe"], l),
        "pooled_all_sets": lambda l: class_gap(data, POSITIVE, NEGATIVE, l),
    }
    per_layer = {
        name: {l: pairs(data, l, fn(l)) for l in range(lo, hi + 1)} for name, fn in d_by_name.items()
    }
    key = "same_format_xstest_unsafe_vs_safe"
    best_same_format = {
        name: max(table, key=lambda l: table[l][key]) for name, table in per_layer.items()
    }
    results = {
        "selected_layer": selected,
        "at_selected_layer": {name: table[selected] for name, table in per_layer.items()},
        "best_layer_by_same_format_auroc_eval_optimistic": {
            name: {"layer": l, **per_layer[name][l]} for name, l in best_same_format.items()
        },
        "surface_baselines": surface_baselines(data),
        "per_layer": per_layer,
    }
    with open(DAY1 / "confound.json", "w") as f:
        json.dump(results, f, indent=1)

    print(json.dumps({k: results[k] for k in list(results)[:4]}, indent=1))


if __name__ == "__main__":
    main()
