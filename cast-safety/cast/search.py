"""Best condition point (paper App C.2; release MalleableModel.find_best_condition_point).

Over single layers (max_layers_to_combine=1, as used in the paper), thresholds on a fixed grid and both comparison
directions, pick the configuration with the highest F1 of "condition met" against positive=1 / negative=0.
Ties keep the first configuration in the release's iteration order (layer ascending, threshold ascending,
release direction 'larger' (sim < theta) before 'smaller' (sim > theta)), with a strict '>' improvement test.
"""
import numpy as np


def f1_grid(sims_pos, sims_neg, thresholds):
    """F1 for each threshold, for met = sim < t ('<') and met = sim > t ('>'). Returns (f1_lt, f1_gt)."""
    s = np.concatenate([sims_pos, sims_neg])
    y = np.concatenate([np.ones(len(sims_pos), bool), np.zeros(len(sims_neg), bool)])
    P = len(sims_pos)
    out = []
    for lt in (True, False):
        pred = (s[None, :] < thresholds[:, None]) if lt else (s[None, :] > thresholds[:, None])
        tp = (pred & y[None, :]).sum(1)
        fp = (pred & ~y[None, :]).sum(1)
        fn = P - tp
        denom = 2 * tp + fp + fn
        out.append(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
    return out


def best_condition_point(sims_pos, sims_neg, layers, threshold_range=None, step=1e-4):
    """sims_*: {layer: (N,) array}. threshold_range None = span the observed similarities at all searched layers."""
    if threshold_range is None:
        allv = np.concatenate([np.concatenate([sims_pos[l], sims_neg[l]]) for l in layers])
        lo, hi = np.floor(allv.min() / step) * step, np.ceil(allv.max() / step) * step + step
    else:
        lo, hi = threshold_range
    ths = np.arange(lo, hi, step)
    best, table = (0.0, None), {}
    for l in layers:
        f_lt, f_gt = f1_grid(sims_pos[l], sims_neg[l], ths)
        table[l] = {"<": (float(f_lt.max()), float(ths[f_lt.argmax()])), ">": (float(f_gt.max()), float(ths[f_gt.argmax()]))}
        for i, t in enumerate(ths):          # release order: per threshold, 'larger' (<) then 'smaller' (>)
            for f, cmp in ((f_lt[i], "<"), (f_gt[i], ">")):
                if f > best[0]:
                    best = (float(f), (int(l), float(round(t, 6)), cmp))
    f1, (layer, th, cmp) = best
    return {"layer": layer, "threshold": th, "cmp": cmp, "f1": f1,
            "release_direction": "larger" if cmp == "<" else "smaller",
            "range": [float(lo), float(hi)], "step": step, "per_layer": table}
