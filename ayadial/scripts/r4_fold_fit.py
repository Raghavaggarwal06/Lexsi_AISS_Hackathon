"""Fit the fold in x-space (x = LayerNorm'd input of layers[22].mlp). No training loop: ridge solves only.
Unit (Fable F2, tighten half): gate row = -kappa0*m - kappa*q + s*alpha*w ; up row = m ; down column = gain*r/alpha.
At the marker (m.x=1, q.x=0): push = gain * relu(s*z - theta0) * r with z = w.x ~ 0 on harmless, ~1 on harmful."""
import sys
from r4lib import *
LYR, KIND = 22, "lda_mix"
ALPHA, KAPPA, GAIN = 20.0, 60.0, 2.0
MARKER_WEIGHT = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
d = np.load(CACHE / "reads.npz")[KIND][LYR - 8, 4].astype(np.float64)
FIT = ["trfit_h", "trfit_b", "harmful_fit", "benign_fit", "xstest_safe_fit", "orbench_hard_fit", "xstest_unsafe_fit"]
DEVS = ["wj_adversarial_harmful", "wj_adversarial_benign", "dev_ho_h", "dev_ho_b", "dev_in_h", "dev_in_b", "harmful", "harmful_jb", "xstest_safe", "orbench_hard", "benign"]
T = lambda n: np.load(CACHE / f"x{LYR}_{n}.npz")
dv = "cuda"


def ridge(X, y, lam):
    X = torch.tensor(X, dtype=torch.float64, device=dv)
    y = torch.tensor(y, dtype=torch.float64, device=dv)
    G = X.T @ X
    G = G + lam * G.diagonal().mean() * torch.eye(G.shape[0], dtype=torch.float64, device=dv)
    return torch.linalg.solve(G, X.T @ y).cpu().numpy()


xm = {n: T(n)["x"][:, 4].astype(np.float64) for n in FIT + DEVS}
pm = {n: T(n)["h"][:, 4].astype(np.float64) @ d for n in FIT + DEVS}
b = float(pm["trfit_b"].mean())
U = float(pm["trfit_h"].mean() - b)
z = {n: (pm[n] - b) / U for n in pm}
harmless_fit = np.concatenate([z[n] for n in ("trfit_b", "benign_fit", "xstest_safe_fit", "orbench_hard_fit")])
theta0 = float(np.quantile(z["trfit_b"], 0.98))
out = {"layer": LYR, "read": KIND, "alpha": ALPHA, "kappa": KAPPA, "gain": GAIN, "b": b, "U": U, "theta0": theta0, "kappa0": ALPHA * theta0,
       "harmless_fit_above_theta0": float((harmless_fit > theta0).mean())}

# w: x -> z at the marker
Xf, yf = np.concatenate([xm[n] for n in FIT]), np.concatenate([z[n] for n in FIT])
best = None
for lam in (1e-4, 1e-3, 1e-2, 1e-1):
    w = ridge(Xf, yf, lam)
    Xd, yd = np.concatenate([xm[n] for n in DEVS]), np.concatenate([z[n] for n in DEVS])
    r2 = 1 - ((Xd @ w - yd) ** 2).mean() / yd.var()
    if best is None or r2 > best[1]:
        best = (lam, float(r2), w)
lam_w, r2, w = best
out["w"] = {"lambda": lam_w, "dev_r2_vs_h_space_z": r2, "norm": float(np.linalg.norm(w))}
base = labels("base_dev")
ans = lambda n: np.array([base[f"{n}|{i}"][0] == 0 for i in T(n)["ids"]])
for pn, (h_, b_) in {"wj": ("wj_adversarial_harmful", "wj_adversarial_benign"), "ho": ("dev_ho_h", "dev_ho_b"), "in": ("dev_in_h", "dev_in_b")}.items():
    out["w"][f"auroc_{pn}_ans_x"] = auroc((xm[h_] @ w)[ans(h_)], (xm[b_] @ w)[ans(b_)])
    out["w"][f"auroc_{pn}_ans_h"] = auroc(z[h_][ans(h_)], z[b_][ans(b_)])

# m, q: marker vs every other position
pf, pd_ = np.load(CACHE / f"xpos{LYR}_fit.npz"), np.load(CACHE / f"xpos{LYR}_dev.npz")
kf, kd = pf["meta"][:, 1], pd_["meta"][:, 1]
Xp = np.concatenate([pf["x"].astype(np.float64), Xf[:: 4]])
ym = np.concatenate([(kf == 6).astype(np.float64), np.ones(len(Xf[:: 4]))])
bestm = None
for lam in (1e-4, 1e-3, 1e-2):
    sw = np.sqrt(np.where(ym > 0.5, MARKER_WEIGHT, 1.0))
    m = ridge(Xp * sw[:, None], ym * sw, lam)
    q = ridge(Xp * sw[:, None], (1 - ym) * sw, lam)
    xd = pd_["x"].astype(np.float64)
    margin = -float((xd[kd == 6] @ q).std())  # v2: choose the guard that is quietest at the marker
    if bestm is None or margin > bestm[1]:
        bestm = (lam, margin, m, q)
lam_m, margin, m, q = bestm
xd = pd_["x"].astype(np.float64)
KN = {0: "preamble", 1: "user", 2: "last_instruction", 3: "END_OF_TURN", 4: "START_OF_TURN", 5: "CHATBOT", 6: "START_RESPONSE(marker)", 7: "generated"}
out["marker"] = {"lambda": lam_m, "marker_weight": MARKER_WEIGHT, "q_sd_at_marker": -margin, "dev_margin_m": float((xd[kd == 6] @ m).min() - (xd[kd != 6] @ m).max()), "dev_margin_q": float((xd[kd != 6] @ q).min() - (xd[kd == 6] @ q).max()), "n_dev_positions": int(len(kd)),
                 "m_by_kind": {KN[k]: [float(v) for v in np.percentile(xd[kd == k] @ m, [0, 1, 50, 99, 100])] for k in sorted(set(kd.tolist()))},
                 "q_by_kind": {KN[k]: [float(v) for v in np.percentile(xd[kd == k] @ q, [0, 1, 50, 99, 100])] for k in sorted(set(kd.tolist()))}}
# what the unit writes off the marker, in units of ||r||, for each s
silu = lambda v: v / (1 + np.exp(-np.clip(v, -60, 60)))
out["off_marker_write"] = {}
for s in (1, 2, 3, -3):
    g = -ALPHA * theta0 * (xd @ m) - KAPPA * (xd @ q) + s * ALPHA * (xd @ w)
    wr = GAIN * silu(g) * (xd @ m) / ALPHA
    off, on = np.abs(wr[kd != 6]), wr[kd == 6]
    out["off_marker_write"][str(s)] = {"off_max": float(off.max()), "off_p999": float(np.quantile(off, 0.999)), "off_frac_above_0.01": float((off > 0.01).mean()),
                                       "on_mean": float(on.mean()), "on_p50": float(np.median(on)), "worst_kind": KN[int(kd[kd != 6][off.argmax()])]}
# what the unit writes at the marker for each dev set (mean push in units of r) and how it tracks the h-space hinge
out["marker_push"] = {}
for s in (1, 2, 3, -1, -3):
    row = {}
    for n in DEVS:
        x = xm[n]
        g = -ALPHA * theta0 * (x @ m) - KAPPA * (x @ q) + s * ALPHA * (x @ w)
        wr = GAIN * silu(g) * (x @ m) / ALPHA
        ref = GAIN * np.clip(s * z[n] - theta0, 0, None)
        row[n] = {"mean_push": float(wr.mean()), "frac_pushed_gt_0.1": float((wr > 0.1).mean()), "corr_with_h_hinge": float(np.corrcoef(wr, ref)[0, 1]) if ref.std() > 0 else None, "mean_h_hinge": float(ref.mean())}
    out["marker_push"][str(s)] = row
np.savez(CACHE / "fold.npz", m=m.astype(np.float32), q=q.astype(np.float32), w=w.astype(np.float32), consts=np.array([ALPHA, KAPPA, GAIN, theta0, b, U], np.float64))
(OUT / "fold").mkdir(exist_ok=True)
(OUT / "fold/fit.json").write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k not in ("marker_push", "marker", "off_marker_write")})
print("marker", {k: v for k, v in out["marker"].items() if not k.endswith("by_kind")})
for k in out["marker"]["m_by_kind"]: print("  ", k, "m", [round(v, 2) for v in out["marker"]["m_by_kind"][k]], "q", [round(v, 2) for v in out["marker"]["q_by_kind"][k]])
for k, v in out["off_marker_write"].items(): print("off", k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
for s in ("1", "2", "-3"):
    print(s, {n: (round(v["mean_push"], 3), round(v["frac_pushed_gt_0.1"], 3), None if v["corr_with_h_hinge"] is None else round(v["corr_with_h_hinge"], 3)) for n, v in out["marker_push"][s].items()})
sync()
