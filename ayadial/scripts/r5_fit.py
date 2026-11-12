"""Fit a read and its x-space row with v1's recipe. Closed form, no training loop.
  shrunk LDA (0.1) on h (hidden index 22, last template token) -> d ;  z = (h.d - b) / U with b, U from trfit_b, trfit_h ;
  theta0 = 98th percentile of z on trfit_b ;  ridge from x (LayerNorm'd MLP input) to z -> w.   m and q (the position guard) are v1's.
  python scripts/r5_fit.py v1again                 v1's data only: shows that tonight's pipeline reproduces v1 before the data changes
  python scripts/r5_fit.py v2                      + pool GCG and PAIR artifacts + suffix transplants on harmful and on harmless fit prompts
  python scripts/r5_fit.py <name> <extra harmful sets> <extra harmless sets>     (comma-separated set names, '-' for none)
Writes cache_r5/vec_<name>.npz and results/round5/fit/<name>.json (numbers only)."""
import sys
from r5lib import *

FIT_H, FIT_B = ["trfit_h", "harmful_fit", "xstest_unsafe_fit"], ["trfit_b", "benign_fit", "xstest_safe_fit", "orbench_hard_fit"]
EXTRA = {"v1again": ([], []), "v2": (["jbb_GCG_pool", "jbb_PAIR_pool", "aug_gcg_h_fit"], ["aug_gcg_b_fit"])}
DEVS = ["wj_adversarial_harmful", "wj_adversarial_benign", "dev_ho_h", "dev_ho_b", "dev_in_h", "dev_in_b", "harmful", "harmful_jb", "xstest_safe", "orbench_hard", "benign"]
dv = DEV if DEV == "cuda" else "cpu"


def lda(Xh, Xb, shrink=0.1):
    Xh, Xb = torch.tensor(Xh, dtype=torch.float32, device=dv), torch.tensor(Xb, dtype=torch.float32, device=dv)
    mu_h, mu_b = Xh.mean(0), Xb.mean(0)
    C = torch.cat([Xh - mu_h, Xb - mu_b])
    S = C.T @ C / len(C)
    S = S + shrink * S.diagonal().mean() * torch.eye(S.shape[0], device=dv)
    w = torch.linalg.solve(S, mu_h - mu_b)
    return (w / w.norm()).cpu().numpy().astype(np.float64)


def ridge(X, y, lam):
    X, y = torch.tensor(X, dtype=torch.float64, device=dv), torch.tensor(y, dtype=torch.float64, device=dv)
    G = X.T @ X
    G = G + lam * G.diagonal().mean() * torch.eye(G.shape[0], dtype=torch.float64, device=dv)
    return torch.linalg.solve(G, X.T @ y).cpu().numpy()


def main():
    name = sys.argv[1]
    eh, eb = EXTRA[name] if name in EXTRA else ([s for s in sys.argv[2].split(",") if s != "-"], [s for s in sys.argv[3].split(",") if s != "-"])
    H, B = FIT_H + eh, FIT_B + eb
    model, tok = load()
    A = {n: acts(model, tok, n) for n in H + B + DEVS}
    x = {n: A[n][0].astype(np.float64) for n in A}
    h = {n: A[n][1].astype(np.float64) for n in A}
    d = lda(np.concatenate([h[n] for n in H]), np.concatenate([h[n] for n in B]))
    p = {n: h[n] @ d for n in h}
    b = float(p["trfit_b"].mean())
    U = float(p["trfit_h"].mean() - b)
    z = {n: (p[n] - b) / U for n in p}
    theta0 = float(np.quantile(z["trfit_b"], 0.98))
    Xf, yf = np.concatenate([x[n] for n in H + B]), np.concatenate([z[n] for n in H + B])
    Xd, yd = np.concatenate([x[n] for n in DEVS]), np.concatenate([z[n] for n in DEVS])
    best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1):
        w = ridge(Xf, yf, lam)
        r2 = 1 - ((Xd @ w - yd) ** 2).mean() / yd.var()
        if best is None or r2 > best[1]:
            best = (lam, float(r2), w)
    lam, r2, w = best
    V1 = vec("v1")
    cos = lambda a, c: float(a @ c / (np.linalg.norm(a) * np.linalg.norm(c)))
    base = json.loads((R4OUT / "labels/base_dev.json").read_text())
    ans = lambda n: np.array([base[f"{n}|{i}"][0] == 0 for i in A[n][2]])
    out = {"name": name, "fit_harmful_sets": {n: int(len(x[n])) for n in H}, "fit_harmless_sets": {n: int(len(x[n])) for n in B}, "layer": LYR, "shrink": 0.1,
           "b": b, "U": U, "theta0": theta0, "ridge_lambda": lam, "dev_r2_x_vs_h": r2, "norm_w": float(np.linalg.norm(w)), "cos_w_with_v1": cos(w, V1["w"].astype(np.float64)),
           "v1": {"theta0": V1["theta0"], "b": V1["b"], "U": V1["U"], "norm_w": float(np.linalg.norm(V1["w"]))}, "sets": {}, "auroc_answered_x": {}}
    zx = {n: x[n] @ w for n in x}
    z1 = {n: x[n] @ V1["w"].astype(np.float64) for n in x}
    out["corr_z_with_v1_on_dev"] = float(np.corrcoef(np.concatenate([zx[n] for n in DEVS]), np.concatenate([z1[n] for n in DEVS]))[0, 1])
    for pn, (h_, b_) in {"wj": ("wj_adversarial_harmful", "wj_adversarial_benign"), "ho": ("dev_ho_h", "dev_ho_b"), "in": ("dev_in_h", "dev_in_b")}.items():
        out["auroc_answered_x"][pn] = {"this": auroc(zx[h_][ans(h_)], zx[b_][ans(b_)]), "v1_tonight": auroc(z1[h_][ans(h_)], z1[b_][ans(b_)])}
    r4fit = json.loads((R4OUT / "fold/fit.json").read_text())["w"]
    out["auroc_answered_x_in_round4_fit_json"] = {k: r4fit[f"auroc_{k}_ans_x"] for k in ("wj", "ho", "in")}
    for n in H + B + DEVS:
        out["sets"][n] = {"n": int(len(zx[n])), "z_mean": float(zx[n].mean()), "above_theta0": float((zx[n] > theta0).mean()), "v1_above_its_theta0": float((z1[n] > V1["theta0"]).mean())}
    np.savez(CACHE / f"vec_{name}.npz", m=V1["m"], q=V1["q"], w=w.astype(np.float32), d=d.astype(np.float32), theta0=np.float64(theta0), b=np.float64(b), U=np.float64(U))
    (OUT / "fit").mkdir(exist_ok=True)
    (OUT / "fit" / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "sets"}, indent=1))
    for n, v in out["sets"].items():
        print(f"  {n:24s} n {v['n']:5d} z {v['z_mean']:+.2f} above {v['above_theta0']:.3f} (v1 {v['v1_above_its_theta0']:.3f})")
    sync()
    log("R5_FIT DONE")


if __name__ == "__main__":
    main()
