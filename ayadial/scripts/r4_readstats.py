
"""Read statistics for the hook specs: projections of hidden[L] at the last template token on the chosen read."""
import sys
from r4lib import *
kind, L = sys.argv[1], int(sys.argv[2])
R = np.load(CACHE / "reads.npz")
d = R[kind][L - 8, 4]
P = lambda n: np.asarray(np.load(CACHE / f"acts_{n}.npy", mmap_mode="r")[:, L - 8, 4, :], dtype=np.float32) @ d
base = labels("base_dev")
ids = lambda n: json.loads((CACHE / f"ids_{n}.json").read_text())["ids"]
out = {"read": [kind, L, 4], "fit": {}, "dev": {}}
for n in ("trfit_h", "trfit_b", "harmful_fit", "benign_fit", "xstest_safe_fit", "orbench_hard_fit", "xstest_unsafe_fit"):
    p = P(n); out["fit"][n] = {"mean": float(p.mean()), "sd": float(p.std()), "q": {str(q): float(np.quantile(p, q)) for q in (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.98, 0.99)}}
harmless = np.concatenate([P("trfit_b"), P("benign_fit"), P("xstest_safe_fit"), P("orbench_hard_fit")])
out["tau"] = {"b95": float(np.quantile(P("trfit_b"), 0.95)), "b98": float(np.quantile(P("trfit_b"), 0.98)), "b99": float(np.quantile(P("trfit_b"), 0.99)), "harmless98": float(np.quantile(harmless, 0.98))}
out["unit98"] = float(np.median(P("trfit_h")) - out["tau"]["b98"])
for n in ("wj_adversarial_harmful", "wj_adversarial_benign", "dev_ho_h", "dev_ho_b", "dev_in_h", "dev_in_b", "harmful", "harmful_jb", "xstest_safe", "orbench_hard", "benign"):
    p = P(n); ans = np.array([base[f"{n}|{i}"][0] == 0 for i in ids(n)])
    out["dev"][n] = {t: {"n": int(m.sum()), **{f"above_{k}": round(float((p[m] > v).mean()), 3) for k, v in out["tau"].items()}} for t, m in (("answered", ans), ("refused", ~ans))}
(OUT / "step2").mkdir(exist_ok=True)
(OUT / f"step2/read_stats_{kind}_{L}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({"tau": out["tau"], "unit98": out["unit98"], "fit_means": {n: round(v["mean"], 2) for n, v in out["fit"].items()}}))
for n, v in out["dev"].items(): print(n, v)
