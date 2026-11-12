"""Step 2: score reads of adversarial_harmful vs adversarial_benign. Forward-pass caches only."""
import sys
from r4lib import *
LAYERS = list(range(8, 31))
A = lambda n: np.load(CACHE / f"acts_{n}.npy", mmap_mode="r")
IDS = lambda n: json.loads((CACHE / f"ids_{n}.json").read_text())["ids"]
HAVE_FIT = (OUT / "labels/base_fit.json").exists()
base = {**labels("base_dev"), **(labels("base_fit") if HAVE_FIT else {})}
TAG = "" if HAVE_FIT else "_prelim"
guard = json.loads((CACHE / "guard.json").read_text())
old = np.load(ROOT / "results/day1/directions.npz")
dv = "cuda"


def answered(name):
    return np.array([base.get(f"{name}|{i}", [0, 0, 0])[0] == 0 and base.get(f"{name}|{i}", [0, 0, 0])[2] == 0 for i in IDS(name)])


def X(name, li, p):
    return torch.tensor(np.asarray(A(name)[:, li, p, :]), dtype=torch.float32, device=dv)


def lda(Xh, Xb, shrink=0.1):
    mu_h, mu_b = Xh.mean(0), Xb.mean(0)
    C = torch.cat([Xh - mu_h, Xb - mu_b])
    S = C.T @ C / len(C)
    S = S + shrink * S.diagonal().mean() * torch.eye(S.shape[0], device=dv)
    w = torch.linalg.solve(S, mu_h - mu_b)
    return w / w.norm()


PAIRS = {"wj": ("wj_adversarial_harmful", "wj_adversarial_benign"), "ho": ("dev_ho_h", "dev_ho_b"), "in": ("dev_in_h", "dev_in_b"),
         "plain": ("harmful", "benign"), "xs": ("xstest_unsafe", "xstest_safe"), "orb": ("harmful", "orbench_hard")}
ANS = {n: answered(n) for pr in PAIRS.values() for n in pr}
FIT_ANS = {n: answered(n) for n in ("trfit_h", "trfit_b")}
KINDS = ["d_old", "dm", "lda", "lda_ans", "lda_mix", "shuf"]
res = {k: {f"{p}{s}": np.zeros((len(LAYERS), 5)) for p in PAIRS for s in ("", "_ans")} for k in KINDS}
W = {k: np.zeros((len(LAYERS), 5, 2048), dtype=np.float32) for k in KINDS}
g = torch.Generator(device="cpu").manual_seed(0)
for li, L in enumerate(LAYERS):
    for p in range(5):
        Xh, Xb = X("trfit_h", li, p), X("trfit_b", li, p)
        ws = {"d_old": torch.tensor(old["d_cleaned"][L], dtype=torch.float32, device=dv), "dm": Xh.mean(0) - Xb.mean(0), "lda": lda(Xh, Xb),
              "lda_ans": lda(Xh[torch.tensor(FIT_ANS["trfit_h"])], Xb[torch.tensor(FIT_ANS["trfit_b"])])}
        Mh = torch.cat([Xh, X("harmful_fit", li, p), X("xstest_unsafe_fit", li, p)])
        Mb = torch.cat([Xb, X("benign_fit", li, p), X("xstest_safe_fit", li, p), X("orbench_hard_fit", li, p)])
        ws["lda_mix"] = lda(Mh, Mb)
        allx = torch.cat([Xh, Xb])
        perm = torch.randperm(len(allx), generator=g)
        ws["shuf"] = lda(allx[perm[: len(Xh)]], allx[perm[len(Xh):]])
        ev = {n: X(n, li, p) for pr in PAIRS.values() for n in pr}
        for k, w in ws.items():
            w = w / w.norm()
            W[k][li, p] = w.cpu().numpy()
            for pn, (h, b) in PAIRS.items():
                sh_, sb_ = (ev[h] @ w).cpu().numpy(), (ev[b] @ w).cpu().numpy()
                res[k][pn][li, p] = auroc(sh_, sb_)
                res[k][pn + "_ans"][li, p] = auroc(sh_[ANS[h]], sb_[ANS[b]])
    log(f"layer {L} lda ho_ans={res['lda']['ho_ans'][li].round(3).tolist()}")
np.savez(CACHE / f"reads{TAG}.npz", **W)

out = {"lda_ans_valid": HAVE_FIT, "layers": LAYERS, "positions": ["last_instruction", "END_OF_TURN", "START_OF_TURN", "CHATBOT", "START_RESPONSE"],
       "n": {n: {"all": int(len(a)), "base_answered": int(a.sum())} for n, a in {**ANS, **FIT_ANS}.items()}, "grids": {k: {m: v.round(4).tolist() for m, v in d.items()} for k, d in res.items()}, "cells": {}}
rng = np.random.default_rng(0)
for k in KINDS:
    sel = (res[k]["in_ans"] + res[k]["wj_ans"]) / 2
    li, p = np.unravel_index(sel.argmax(), sel.shape)
    w = torch.tensor(W[k][li, p], device=dv)
    sh_, sb_ = (X("dev_ho_h", li, p) @ w).cpu().numpy()[ANS["dev_ho_h"]], (X("dev_ho_b", li, p) @ w).cpu().numpy()[ANS["dev_ho_b"]]
    boot = [auroc(sh_[rng.integers(0, len(sh_), len(sh_))], sb_[rng.integers(0, len(sb_), len(sb_))]) for _ in range(1000)]
    out["cells"][k] = {"layer": LAYERS[li], "position": out["positions"][p], "selection_score": float(sel[li, p]),
                       **{m: float(res[k][m][li, p]) for m in res[k]}, "ho_ans_ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
                       "best_ho_ans_any_cell_optimistic": float(res[k]["ho_ans"].max()),
                       "at_L23_last": {m: float(res[k][m][LAYERS.index(23), 4]) for m in ("wj", "wj_ans", "ho", "ho_ans", "in_ans", "plain", "xs", "orb")}}
gs = lambda n, m=None: np.array([guard[f"{n}|{i}"] for i in IDS(n)], dtype=float)[slice(None) if m is None else m]
out["guard_qwen3guard_prompt_mode"] = {}
for pn, (h, b) in PAIRS.items():
    out["guard_qwen3guard_prompt_mode"][pn] = auroc(np.nan_to_num(gs(h), nan=0.5), np.nan_to_num(gs(b), nan=0.5))
    out["guard_qwen3guard_prompt_mode"][pn + "_ans"] = auroc(np.nan_to_num(gs(h, ANS[h]), nan=0.5), np.nan_to_num(gs(b, ANS[b]), nan=0.5))
    out["guard_qwen3guard_prompt_mode"][pn + "_unsafe_rate_h_b"] = [float(np.nanmean(gs(h) == 1)), float(np.nanmean(gs(b) == 1))]
(OUT / "step2").mkdir(exist_ok=True)
(OUT / f"step2/read_auroc{TAG}.json").write_text(json.dumps(out, indent=1))
for k in KINDS:
    c = out["cells"][k]
    print(k, c["layer"], c["position"], "| wj", round(c["wj"], 3), round(c["wj_ans"], 3), "| in", round(c["in"], 3), round(c["in_ans"], 3), "| HO", round(c["ho"], 3), round(c["ho_ans"], 3), [round(x, 3) for x in c["ho_ans_ci95"]],
          "| plain", round(c["plain"], 3), "xs", round(c["xs"], 3), "orb", round(c["orb"], 3), "| best-any", round(c["best_ho_ans_any_cell_optimistic"], 3))
    print("   L23 last:", {m: round(v, 3) for m, v in c["at_L23_last"].items()})
print("guard", {m: (round(v, 3) if not isinstance(v, list) else [round(x, 3) for x in v]) for m, v in out["guard_qwen3guard_prompt_mode"].items()})
print("n", out["n"])
sync()
