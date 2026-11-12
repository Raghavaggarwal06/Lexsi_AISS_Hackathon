"""Loosen read and the dataset-identity confound (charter item; round 3 test 3.1). CPU/GPU-light, cached activations only.
Question: does a read separate harmless prompts the base REFUSES from harmful prompts the base refuses, when both come from the same dataset?
Round 3's read was fit on our own sets, where refused-harmless (XSTest, OR-Bench) and harmful (HarmBench, JBB) are different datasets."""
from r4lib import *
LAYERS = list(range(8, 31))
A = lambda n: np.load(CACHE / f"acts_{n}.npy", mmap_mode="r")
IDS = lambda n: json.loads((CACHE / f"ids_{n}.json").read_text())["ids"]
base = {**labels("base_dev"), **labels("base_fit")}
refused = lambda n: np.array([base[f"{n}|{i}"][0] == 1 for i in IDS(n)])
X = lambda n, L, p, m=None: torch.tensor(np.asarray(A(n)[:, L - 8, p, :]), dtype=torch.float32, device="cuda")[slice(None) if m is None else torch.tensor(m)]
R = np.load(CACHE / "reads.npz")
POS = ["last_instruction", "END_OF_TURN", "START_OF_TURN", "CHATBOT", "START_RESPONSE"]


def ridge_read(pos_sets, zero_sets, L, p, lam=1e-2):
    Xp = torch.cat([X(n, L, p, m) for n, m in pos_sets]); Xz = torch.cat([X(n, L, p, m) for n, m in zero_sets])
    Xa = torch.cat([Xp, Xz]); mu = Xa.mean(0)
    y = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xz))]).to(Xa.device)
    Xc = Xa - mu
    G = Xc.T @ Xc
    w = torch.linalg.solve(G + lam * G.diagonal().mean() * torch.eye(G.shape[0], device=Xa.device), Xc.T @ (y - y.mean()))
    return w, mu


own_pos = [(n, refused(n)) for n in ("xstest_safe_fit", "orbench_hard_fit", "benign_fit")]
own_zero = [("harmful_fit", None)]
wj_pos, wj_zero = [("trfit_b", refused("trfit_b"))], [("trfit_h", None)]
pairs = {"own (cross-dataset, as in round 3)": ([("xstest_safe", refused("xstest_safe")), ("orbench_hard", refused("orbench_hard"))], [("harmful", refused("harmful"))]),
         "WildJailbreak eval rows (same dataset)": ([("wj_adversarial_benign", refused("wj_adversarial_benign"))], [("wj_adversarial_harmful", refused("wj_adversarial_harmful"))]),
         "held-out clusters (same dataset)": ([("dev_ho_b", refused("dev_ho_b"))], [("dev_ho_h", refused("dev_ho_h"))])}
out = {"n": {k: [int(sum(m.sum() for _, m in a)), int(sum(m.sum() for _, m in b))] for k, (a, b) in pairs.items()}, "n_fit_positives": {"own": int(sum(m.sum() for _, m in own_pos)), "wj": int(refused("trfit_b").sum())}, "reads": {}}
for name, (ps, zs) in {"fit on our own sets (round 3 recipe)": (own_pos, own_zero), "fit on WildJailbreak train": (wj_pos, wj_zero), "fit on both": (own_pos + wj_pos, own_zero + wj_zero)}.items():
    grid = {k: np.zeros((len(LAYERS), 5)) for k in pairs}
    for li, L in enumerate(LAYERS):
        for p in range(5):
            w, mu = ridge_read(ps, zs, L, p)
            for k, (a, b) in pairs.items():
                sa = torch.cat([(X(n, L, p, m) - mu) @ w for n, m in a]).cpu().numpy(); sb_ = torch.cat([(X(n, L, p, m) - mu) @ w for n, m in b]).cpu().numpy()
                grid[k][li, p] = auroc(sa, sb_)
    sel = grid["own (cross-dataset, as in round 3)"] if name.startswith("fit on our own") else grid["held-out clusters (same dataset)"]
    li, p = np.unravel_index(sel.argmax(), sel.shape)
    out["reads"][name] = {"cell_chosen_on": "own pair" if name.startswith("fit on our own") else "held-out clusters", "layer": LAYERS[li], "position": POS[p], **{k: float(g[li, p]) for k, g in grid.items()},
                          "at_layer24_END_OF_TURN": {k: float(g[LAYERS.index(24), 1]) for k, g in grid.items()}, "at_layer22_START_RESPONSE": {k: float(g[LAYERS.index(22), 4]) for k, g in grid.items()}}
# the tighten read, inverted (refused-harmless should score LOW)
d = torch.tensor(R["lda_mix"][22 - 8, 4], device="cuda")
out["reads"]["tighten read lda_mix@22, inverted"] = {k: auroc(torch.cat([-(X(n, 22, 4, m) @ d) for n, m in a]).cpu().numpy(), torch.cat([-(X(n, 22, 4, m) @ d) for n, m in b]).cpu().numpy()) for k, (a, b) in pairs.items()}
(OUT / "controls").mkdir(exist_ok=True)
(OUT / "controls/loosen_read_identity.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
sync()
