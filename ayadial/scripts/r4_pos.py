"""Fold inputs: x22 = LayerNorm'd input of layers[22].mlp and h22, at the 5 tail positions for every fit/dev prompt,
plus x22 at sampled other positions (preamble, user text, generated text) for a subset. Forward passes only."""
import sys
from r4lib import *
FIT = ["trfit_h", "trfit_b", "harmful_fit", "benign_fit", "xstest_safe_fit", "orbench_hard_fit", "xstest_unsafe_fit"]
DEVS = ["wj_adversarial_harmful", "wj_adversarial_benign", "dev_ho_h", "dev_ho_b", "dev_in_h", "dev_in_b", "harmful", "harmful_jb", "xstest_safe", "orbench_hard", "benign"]
LYR = int(sys.argv[1]) if len(sys.argv) > 1 else 22
model, tok = load()
store = {}
mlp = model.model.layers[LYR].mlp
hk = [mlp.register_forward_pre_hook(lambda m, a: store.__setitem__("x", a[0])), mlp.down_proj.register_forward_pre_hook(lambda m, a: store.__setitem__("m", a[0])),
      model.model.layers[LYR - 1].register_forward_hook(lambda m, i, o: store.__setitem__("h", o[0] if isinstance(o, tuple) else o))]
allrows = sets()


@torch.no_grad()
def tails(rows, batch=32):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]))
    X, H = np.zeros((len(rows), 5, 2048), np.float16), np.zeros((len(rows), 5, 2048), np.float16)
    M = np.zeros((len(rows), 11008), np.float16)
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        model.model(**enc(tok, [rows[i]["prompt"] for i in idx]), use_cache=False)
        X[idx], H[idx], M[idx] = store["x"][:, -5:].float().cpu().numpy(), store["h"][:, -5:].float().cpu().numpy(), store["m"][:, -1].float().cpu().numpy()
    return X, H, M


for name in FIT + DEVS:
    path = CACHE / f"x{LYR}_{name}.npz"
    if not path.exists():
        rows = [r for r in allrows if r["set"] == name]
        X, H, M = tails(rows)
        np.savez(path, x=X, h=H, m_last=M, ids=np.array([r["id"] for r in rows]))
        log(f"tails {name} {X.shape}")

# other positions
rng = random.Random(5)
sub = []
for name, n in [("trfit_h", 300), ("trfit_b", 300), ("harmful_fit", 100), ("benign_fit", 100), ("xstest_safe_fit", 60), ("orbench_hard_fit", 60)]:
    sub += rng.sample([r for r in allrows if r["set"] == name], n)
fit_lab = run_cell(model, tok, "base_posfit", sub)
dev_sub = []
for name in ["wj_adversarial_harmful", "wj_adversarial_benign", "dev_ho_h", "dev_ho_b", "harmful", "harmful_jb", "xstest_safe", "orbench_hard", "benign"]:
    dev_sub += rng.sample([r for r in allrows if r["set"] == name], 80)
resp = {rk(r): r["response"] for f in ("base_posfit", "base_dev") for r in read_jsonl(RAW / "gen" / f"{f}.jsonl")}
K = len(tok(render(tok, "x"), add_special_tokens=False)["input_ids"]) - 5 - len(tok("x", add_special_tokens=False)["input_ids"])
log(f"preamble length {K}")
for tag, rows in (("fit", sub), ("dev", dev_sub)):
    xs, meta = [], []
    with torch.no_grad():
        for j, r in enumerate(rows):
            p_ids = tok(render(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
            g_ids = tok(resp[rk(r)], add_special_tokens=False)["input_ids"][:128]
            P = len(p_ids)
            ids = torch.tensor([p_ids + g_ids], device=DEV)
            model.model(input_ids=ids, use_cache=False)
            pos = [(i, 0) for i in rng.sample(range(K), 30)] + [(i, 1) for i in rng.sample(range(K, P - 5), min(30, P - 5 - K))] + [(P - 5 + i, 2 + i) for i in range(5)]
            pos += [(i, 7) for i in rng.sample(range(P, P + len(g_ids)), min(40, len(g_ids)))]
            ii = torch.tensor([i for i, _ in pos], device=DEV)
            xs.append(store["x"][0, ii].float().cpu().numpy().astype(np.float16))
            meta += [(j, k, i) for i, k in pos]
    np.savez(CACHE / f"xpos{LYR}_{tag}.npz", x=np.concatenate(xs), meta=np.array(meta, np.int32), sets=np.array([r["set"] for r in rows]), ids=np.array([r["id"] for r in rows]))
    log(f"positions {tag} {len(meta)}")
log("R4_POS DONE")
