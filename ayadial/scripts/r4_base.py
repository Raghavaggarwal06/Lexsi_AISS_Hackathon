"""Base responses, cached activations (hidden states 8..30 at the last 5 template positions) and guard prompt scores."""
import sys
from r4lib import *
SMOKE = "--smoke" in sys.argv
LAYERS = list(range(8, 31))
DEVSETS = ["wj_adversarial_harmful", "wj_adversarial_benign", "dev_ho_h", "dev_ho_b", "dev_in_h", "dev_in_b", "harmful", "harmful_jb", "xstest_safe", "orbench_hard", "benign", "xstest_unsafe"]
FITSETS = ["trfit_h", "trfit_b", "harmful_fit", "benign_fit", "xstest_safe_fit", "orbench_hard_fit", "xstest_unsafe_fit"]


@torch.no_grad()
def capture(model, tok, rows, batch=32):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]))
    out = np.zeros((len(rows), len(LAYERS), 5, 2048), dtype=np.float16)
    ntok = np.zeros(len(rows), dtype=np.int32)
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        e = enc(tok, [rows[i]["prompt"] for i in idx])
        hs = model.model(**e, output_hidden_states=True, use_cache=False).hidden_states
        a = torch.stack([hs[L][:, -5:, :] for L in LAYERS], 1).float().cpu().numpy()
        out[idx] = a.astype(np.float16)
        ntok[idx] = e["attention_mask"].sum(1).cpu().numpy()
    return out, ntok


def main():
    model, tok = load()
    allrows = sets()
    pick = lambda names: [r for n in names for r in [x for x in allrows if x["set"] == n][: 4 if SMOKE else None]]
    tag = "smoke_" if SMOKE else ""
    dev, fit = pick(DEVSETS), pick(FITSETS)
    lab = run_cell(model, tok, f"{tag}base_dev", dev)
    log(f"base dev done n={len(lab)} refused={np.mean([v[0] for v in lab.values()]):.3f}")
    sync()
    for name in DEVSETS + FITSETS:
        path = CACHE / f"{tag}acts_{name}.npy"
        rows = pick([name])
        if path.exists() or not rows:
            continue
        a, nt = capture(model, tok, rows)
        np.save(path, a)
        (CACHE / f"{tag}ids_{name}.json").write_text(json.dumps({"ids": [r["id"] for r in rows], "ntok": nt.tolist()}))
        log(f"acts {name} {a.shape} max_tokens={nt.max()} truncated={(nt >= MAXLEN).sum()}")
    gp = CACHE / f"{tag}guard.json"
    if not gp.exists():
        rows = dev + fit
        g = guard_prompt([r["prompt"] for r in rows])
        gp.write_text(json.dumps({rk(r): v for r, v in zip(rows, g)}))
        log("guard prompt scores done")
    lab = run_cell(model, tok, f"{tag}base_fit", fit)
    log(f"base fit done n={len(lab)}")
    sync()
    log("R4_BASE DONE")


main()
