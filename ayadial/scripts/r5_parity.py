"""Parity of computed units: the float32 hook against the weights, on next-token distributions. python scripts/r5_parity.py specs5/<spec>.json
spec = {"name": "lax", "n_per_set": 12, "sets": [...], "cells": [{"name": "lax@1", "f": "lax", "t": 1, "theta": 0.4}, ...]}
For each cell: KL(hook || weights) and KL(base || weights) at the first reply token and over up to 16 teacher-forced reply positions.
Also: base against the four dormant units zeroed (what both adapters do at scale 0), and what the units write away from the marker.
Charter bar: mean next-token KL under 1e-3 between hook and weights. Writes results/round5/parity/<name>.json."""
import sys
from r5lib import *
from r4_fold import logits_at, kl

DEFAULT_SETS = ["wj_adversarial_harmful", "wj_adversarial_benign", "dev_ho_h", "dev_ho_b", "harmful", "xstest_safe", "orbench_hard", "benign",
                "itw_h_dev", "itw_b_dev", "mj_zh_dev", "mj_sw_dev", "aya_ar_dev", "pht_harmless_dev", "jbb_GCG_dev", "jbb_PAIR_dev"]


def cmp(ref, var):
    return {"kl_first_token": float(np.mean([kl(a, b) for a, b in zip(ref[0], var[0])])), "kl_first_token_max": float(np.max([kl(a, b) for a, b in zip(ref[0], var[0])])),
            "kl_reply_positions": float(np.mean([kl(a, b) for a, b in zip(ref[1], var[1]) if len(a)])), "argmax_agree_first": float(np.mean([int(a.argmax() == b.argmax()) for a, b in zip(ref[0], var[0])]))}


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    model, tok = load()
    rng = random.Random(11)
    allr = sets()
    resp = {}
    for f in sorted((RAW / "gen").glob("base_*.jsonl")):
        resp.update({rk(r): r["response"] for r in read_jsonl(f)})
    rows = []
    for n in spec.get("sets", DEFAULT_SETS):
        pool = [r for r in allr if r["set"] == n and rk(r) in resp]
        rows += rng.sample(pool, min(spec.get("n_per_set", 12), len(pool)))
    out = {"n_prompts": len(rows), "sets": sorted({r["set"] for r in rows}), "note": "KL(reference || variant), natural log; first reply token, and the mean over up to 16 teacher-forced reply positions of the base reply", "cells": {}}
    b = logits_at(model, tok, rows, resp)
    out["base_vs_itself"] = cmp(b, logits_at(model, tok, rows, resp))
    mlp = model.model.layers[LYR].mlp
    G, U = mlp.gate_proj.weight.data.float().cpu().numpy(), mlp.up_proj.weight.data.float().cpu().numpy()
    zero = [dict(n=n, gate=G[n], up=U[n], down=np.zeros(2048, np.float32)) for n in STRICT_UNITS + LAX_UNITS]
    out["base_vs_four_units_zeroed"] = cmp(b, logits_at(model, tok, rows, resp, Units(model, zero)))
    out["base_vs_strict_units_zeroed"] = cmp(b, logits_at(model, tok, rows, resp, Units(model, zero[:2])))
    log(f"zeroed four: {out['base_vs_four_units_zeroed']}")
    for c in spec["cells"]:
        fw = logits_at(model, tok, rows, resp, context(model, {**c, "mode": "weights"}))
        fh = logits_at(model, tok, rows, resp, context(model, {**c, "mode": "hook"}))
        e = {"spec": {k: v for k, v in c.items() if k != "name"}, "hook_vs_weights": cmp(fh, fw), "base_vs_weights": cmp(b, fw)}
        e["pass_kl_1e-3"] = bool(e["hook_vs_weights"]["kl_first_token"] < 1e-3 and e["hook_vs_weights"]["kl_reply_positions"] < 1e-3)
        out["cells"][c["name"]] = e
        log(f"{c['name']}: hook_vs_weights {e['hook_vs_weights']} | base_vs_weights first {e['base_vs_weights']['kl_first_token']:.4f} reply {e['base_vs_weights']['kl_reply_positions']:.5f} | pass {e['pass_kl_1e-3']}")
    (OUT / "parity").mkdir(exist_ok=True)
    (OUT / "parity" / f"{spec['name']}.json").write_text(json.dumps(out, indent=1))
    sync()
    log("R5_PARITY DONE")


if __name__ == "__main__":
    main()
