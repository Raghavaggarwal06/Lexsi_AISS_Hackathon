"""SafeTune baselines on the full development scoreboard (round 5 ran a 300-prompt screen only).
  python scripts/r6_safetune.py rd@1.44 rd@2.87 rd@5.74 adasteer@1 adasteer@2 adasteer@4        suite wjx (1910 rows, round 4 batching)
  python scripts/r6_safetune.py --small cast@0.25 cast@0.5 cast@1                               suite wjx8 (same rows in batches of 8, own base)
Calibration as in round 5 (256 harmful + 256 harmless prompts from the data v1's read was fit on). AdaSteer is rebuilt for every strength, because its
coefficient keeps the bound it was built with (round 5, J30). Cells are named st_<arm>@<strength> (wjx) or wjx8.st_<arm>@<strength>.
Writes results/round6/safetune/full.json or full_small.json (ids, rates, timings); the budget rule is applied afterwards by r6_report.py."""
import sys
from r6lib import *

small = "--small" in sys.argv
todo = [(c.split("@")[0], float(c.split("@")[1])) for c in sys.argv[1:] if "@" in c]


def calib():
    rng = random.Random(0)
    by = {}
    for r in sets():
        by.setdefault(r["set"], []).append(r)
    return rng.sample(by["trfit_h"], 128) + rng.sample(by["harmful_fit"], 128), rng.sample(by["trfit_b"], 128) + rng.sample(by["benign_fit"] + by["xstest_safe_fit"] + by["orbench_hard_fit"], 128)


def build(arm, s, model, tok, hp, bp):
    from safetune.runner.steer import AdaSteerTrainer, CASTTrainer, RefusalDirectionTrainer
    if arm == "rd":
        w, _ = RefusalDirectionTrainer(model, tok, layers=[22], alpha=1.0).calibrate(harmful=hp, harmless=bp)
        w.strength = float(s)
        return w, w.remove
    if arm == "cast":
        w, _ = CASTTrainer(model, tok, alpha=float(s)).calibrate(harmful=hp, harmless=bp)
        w.alpha = float(s)
        return w, w.remove
    w, _ = AdaSteerTrainer(model, tok, alpha=float(s)).calibrate(harmful=hp, harmless=bp)  # alpha is the bound of both coefficients
    return w, w.remove_hooks


def main():
    import safetune
    model, tok = load()
    harmful, harmless = calib()
    hp, bp = [r["prompt"] for r in harmful], [r["prompt"] for r in harmless]
    suite, prefix = ("wjx8", "wjx8.") if small else ("wjx", "")
    (OUT / "safetune").mkdir(exist_ok=True)
    outp = OUT / "safetune" / ("full_small.json" if small else "full.json")
    out = json.loads(outp.read_text()) if outp.exists() else {"cells": {}, "failures": {}}
    out.update({"safetune_version": getattr(safetune, "__version__", "?"), "suite": suite, "batching": SUITE_ENV[suite]})
    rows = suite_rows(suite)
    base = run(model, tok, suite, base_name(suite), None, rows)
    H, B = [rk(r) for r in rows if r["set"] in WJ_H], [rk(r) for r in rows if r["set"] in WJ_B]
    rate = lambda lab, keys, c: round(float(np.mean([lab[k][c] for k in keys]) * 100), 1)
    out["base"] = {"harmful_refusal": rate(base, H, 0), "harmful_unsafe": rate(base, H, 1), "benign_refusal": rate(base, B, 0), "n": [len(H), len(B)]}
    for arm, s in todo:
        name = f"{prefix}st_{arm}@{s:g}"
        t0 = time.time()
        try:
            w, cleanup = build(arm, s, model, tok, hp, bp)
            lab = run(w, tok, suite, name, None, rows)
            out["cells"][f"st_{arm}@{s:g}"] = {"harmful_refusal": rate(lab, H, 0), "harmful_unsafe": rate(lab, H, 1), "benign_refusal": rate(lab, B, 0), "degenerate": rate(lab, H + B, 2), "seconds": round(time.time() - t0)}
            log(f"{name} {out['cells'][f'st_{arm}@{s:g}']}")
        except Exception as err:
            out["failures"][name] = f"{type(err).__name__}: {str(err)[:300]}"
            log(f"{name}: failed: {out['failures'][name]}")
        try:
            cleanup()
        except Exception as err:
            log(f"{name}: cleanup: {type(err).__name__} {err}")
        torch.cuda.empty_cache()
        outp.write_text(json.dumps(out, indent=1))
    sync()
    log("R6_SAFETUNE DONE")


if __name__ == "__main__":
    main()
