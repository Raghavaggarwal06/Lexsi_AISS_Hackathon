"""SafeTune baselines (https://github.com/Lexsi-Labs/SafeTune) on the development scoreboard: RefusalDirection, CAST, AdaSteer.
Each arm is calibrated on fit data only, then generated through SafeTune's own wrapper on a suite, judged like every other cell,
and scored by the same budget rule against the ungated control.
  python scripts/r5_safetune.py screen     calibrate, then every arm and strength on a 300-prompt screen of the wjx rows
  python scripts/r5_safetune.py screen8 [arm ...]   the same screen rows in batches of 8 with their own base (for arms that run out of memory in large batches)
  python scripts/r5_safetune.py full <arm>@<strength> [...]     chosen cells on all 1910 wjx rows (cells named st_<arm>@<strength>)
Calibration data: 256 harmful and 256 harmless prompts drawn from the data v1's read was fit on (half WildJailbreak train rows, half our fit halves),
so the baselines are not handicapped by seeing less than v1. Writes results/round5/safetune/<mode>.json (ids, rates, timings; no text)."""
import sys
from r5lib import *

ARMS = {"rd": [1.44, 2.87, 5.74],  # RefusalDirectionTrainer, layer 22; strengths = 0.05, 0.1, 0.2 times ||r|| (28.72), the ungated control's grid
        "cast": [0.5, 1.0, 2.0],   # CASTTrainer: alpha
        "adasteer": [4.0, 8.0, 15.0]}  # AdaSteerTrainer: rd_max = hd_max


def calib():
    rng = random.Random(0)
    by = {}
    for r in sets():
        by.setdefault(r["set"], []).append(r)
    harmful = rng.sample(by["trfit_h"], 128) + rng.sample(by["harmful_fit"], 128)
    harmless = rng.sample(by["trfit_b"], 128) + rng.sample(by["benign_fit"] + by["xstest_safe_fit"] + by["orbench_hard_fit"], 128)
    return harmful, harmless


def build(arm, model, tok, harmful, harmless):
    from safetune.runner.steer import AdaSteerTrainer, CASTTrainer, RefusalDirectionTrainer
    hp, bp = [r["prompt"] for r in harmful], [r["prompt"] for r in harmless]
    if arm == "rd":
        w, _ = RefusalDirectionTrainer(model, tok, layers=[22], alpha=1.0).calibrate(harmful=hp, harmless=bp)
        return w, lambda a: setattr(w, "strength", float(a)), w.remove
    if arm == "cast":
        w, _ = CASTTrainer(model, tok, alpha=1.0).calibrate(harmful=hp, harmless=bp)
        return w, lambda a: setattr(w, "alpha", float(a)), w.remove
    w, _ = AdaSteerTrainer(model, tok, alpha=1.0).calibrate(harmful=hp, harmless=bp)
    def set_(a):
        w.config.rd_max = float(a); w.config.hd_max = float(a)
    return w, set_, w.remove_hooks


def main():
    mode = sys.argv[1]
    import safetune
    model, tok = load()
    harmful, harmless = calib()
    (OUT / "safetune").mkdir(exist_ok=True)
    outp = OUT / "safetune" / f"{mode}.json"
    out = json.loads(outp.read_text()) if outp.exists() else {}
    out.update({"safetune_version": getattr(safetune, "__version__", "?"), "calibration": {"harmful_ids": [rk(r) for r in harmful], "harmless_ids": [rk(r) for r in harmless]}})
    out.setdefault("cells", {}); out.setdefault("failures", {})
    if mode == "screen":
        (RAW / "suites").mkdir(exist_ok=True)
        sp = RAW / "suites" / "stscreen.jsonl"
        if not sp.exists():
            rng = random.Random(7)
            allr = sets()
            write_jsonl(sp, [r for n in WJ_H + WJ_B for r in rng.sample([x for x in allr if x["set"] == n], 50)])
        suite, prefix, todo = "stscreen", "stscreen.", [(a, s) for a, ss in ARMS.items() for s in ss]
    elif mode == "screen8":  # the same 300 screen rows in batches of 8, with their own base: python scripts/r5_safetune.py screen8 cast
        # CAST's wrapper asked for 37.6 GiB in one allocation at the default batch (up to 160 prompts), so it gets a small batch and a base generated the same way.
        sp8 = RAW / "suites" / "stscreen8.jsonl"
        if not sp8.exists():
            write_jsonl(sp8, read_jsonl(RAW / "suites" / "stscreen.jsonl"))
        SUITE_ENV["stscreen8"] = (8000, 8)
        suite, prefix, todo = "stscreen8", "stscreen8.", [(a, s) for a in (sys.argv[2:] or ["cast"]) for s in ARMS[a]]
        out["batching"] = "8000 tokens, 8 prompts per batch (the screen itself: 90000, 160)"
    else:
        suite, prefix, todo = "wjx", "", [(c.split("@")[0], float(c.split("@")[1])) for c in sys.argv[2:]]
    rows = suite_rows(suite)
    base = run(model, tok, suite, base_name(suite), None, rows)
    H, B = [rk(r) for r in rows if r["set"] in WJ_H], [rk(r) for r in rows if r["set"] in WJ_B]
    rate = lambda lab, keys, c: round(float(np.mean([lab[k][c] for k in keys]) * 100), 1)
    out["base"] = {"harmful_refusal": rate(base, H, 0), "harmful_unsafe": rate(base, H, 1), "benign_refusal": rate(base, B, 0), "n": [len(H), len(B)]}
    if mode == "screen":  # the ungated control on the same screen rows, for reference
        for s in (0.05, 0.1, 0.2):
            lab = run(model, tok, suite, f"{prefix}ungated@{s:g}", context(model, {"f": "ungated", "scale": s}), rows)
            out["cells"][f"ungated@{s:g}"] = {"harmful_refusal": rate(lab, H, 0), "harmful_unsafe": rate(lab, H, 1), "benign_refusal": rate(lab, B, 0), "degenerate": rate(lab, H + B, 2)}
            log(f"ungated@{s:g} {out['cells'][f'ungated@{s:g}']}")
    for arm in dict.fromkeys(a for a, _ in todo):
        t0 = time.time()
        try:
            w, set_strength, cleanup = build(arm, model, tok, harmful, harmless)
        except Exception as err:
            out["failures"][arm] = f"calibrate: {type(err).__name__}: {str(err)[:300]}"
            log(f"{arm}: calibration failed: {out['failures'][arm]}")
            outp.write_text(json.dumps(out, indent=1))
            continue
        log(f"{arm}: calibrated in {time.time() - t0:.0f}s")
        for a, s in todo:
            if a != arm:
                continue
            name = f"{prefix}st_{arm}@{s:g}"
            try:
                t1 = time.time()
                set_strength(s)
                lab = run(w, tok, suite, name, None, rows)  # generation goes through SafeTune's wrapper
                out["cells"][f"st_{arm}@{s:g}"] = {"harmful_refusal": rate(lab, H, 0), "harmful_unsafe": rate(lab, H, 1), "benign_refusal": rate(lab, B, 0), "degenerate": rate(lab, H + B, 2), "seconds": round(time.time() - t1)}
                log(f"{name} {out['cells'][f'st_{arm}@{s:g}']}")
            except Exception as err:
                out["failures"][name] = f"generate: {type(err).__name__}: {str(err)[:300]}"
                log(f"{name}: failed: {out['failures'][name]}")
            outp.write_text(json.dumps(out, indent=1))
        try:
            cleanup()
        except Exception as err:
            log(f"{arm}: cleanup: {type(err).__name__} {err}")
    outp.write_text(json.dumps(out, indent=1))
    sync()
    log("R5_SAFETUNE DONE")


if __name__ == "__main__":
    main()
