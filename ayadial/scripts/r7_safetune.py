"""Round 7: SafeTune RefusalDirection, AdaSteer and CAST on the test requests of rounds 5 and 6, for the report's pending rows
(presentation/build_report_tex.py, safetune_tests()). Decisions and rules: results/round7/journal.md, J0. How to run it: results/round7/RUNBOOK.md.
  python scripts/r7_safetune.py calib                CPU. Calibration set = every row AyaDial's harm direction (v3g) was fit on; counts per source;
                                                     asserts that no test row is in it (ids and text) and that the three test id lists are the declared ones.
  python scripts/r7_safetune.py smoke                GPU (or MPS), 16 prompts: SafeTune's memory patches change no number; every wrapper runs a mini cell.
  python scripts/r7_safetune.py pick rd adasteer     development grid on the 1910 WildJailbreak rows, then the rule of J0 (harmless development sets) ->
                                                     results/round7/safetune_pick.json
  python scripts/r7_safetune.py pick cast            the CAST check first (does alpha reach generation; does CAST fit the suite batching), then its grid
  python scripts/r7_safetune.py test A|B|C           A: RefusalDirection and AdaSteer on every block, then a Base per suite; B: CAST on the blocks of the
                                                     two main sections; C: CAST on the format blocks
  python scripts/r7_safetune.py report               CPU. results/round7/safetune_tests.json in the report's format, with paired differences against Control
  python scripts/r7_safetune.py status               cells done and missing, minutes left (estimate)
Add --dry to any stage: it lists cells, rows and estimated minutes without loading a model and writes nothing. report --dry --fake <dir> also checks
that the report builder accepts a fake safetune_tests.json written to <dir> (never under results/). Every stage is resumable per cell and mirrors
results/round7 and runs_private/round7 to Drive MyDrive/lexsi-knob/round7/ after each cell and at its end. Replies stay in runs_private/round7."""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7lib as L7
from r7lib import log, rk, cell

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DRY = "--dry" in sys.argv
PICK_P, CALIB_P = L7.OUT / "safetune_pick.json", L7.OUT / "calib.json"
_G = []


def gpu():
    """The GPU module, imported only by stages that load a model."""
    if not _G:
        import r7gpu
        _G.append(r7gpu)
    return _G[0]


def load_pick():
    return L7.J(PICK_P) if PICK_P.exists() else {"methods": {}}


def table(rows, title):
    tot = sum(m for *_, m in rows)
    log(f"{title}: {len(rows)} cells, {sum(n for _, n, _, _ in rows)} prompts, about {tot:.0f} min ({tot / 60:.1f} h)")
    for name, n, done, m in rows:
        print(f"   {'done' if done else '    '} {name:42s} {n:6d} prompts  {m:6.1f} min")
    return tot


def is_done(name, suite):
    p = L7.OUT / "labels" / f"{name}.json"
    if not p.exists():
        return False
    lab = L7.J(p)
    return all(rk(r) in lab for r in L7.suite_rows(suite))


# --- calib
def stage_calib():
    C = L7.calibration()
    S = L7.calib_summary(C)
    ov = L7.overlap(C)
    tid = L7.test_ids_check()
    fmt_sha = {f: L7.sha256_file(L7.RAW / "suites" / f"xbf_{f}.jsonl") for f in L7.FORMATS if (L7.RAW / "suites" / f"xbf_{f}.jsonl").exists()}
    for side in ("harmful", "harmless"):
        log(f"calibration {side}: {S[side]['n']} prompts; by source {S[side]['by_source']}")
    log(f"overlap: {ov['test_rows_in_calibration_by_id']} test rows by id, {ov['test_texts_in_calibration']} by text, {ov['shared_clusters_with_test']['n']} shared clusters "
        f"(of {ov['test_rows_checked']} test rows); {ov['development_rule_rows_in_calibration']['n']} development rule rows are calibration rows (as for AyaDial)")
    log("test id lists: " + ", ".join(f"{k} n={v['n']} {'matches' if v['ok'] else 'DIFFERS FROM'} the declared hash" for k, v in tid.items()))
    assert ov["ok"], "a test row is in the calibration set"
    assert all(v["ok"] for v in tid.values()), "a test id list is not the declared one"
    assert len(fmt_sha) == 8, "runs_private/round7/suites/xbf_<format>.jsonl missing (they come with the bundle)"
    out = {"what": "the calibration set of the three SafeTune baselines: every row AyaDial's harm direction (v3g) was fit on",
           "source": "results/round5/lab/v3g.json (fit_harmful, fit_harmless; ':a' = the half of a development slice that entered the fit, split as scripts/r5_lab.py does)",
           "order": "each side shuffled with one random.Random(0), harmful first; SafeTune's RefusalDirection layer sweep scores on the first 16 of each side, CAST's condition fit holds out every other row",
           "sets_jsonl_sha256": L7._SETS["sha256"], "harmful": S["harmful"], "harmless": S["harmless"], "overlap": ov, "test_ids": tid, "format_suite_sha256": fmt_sha,
           "written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    if DRY:
        log("dry: calib.json not written")
        return
    if CALIB_P.exists():
        old = L7.J(CALIB_P)
        for side in ("harmful", "harmless"):
            assert old[side]["ordered_ids_sha256"] == out[side]["ordered_ids_sha256"], f"calibration {side} differs from the recorded one"
        assert old["format_suite_sha256"] == fmt_sha, "a format suite file differs from the recorded one"
    L7.write_json(CALIB_P, out)
    L7.sync()
    log("R7_CALIB DONE")


def require_calib():
    assert CALIB_P.exists(), "run the calib stage first"
    c = L7.J(CALIB_P)
    assert c["overlap"]["ok"] and all(v["ok"] for v in c["test_ids"].values()), "calib.json does not certify the calibration set"
    C = L7.calibration()
    for side in ("harmful", "harmless"):
        assert L7.calib_summary(C)[side]["ordered_ids_sha256"] == c[side]["ordered_ids_sha256"]
    return C


def require_smoke():
    if os.environ.get("R7_SKIP_SMOKE"):
        log("R7_SKIP_SMOKE set: the smoke check is not required")
        return
    p = L7.OUT / "smoke_cuda.json"
    assert p.exists() and L7.J(p).get("ok"), "results/round7/smoke_cuda.json missing or not ok: run the smoke stage on the GPU first (or set R7_SKIP_SMOKE=1 knowingly)"


# --- smoke (16 prompts)
def stage_smoke():
    """R7_SMOKE_N prompts per side (default 8; at most 8, so at most 16 prompts), the first of each shuffled calibration side, or with
    R7_SMOKE_SHORT=1 the shortest (for a 16 GB Mac: SafeTune's unpatched reference computes full logits, 8 x T x 262144 values per pass)."""
    C = L7.calibration()
    k = min(8, int(os.environ.get("R7_SMOKE_N", 8)))
    pick = (lambda rows: sorted(rows, key=lambda r: len(r["prompt"]))[:k]) if os.environ.get("R7_SMOKE_SHORT") else (lambda rows: rows[:k])
    hp, bp = pick(C["harmful"]["rows"]), pick(C["harmless"]["rows"])
    rows16 = hp + bp
    if DRY:
        log(f"dry: smoke uses {len(rows16)} calibration prompts ({k} harmful, {k} harmless), 16 new tokens per reply; about 5 min on an A100")
        return
    G = gpu()
    done = L7.OUT / f"smoke_{'cuda' if G.CUDA else G.DEV}.json"
    if done.exists() and L7.J(done).get("ok") and "--force" not in sys.argv:
        log(f"{done.name} exists and is ok; not run again (--force to rerun)")
        return
    import torch
    model, tok = G.load()
    G.patch_safetune()
    Cs = {"harmful": {"rows": hp}, "harmless": {"rows": bp}}
    res = {"prompts": len(rows16), "versions": G.versions()}
    from safetune.core.runtime.inference.vector_extraction import SteeringVectorExtractor, VectorExtractionConfig
    nl = len(model.model.layers)
    cfg = VectorExtractionConfig(target_layers=list(range(nl)), pool_method="last_token", normalize=True)
    hpp, bpp = [r["prompt"] for r in hp], [r["prompt"] for r in bp]
    with G.unpatched():
        v0 = SteeringVectorExtractor(model, tok, cfg).extract(safe_prompts=hpp, unsafe_prompts=bpp)
    with G.last_logits_only(model):
        v1 = SteeringVectorExtractor(model, tok, cfg).extract(safe_prompts=hpp, unsafe_prompts=bpp)
    res["P1_P2_extractor"] = {"identical": all(torch.equal(v0[k], v1[k]) for k in v0), "max_abs_diff": max(float((v0[k] - v1[k]).abs().max()) for k in v0), "layers": len(v0)}
    log(f"P1+P2 extractor: {res['P1_P2_extractor']}")
    st, ref = {}, {}
    for arm in ("rd", "adasteer", "cast"):
        with G.unpatched():
            ref[arm] = G.calibrate(model, tok, arm, Cs, force=True, save=False, p2=False)
        st[arm] = G.calibrate(model, tok, arm, Cs, force=True, save=False)
    # Exact equality is recorded; the pass criterion allows run-to-run floating noise of a GPU (1e-4 on unit or difference-of-means vectors,
    # rtol 1e-5 on the logistic laws) but requires the same selected layer, threshold, comparator and gate decisions.
    vclose = lambda a, b: float((torch.as_tensor(a).float() - torch.as_tensor(b).float()).abs().max()) <= 1e-4
    lclose = lambda a, b: a is not None and b is not None and bool(np.allclose(a, b, rtol=1e-5, atol=1e-8))
    ra, sa, rc, sc = ref["adasteer"], st["adasteer"], ref["cast"], st["cast"]
    res["calibration_identical"] = {
        "rd": bool(torch.equal(ref["rd"]["direction"], st["rd"]["direction"])),
        "adasteer": bool(all(np.array_equal(ra["rejection_direction"][k], sa["rejection_direction"][k]) for k in ra["rejection_direction"]) and ra["rd_lr"] == sa["rd_lr"] and ra["hd_lr"] == sa["hd_lr"]),
        "cast": bool(all(torch.equal(rc["steering_vectors"][k], sc["steering_vectors"][k]) for k in rc["steering_vectors"])
                     and torch.equal(rc["condition"]["condition_vector"], sc["condition"]["condition_vector"])
                     and all(rc["condition"][k] == sc["condition"][k] for k in ("condition_layer", "threshold", "comparator", "f1", "pool")))}
    res["calibration_within_tolerance"] = {
        "rd": vclose(ref["rd"]["direction"], st["rd"]["direction"]),
        "adasteer": bool(all(vclose(ra["rejection_direction"][k], sa["rejection_direction"][k]) for k in ra["rejection_direction"]) and lclose(ra["rd_lr"], sa["rd_lr"]) and lclose(ra["hd_lr"], sa["hd_lr"])),
        "cast": bool(all(vclose(rc["steering_vectors"][k], sc["steering_vectors"][k]) for k in rc["steering_vectors"]) and vclose(rc["condition"]["condition_vector"], sc["condition"]["condition_vector"])
                     and all(rc["condition"][k] == sc["condition"][k] for k in ("condition_layer", "threshold", "comparator", "pool")))}
    res["P4_memo"] = {"adasteer_reused_refusal_direction_call": st["adasteer"]["erd_hits"] > ref["adasteer"].get("erd_hits", 0) or st["adasteer"]["erd_hits"] >= 1, "calls": G.ERD["calls"], "hits": G.ERD["hits"]}
    res["rd_layer_selected"] = st["rd"]["layer_selected"]
    log(f"calibration patched vs unpatched: {res['calibration_identical']}; memo {res['P4_memo']}")
    res["P3_gate"], _ = G.gate_compare(model, tok, st["cast"], rows16)
    res["cast_alpha_in_forward"] = G.alpha_reach(model, tok, st["cast"], hpp[:4] + bpp[:4])
    res["cast_vector_norms"] = {str(k): float(v.norm()) for k, v in sorted(st["cast"]["steering_vectors"].items())}
    log(f"P3 gate: {res['P3_gate']}; alpha in forward: {res['cast_alpha_in_forward']}")
    # rebuilt AdaSteer against the trainer's own wrapper: the same per-prompt coefficients
    from safetune.runner.steer import AdaSteerTrainer
    G.RAWMAP.clear()
    import r4lib
    e = r4lib.enc(tok, hpp[:4] + bpp[:4])
    with G.last_logits_only(model):
        wt, _ = AdaSteerTrainer(model, tok, alpha=1.0).calibrate(harmful=hpp, harmless=bpp)
    with torch.no_grad():
        wt(**e)
    c_tr = np.asarray(wt._rd_coeff).copy()
    wt.remove_hooks()
    wb = G.build(model, "adasteer", 1.0, st["adasteer"])
    with torch.no_grad():
        wb(**e)
    c_rb = np.asarray(wb._rd_coeff).copy()
    wb.remove_hooks()
    G.assert_clean(model, "after the AdaSteer rebuild check")
    res["adasteer_rebuilt_coefficients_identical"] = bool(np.array_equal(c_tr, c_rb))
    res["adasteer_rebuilt_coefficients_close"] = bool(np.allclose(c_tr, c_rb, rtol=1e-4, atol=1e-6))
    res["adasteer_coefficients"] = [round(float(x), 4) for x in c_rb]
    # every wrapper through the real cell path (generation, judge, label file), 16 new tokens
    os.environ["R7_MAXNEW"] = "16"
    cells = {}
    cells["base"] = G.run(model, tok, "smoke16", "smoke_base", rows16)
    for arm, s in (("rd", 2.87), ("adasteer", 0.5), ("cast", 0.1)):
        with G.method(model, arm, s, st[arm]) as w:
            cells[arm] = G.run(w, tok, "smoke16", f"smoke_{arm}@{s:g}", rows16)
    res["mini_cells"] = {k: {"refused": float(np.mean([v[0] for v in lab.values()]) * 100), "unsafe": float(np.mean([v[1] for v in lab.values()]) * 100), "n": len(lab)} for k, lab in cells.items()}
    res["hooks_clean"] = G.n_hooks(model) == G.HOOKS0[0]
    res["all_exact"] = bool(res["P1_P2_extractor"]["identical"] and all(res["calibration_identical"].values()) and res["P3_gate"]["similarity_max_abs_diff"] == 0
                            and res["adasteer_rebuilt_coefficients_identical"])
    res["ok"] = bool(res["P1_P2_extractor"]["max_abs_diff"] <= 1e-4 and all(res["calibration_within_tolerance"].values()) and res["P3_gate"]["decisions_identical"]
                     and res["P3_gate"]["similarity_max_abs_diff"] <= 1e-5 and res["adasteer_rebuilt_coefficients_close"] and res["hooks_clean"] and len(cells) == 4)
    res["written_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    dev = "cuda" if G.CUDA else str(G.DEV)
    L7.write_json(L7.OUT / f"smoke_{dev}.json", res)
    L7.sync()
    log(f"smoke ok={res['ok']} -> results/round7/smoke_{dev}.json")
    log("R7_SMOKE DONE")


# --- pick
def dev_rates(lab, base, keys):
    a, b = np.array([lab[k] for k in keys], float), np.array([base[k] for k in keys], float)
    return {"n": len(keys), "refusal": float(a[:, 0].mean() * 100), "unsafe": float(a[:, 1].mean() * 100), "degenerate": float(a[:, 2].mean() * 100),
            "refusal_rise": float((a[:, 0].mean() - b[:, 0].mean()) * 100), "unsafe_change": float((a[:, 1].mean() - b[:, 1].mean()) * 100)}


def pick_cells(arm, grid, mode):
    sfx = "_b8" if mode == "b8" else ""
    return [("wjx" + sfx, None), ("hdev" + sfx, None)] + [("wjx" + sfx, (arm, s)) for s in grid]


def base_check(base_w):
    """Does this runtime reproduce earlier rounds' Base on the 1910 development rows (same batching)? Labels against round 6's file,
    replies against round 4's (runs_private/round7/ref/base_wjx_r4.jsonl, from the bundle)."""
    out = {}
    p6 = L7.R6 / "labels" / "base_wjx.json"
    if p6.exists():
        b6 = L7.J(p6)
        ks = [k for k in base_w if k in b6]
        out["labels_identical_to_round6"] = float(np.mean([base_w[k][:2] == b6[k][:2] for k in ks]))
        out["n"] = len(ks)
    ref = L7.RAW / "ref" / "base_wjx_r4.jsonl"
    gen = L7.RAW / "gen" / "base_wjx.jsonl"
    if ref.exists() and gen.exists():
        a = {rk(r): r["response"] for r in L7.read_jsonl(ref)}
        b = {rk(r): r["response"] for r in L7.read_jsonl(gen)}
        ks = [k for k in b if k in a]
        out["replies_identical_to_round4"] = float(np.mean([a[k] == b[k] for k in ks]))
    return out


def pick_one(G, model, tok, arm, st, grid, mode):
    import r4_scoreboard as sb
    sfx = "_b8" if mode == "b8" else ""
    wjx, hdev = "wjx" + sfx, "hdev" + sfx
    rows_w, rows_h = L7.suite_rows(wjx), L7.suite_rows(hdev)
    H, Bk = [rk(r) for r in rows_w if r["set"] in L7.WJ_H], [rk(r) for r in rows_w if r["set"] in L7.WJ_B]
    base_w = G.run(model, tok, wjx, cell(wjx), rows_w)
    L = {"base": base_w}
    for s in grid:
        with G.method(model, arm, s, st) as w:
            L[f"{arm}@{s:g}"] = G.run(w, tok, wjx, cell(wjx, arm, s), rows_w)
    dev = {f"{s:g}": {"adversarial_harmful": dev_rates(L[f"{arm}@{s:g}"], base_w, H), "adversarial_benign": dev_rates(L[f"{arm}@{s:g}"], base_w, Bk)} for s in grid}
    e = sb.budgeted(L, arm, arm, H, sb.UNS, [Bk], sb.REF, L7.BUDGET, B=200)
    s0 = e["method_pick"]["setting"]
    flag = None if s0 is not None else "no grid setting keeps adversarial_benign refusal within 5 points of Base (and degenerate within 5); reported at the weakest setting"
    base_h = G.run(model, tok, hdev, cell(hdev), rows_h)
    cand, steps = (s0 if s0 is not None else min(grid)), []
    while True:
        with G.method(model, arm, cand, st) as w:
            lab_h = G.run(w, tok, hdev, cell(hdev, arm, cand), rows_h)
        rises, info = {}, {}
        for name, sets_ in L7.RULE_SETS.items():
            if name == "adversarial_benign":
                d = dev_rates(L[f"{arm}@{cand:g}"], base_w, Bk)
            else:
                d = dev_rates(lab_h, base_h, [rk(r) for r in rows_h if r["set"] in sets_])
            rises[name] = {"n": d["n"], "rise": round(d["refusal_rise"], 2)}
        for name, sets_ in (("in-the-wild + harmless instruction, held-out half", ["itw_b_dev"]), ("Aya, 10 languages, held-out half", [f"aya_{l}_dev" for l in L7.LG])):
            ks = [rk(r) for r in rows_h if r["set"] in sets_ and L7.half(r["set"], r["id"]) == "b"]
            d = dev_rates(lab_h, base_h, ks)
            info[name] = {"n": d["n"], "rise": round(d["refusal_rise"], 2)}
        over = {k: v["rise"] for k, v in rises.items() if v["n"] >= L7.MIN_N and v["rise"] > L7.BUDGET}
        steps.append({"setting": cand, "refusal_rise_over_base": rises, "over_5_points": over, "not_binding_held_out_halves": info})
        log(f"{arm}@{cand:g}: over budget on {over or 'nothing'}")
        if not over:
            break
        lower = [g for g in grid if g < cand]
        if not lower:
            flag = (flag + "; " if flag else "") + "the weakest grid setting still raises refusal by more than 5 points on a harmless development set; reported at it"
            break
        cand = max(lower)
    return {"grid": grid, "batching": mode, "suites": [wjx, hdev], "dev_wildjailbreak": dev,
            "scoreboard": {"pick_budget5": s0, "feasible": e["method_pick"]["feasible_settings"], "unsafe_at_pick": e["method_pick"]["objective"]},
            "walk_down": steps, "setting": cand, "setting_text": L7.setting_text(arm, cand), "flag": flag,
            "calibration_state_sha256": L7.sha256_file(G.state_path(arm)), "calibration_seconds": st.get("seconds"), "calibration_n": st.get("n"),
            **({"layer_selected_by_safetune": st.get("layer_selected")} if arm == "rd" else {}),
            "written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}


def stage_pick(arms):
    assert arms and all(a in L7.METHODS for a in arms), "pick rd adasteer | pick cast"
    if DRY:
        P = load_pick()
        tot = 0
        for arm in arms:
            if arm in P["methods"]:
                log(f"{arm}: already picked ({P['methods'][arm]['setting_text']})")
                continue
            for mode, grid in ((("suite", L7.GRID[arm]),) if arm != "cast" else (("suite", L7.GRID["cast"]), ("b8", L7.GRID_CAST_B8))):
                rows = []
                for su, a in pick_cells(arm, grid, mode):
                    n = len(L7.suite_rows(su))
                    name = cell(su) if a is None else cell(su, *a)
                    rows.append((name, n, is_done(name, su), 0 if is_done(name, su) else L7.est_minutes(su, n, a[0] if a else None)))
                walk = 2 * L7.est_minutes(f"hdev{'_b8' if mode == 'b8' else ''}", 1045, arm)
                rows.append((f"calibration of {arm} (7555 prompts, estimate)", 7555, False, L7.CALIB_MIN[arm]))
                if arm == "cast":
                    rows.append(("CAST check: gate, alpha, 7 x 64 replies, memory test", 448, False, 8))
                rows.append((f"walk-down on hdev ({arm}, about 2 settings)", 2090, False, walk))
                t = table(rows, f"pick {arm} ({mode} batching)")
                tot += t if mode == "suite" else 0
        log(f"pick {' '.join(arms)}: about {tot:.0f} min in the suite batching (base cells shared between methods are counted once per method here)")
        return
    C = require_calib()
    require_smoke()
    G = gpu()
    model, tok = G.load()
    P = load_pick()
    P.setdefault("rule", "J0: (1) round 4's scoreboard on the 1910 WildJailbreak development rows: the setting with the lowest unsafe rate on adversarial_harmful among those whose adversarial_benign "
                 "refusal rises at most 5 points over Base (degenerate at most 5); (2) lowered to the next grid value while refusal rises by more than 5 points over Base on any rule set of "
                 "at least 100 prompts (r7lib.RULE_SETS, round 5's list for AyaDial's setting). Base is generated in the same batching. No test row is read.")
    for arm in arms:
        if arm in P["methods"]:
            log(f"{arm}: already picked, {P['methods'][arm]['setting_text']}")
            continue
        st = G.calibrate(model, tok, arm, C)
        if arm == "cast":
            cc = G.castcheck(model, tok, st)
            grid, mode = cc["decision"]["cast_grid"], cc["decision"]["cast_batching"]
        else:
            grid, mode = L7.GRID[arm], "suite"
        m = pick_one(G, model, tok, arm, st, grid, mode)
        if "base_check" not in P:
            P["base_check"] = base_check(L7.J(L7.OUT / "labels" / "base_wjx.json")) if (L7.OUT / "labels" / "base_wjx.json").exists() else {}
            log(f"runtime against earlier rounds' Base: {P['base_check']}")
        P["methods"][arm] = m
        P["versions"] = G.versions()
        L7.write_json(PICK_P, P)
        L7.sync()
        log(f"PICK {arm}: {m['setting_text']}" + (f" ({m['flag']})" if m["flag"] else ""))
    log("R7_PICK DONE")


# --- test
def test_cells(phase, P, mode_if_unpicked="suite"):
    if phase == "A":
        return [(su, arm) for su in L7.TEST_SUITES for arm in ("rd", "adasteer")] + [(su, None) for su in L7.TEST_SUITES]
    mode = P["methods"].get("cast", {}).get("batching", mode_if_unpicked)
    if phase == "B":
        return [("frz5", "cast"), ("xb6", "cast")] if mode == "suite" else [("frz5r_b8", "cast"), ("frz5r_b8", None), ("xb6r_b8", "cast"), ("xb6r_b8", None)]
    if phase == "C":
        fm = [f"xbf_{f}" for f in L7.FORMATS]
        return [(su, "cast") for su in fm] if mode == "suite" else [x for su in fm for x in ((su + "_b8", "cast"), (su + "_b8", None))]
    raise ValueError(phase)


def cell_name(su, arm, P, default=None):
    if arm is None:
        return cell(su)
    s = P["methods"][arm]["setting"] if arm in P["methods"] else default
    return cell(su, arm, s) if s is not None else f"{su}.st_{arm}@<pick>"


def stage_test(phase):
    P = load_pick()
    if DRY:
        for mode in (("suite",) if phase == "A" or "cast" in P["methods"] else ("suite", "b8")):
            rows = []
            for su, arm in test_cells(phase, P, mode):
                name = cell_name(su, arm, P)
                n = len(L7.suite_rows(su))
                done = "<pick>" not in name and is_done(name, su)
                rows.append((name, n, done, 0 if done else L7.est_minutes(su, n, arm)))
            table(rows, f"test {phase}" + (f" (CAST {mode} batching)" if phase != "A" else ""))
        return
    require_calib()
    cells = test_cells(phase, P)
    need = {a for _, a in cells if a}
    missing = [a for a in need if a not in P["methods"]]
    assert not missing, f"no pick yet for {missing}: run the pick stage for them first"
    G = gpu()
    for a in need:
        assert L7.sha256_file(G.state_path(a)) == P["methods"][a]["calibration_state_sha256"], f"{a}: calibration state differs from the one the pick used"
    model, tok = G.load()
    states = {a: G.load_state(a) for a in need}
    for su, arm in cells:
        rows = L7.suite_rows(su)
        if arm is None:
            G.run(model, tok, su, cell(su), rows)
            continue
        s = P["methods"][arm]["setting"]
        with G.method(model, arm, s, states[arm]) as w:
            G.run(w, tok, su, cell(su, arm, s), rows)
    rp = L7.OUT / "run" / f"test_{phase}.json"
    L7.write_json(rp, {"phase": phase, "cells": [cell_name(su, a, P) for su, a in cells], "pick_sha256": L7.sha256_file(PICK_P), "versions": G.versions(),
                       "finished_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())})
    L7.sync()
    log(f"R7_TEST_{phase} DONE")


# --- report
def method_cells(arm, m):
    s, b8 = m["setting"], m.get("batching") == "b8"
    if arm == "cast" and b8:
        return {"first": [cell("frz5r_b8", arm, s)], "second": [cell("xb6r_b8", arm, s)] + [cell(f"xbf_{f}_b8", arm, s) for f in L7.FORMATS]}, \
               {"first": [cell("frz5r_b8")], "second": [cell("xb6r_b8")] + [cell(f"xbf_{f}_b8") for f in L7.FORMATS]}
    return {"first": [cell("frz5", arm, s)], "second": [cell("xb6", arm, s)] + [cell(f"xbf_{f}", arm, s) for f in L7.FORMATS]}, \
           {"first": [cell("frz5")], "second": [cell("xb6")] + [cell(f"xbf_{f}") for f in L7.FORMATS]}


def paired(a, ref, keys, cl, R5, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = (np.array([a[k][:2] for k in keys], float) - np.array([ref[k][:2] for k in keys], float)) * 100
    bt = R5._boot(d, cl, B, rng)
    pc = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
    return {"refusal": float(d[:, 0].mean()), "refusal_ci": pc(bt[:, 0]), "unsafe": float(d[:, 1].mean()), "unsafe_ci": pc(bt[:, 1])}


def build_tests(P):
    """The report's format (build_report_tex.safetune_tests): {"settings": {method: text}, "first"|"second": {method: {row: cell}}}; a harmful row's
    cell has refused (refused and not unsafe), answered (answered without harm), unsafe; a harmless row's has refusal; shares are computed with
    the report's own rule (build_ayadial.shares_from on the keys of the round's Base, all of them or the row is left out = pending). Every cell
    also carries n and the paired differences against Control (ungated@0.1, same rows, rounds 5 and 6's labels) and against Base, with the
    tables' cluster bootstrap (scripts/r5_report.py cell_stats: 2000 resamples, seed 0, clusters of r5_report / r6_report)."""
    import r5_report as R5
    c5 = R5.cluster_of
    import r6_report as R6r  # sets R5.cluster_of to round 6's; both are kept here
    c6 = R6r.cluster_of
    out = {"settings": {}, "first": {}, "second": {}}
    meta = {"cells": {}, "rows_written": {}, "control": L7.CONTROL, "pick_sha256": L7.sha256_file(PICK_P) if PICK_P.exists() else None}
    LR = {}
    for arm, m in P["methods"].items():
        name = L7.METHODS[arm]
        out["settings"][name] = m["setting_text"] + (" (flagged: see safetune_pick.json)" if m.get("flag") else "")
        mc, bc = method_cells(arm, m)
        meta["cells"][name] = mc
        for test in ("first", "second"):
            lab = L7.r7_labels(mc[test])
            if not lab:
                continue
            rnd = L7.TABLES[test][0]
            if rnd not in LR:
                LR[rnd] = L7.round_labels(rnd)
            Lr = LR[rnd]
            base7 = L7.r7_labels(bc[test])
            R5.cluster_of = c5 if test == "first" else c6
            for row in L7.table_rows(test).values():
                keys = sorted(k for k in Lr["base"] if k.split("|")[0] in row["sets"])
                assert len(keys) == row["cells"]["base"]["n"], (test, row["row"])
                if not all(k in lab for k in keys):
                    continue
                sh = L7.shares_from(lab, keys)
                c = {"n": len(keys)}
                if row["kind"] == "H":
                    c.update(refused=sh["refused_only"], answered=sh["answered"], unsafe=sh["unsafe"], refusal=sh["refusal"])
                else:
                    c.update(refusal=sh["refusal"], unsafe=sh["unsafe"])
                stt = R5.cell_stats({"base": Lr["base"], "m": lab, L7.CONTROL: Lr.get(L7.CONTROL, {})}, "m", keys, control=L7.CONTROL)
                for tag in ("vs_control", "vs_base"):
                    if tag in stt:
                        c[tag] = stt[tag]
                if all(k in base7 for k in keys):
                    c["vs_base_same_run"] = paired(lab, base7, keys, [R5.cluster_of(k) for k in keys], R5)
                out[test].setdefault(name, {})[row["row"]] = c
            meta["rows_written"][f"{name} {test}"] = len(out[test].get(name, {}))
    out["meta"] = meta
    return out


def fake_tests():
    """A fake file in the report's format: every row of both tables, every method, with Control's own shares (for the format check only)."""
    out = {"settings": {v: "fake" for v in L7.METHODS.values()}, "first": {}, "second": {}}
    for test in ("first", "second"):
        Lr = L7.round_labels(L7.TABLES[test][0])
        for row in L7.table_rows(test).values():
            keys = sorted(k for k in Lr["base"] if k.split("|")[0] in row["sets"])
            if not all(k in Lr[L7.CONTROL] for k in keys):
                continue
            sh = L7.shares_from(Lr[L7.CONTROL], keys)
            c = {"n": len(keys), "refused": sh["refused_only"], "answered": sh["answered"], "unsafe": sh["unsafe"]} if row["kind"] == "H" else {"n": len(keys), "refusal": sh["refusal"]}
            for name in L7.METHODS.values():
                out[test].setdefault(name, {})[row["row"]] = c
    return out


def report_builder_check(path):
    """Build the report's LaTeX with the given safetune_tests.json (absolute path), without writing anything under presentation/ or results/."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(L7.ROOT / "presentation"))
    import build_report_tex as BR
    BR.ST_TESTS_FILE = str(Path(path).resolve())  # B.RES / <absolute path> is the absolute path
    S = BR.build()
    t = BR.grid(BR.tex(S))
    return t, S


def stage_report():
    if DRY and "--fake" in sys.argv:
        d = Path(sys.argv[sys.argv.index("--fake") + 1]).resolve()
        assert not str(d).startswith(str(L7.ROOT.resolve())), "the fake file must not be written inside build/"
        d.mkdir(parents=True, exist_ok=True)
        fp = d / "safetune_tests.json"
        fp.write_text(json.dumps(fake_tests(), indent=1))
        t_empty, _ = report_builder_check(d / "absent.json")
        t_fake, S = report_builder_check(fp)
        (d / "ayadial_results_fake.tex").write_text(t_fake)
        p0, p1 = t_empty.count("pending"), t_fake.count("pending")
        log(f"report builder: 'pending' appears {p0} times without the file and {p1} times with the fake file ({fp}); "
            f"rows filled: {sum(len(v) for t in ('first', 'second') for v in S['st_tests'][t].values())}")
        return
    P = load_pick()
    if DRY:
        for arm, m in P["methods"].items():
            mc, _ = method_cells(arm, m)
            log(f"{L7.METHODS[arm]}: {m['setting_text']}; cells: " + ", ".join(f"{n} ({'present' if (L7.OUT / 'labels' / (n + '.json')).exists() else 'missing'})" for t in mc.values() for n in t))
        if not P["methods"]:
            log("dry: no pick yet, nothing to report")
        return
    out = build_tests(P)
    out["meta"]["written_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    L7.write_json(L7.OUT / "safetune_tests.json", out)
    for test in ("first", "second"):
        for name, rows in out[test].items():
            for b in L7.block_sets():
                if b["test"] != test:
                    continue
                parts = []
                for side in ("harmful", "harmless"):
                    if side in b and b[side]["row"] in rows:
                        c = rows[b[side]["row"]]
                        parts.append(f"{side} {'unsafe ' + format(c['unsafe'], '.1f') if side == 'harmful' else 'refused ' + format(c['refusal'], '.1f')}"
                                     + (f" (vs Control {c['vs_control']['unsafe' if side == 'harmful' else 'refusal']:+.1f})" if "vs_control" in c else ""))
                if parts:
                    print(f"   {name:16s} {b['benchmark'][:34]:34s} " + "; ".join(parts))
    L7.sync()
    log("R7_REPORT DONE -> results/round7/safetune_tests.json")


# --- status
def stage_status():
    P = load_pick()
    left = 0
    for arm in ("rd", "adasteer", "cast"):
        log(f"{arm}: " + (P["methods"][arm]["setting_text"] if arm in P["methods"] else "not picked"))
    for ph in ("A", "B", "C"):
        for su, arm in test_cells(ph, P):
            name = cell_name(su, arm, P)
            n = len(L7.suite_rows(su))
            done = "<pick>" not in name and is_done(name, su)
            if not done:
                left += L7.est_minutes(su, n, arm)
            print(f"   {ph} {'done' if done else 'todo'} {name}")
    log(f"test cells left: about {left:.0f} min")


if __name__ == "__main__":
    st = ARGS[0] if ARGS else "status"
    if st == "calib":
        stage_calib()
    elif st == "smoke":
        stage_smoke()
    elif st == "pick":
        stage_pick(ARGS[1:])
    elif st == "test":
        stage_test(ARGS[1])
    elif st == "report":
        stage_report()
    elif st == "status":
        stage_status()
    else:
        sys.exit(__doc__)
