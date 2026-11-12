"""Round 7, J4: AyaDial-WSP (the editor's turn-96 edit, hash a0b53cb078d5352a) on the same test requests as the report's SafeTune rows.
  python scripts/r7_wsp.py check    reference AyaDial (v3g rebuilt as an edit, hash 652305b8341870ab) on one format suite, compared with round 6's
                                    stored labels for AyaDial on the same rows: says whether "edit written into the weights at s = 1" reproduces the
                                    report's AyaDial row in this pipeline
  python scripts/r7_wsp.py test     AyaDial-WSP at s = 1 on every test suite (r7lib.TEST_SUITES), cells <suite>.wsp@1
  python scripts/r7_wsp.py report   CPU. results/round7/wsp_tests.json in the format of safetune_tests.json (r7_safetune.build_tests)
The edit is written into layers[22].mlp gate_proj / up_proj / down_proj as install + s * knob in the model's dtype, as wsp/worker.py apply() does,
and removed afterwards. Edit files: runs_private/round7/wsp_edit/{wsp_t096,ref_v3g}.npz (built on the Mac from wsp/runs/real-01/edits/history/t096_*
and cache_r5/vec_v3g.npz; their hashes were checked there). Replies stay in runs_private/round7."""
import contextlib, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7lib as L7
from r7lib import log, rk, cell

EDIT_DIR = L7.RAW / "wsp_edit"
ARM = "wsp"
CHECK_SUITE = "xbf_xa_sysA"


def load_edit(name):
    z = np.load(EDIT_DIR / f"{name}.npz", allow_pickle=False)
    return {t: (z[f"{t}_units"].astype(np.int64), z[f"{t}_install"].astype(np.float32), z[f"{t}_knob"].astype(np.float32)) for t in ("gate_proj", "up_proj", "down_proj")}, str(z["hash"])


@contextlib.contextmanager
def edited(model, edit, scale=1.0):
    import torch
    mlp = model.model.layers[22].mlp
    saved = []
    try:
        for t, (units, inst, knob) in edit.items():
            W = getattr(mlp, t).weight
            u = torch.as_tensor(units, device=W.device)
            vals = torch.from_numpy(np.ascontiguousarray(inst + np.float32(scale) * knob, np.float32)).to(W.device, W.dtype)
            if t == "down_proj":
                saved.append((W, u, W.data[:, u].clone(), 1))
                W.data[:, u] = vals.T
            else:
                saved.append((W, u, W.data[u].clone(), 0))
                W.data[u] = vals
        yield model
    finally:
        for W, u, old, ax in reversed(saved):
            if ax:
                W.data[:, u] = old
            else:
                W.data[u] = old
        for W, u, old, ax in saved:
            assert torch.equal(W.data[:, u] if ax else W.data[u], old), "weights were not restored exactly"


def stage_check():
    import r7gpu as G
    model, tok = G.load()
    edit, h = load_edit("ref_v3g")
    rows = L7.suite_rows(CHECK_SUITE)
    with edited(model, edit) as m:
        lab = G.run(m, tok, CHECK_SUITE, cell(CHECK_SUITE, "refv3g", 1), rows)
    old = L7.J(L7.R6 / "labels" / f"{CHECK_SUITE}.v3gcap@1.json")
    keys = [rk(r) for r in rows if rk(r) in old]
    a, b = np.array([lab[k][:2] for k in keys], float), np.array([old[k][:2] for k in keys], float)
    out = {"suite": CHECK_SUITE, "edit_hash": h, "n": len(keys), "n_rows": len(rows), "labels_identical": float((a == b).all(1).mean()),
           "refusal_now": float(a[:, 0].mean() * 100), "refusal_round6": float(b[:, 0].mean() * 100), "unsafe_now": float(a[:, 1].mean() * 100), "unsafe_round6": float(b[:, 1].mean() * 100),
           "written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    L7.write_json(L7.OUT / "wsp_check.json", out)
    L7.sync()
    log(f"R7_WSP_CHECK {out}")


def stage_test():
    import r7gpu as G
    model, tok = G.load()
    edit, h = load_edit("wsp_t096")
    assert h == "a0b53cb078d5352a", h
    for su in L7.TEST_SUITES:
        rows = L7.suite_rows(su)
        with edited(model, edit) as m:
            G.run(m, tok, su, cell(su, ARM, 1), rows)
    L7.write_json(L7.OUT / "run" / "test_wsp.json", {"edit_hash": h, "scale": 1.0, "cells": [cell(su, ARM, 1) for su in L7.TEST_SUITES], "versions": G.versions(),
                                                    "finished_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())})
    L7.sync()
    log("R7_WSP_TEST DONE")


def stage_report():
    import r7_safetune as S7
    L7.METHODS[ARM] = "AyaDial-WSP"
    _, h = load_edit("wsp_t096")
    P = {"methods": {ARM: {"setting": 1, "setting_text": f"edit {h} written into the weights, s = 1"}}}
    T = S7.build_tests(P)
    L7.write_json(L7.OUT / "wsp_tests.json", T)
    L7.sync()
    for test in ("first", "second"):
        for row, c in T[test].get("AyaDial-WSP", {}).items():
            vc = c.get("vs_control", {})
            print(f"   {row:48s} n={c['n']:5d} " + (f"unsafe {c['unsafe']:.1f} " if "refused" in c else "") + f"refusal {c['refusal']:.1f}" + (f"  vs Control unsafe {vc.get('unsafe', float('nan')):+.1f} refusal {vc.get('refusal', float('nan')):+.1f}" if vc else ""))
    log("R7_WSP_REPORT DONE -> results/round7/wsp_tests.json")


SWEEP = {"wsp": ("wsp_t096", [0, 0.5, 0.75, 1, 1.25, 1.5]), "refv3g": ("ref_v3g", [0.5, 0.75, 1, 1.25, 1.5])}


def stage_sweep():
    """J5: the scale s of AyaDial-WSP (and of AyaDial through the same route) on the 1910 WildJailbreak development rows (suite wjx).
    Writes results/round7/wsp_sweep.json: per scale, unsafe on adversarial_harmful and refusal on adversarial_benign, with Base."""
    import r7gpu as G
    model, tok = G.load()
    rows = L7.suite_rows("wjx")
    H, Bk = [rk(r) for r in rows if r["set"] in L7.WJ_H], [rk(r) for r in rows if r["set"] in L7.WJ_B]
    base = G.run(model, tok, "wjx", cell("wjx"), rows)
    rate = lambda lab, keys, j: round(float(np.mean([lab[k][j] for k in keys]) * 100), 2)
    out = {"suite": "wjx (1910 WildJailbreak development rows)", "n_harmful": len(H), "n_harmless": len(Bk),
           "base": {"unsafe_harmful": rate(base, H, 1), "refusal_harmful": rate(base, H, 0), "refusal_harmless": rate(base, Bk, 0)}}
    for arm, (edit_name, grid) in SWEEP.items():
        edit, h = load_edit(edit_name)
        out[arm] = {"edit_hash": h, "scales": {}}
        for s in grid:
            with edited(model, edit, s) as m:
                lab = G.run(m, tok, "wjx", cell("wjx", arm, s), rows)
            out[arm]["scales"][f"{s:g}"] = {"unsafe_harmful": rate(lab, H, 1), "refusal_harmful": rate(lab, H, 0), "refusal_harmless": rate(lab, Bk, 0),
                                            "degenerate": rate(lab, H + Bk, 2), "labels_identical_to_base": round(float(np.mean([lab[k][:2] == base[k][:2] for k in H + Bk])), 4)}
            L7.write_json(L7.OUT / "wsp_sweep.json", out)
            L7.sync()
    log("R7_WSP_SWEEP DONE -> results/round7/wsp_sweep.json")


if __name__ == "__main__":
    {"check": stage_check, "test": stage_test, "report": stage_report, "sweep": stage_sweep}[sys.argv[1]]()
