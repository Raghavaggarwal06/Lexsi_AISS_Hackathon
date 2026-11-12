"""Rebuild v1 from the round 4 adapter files (the float vectors of round 4 lived only on that night's runtime).
v1 = install adapter + s * knob adapter on units 6666 and 1896 of layers[22].mlp. Writes cache_r5/vec_v1.npz and results/round5/v1_check.json.
Also restores last night's reusable development cells (wj and own rows only) into the round 5 folders.
  python scripts/r5_v1.py"""
import hashlib, shutil
from r5lib import *
from safetensors.torch import load_file

AD = R4RAW / "adapters/fold_cap"
fz = json.loads((OUT / "v1_freeze.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
chk = {"files_match_freeze": {}}
for rel, want in fz["sha256"].items():
    p = ROOT / rel
    if p.exists():
        chk["files_match_freeze"][rel] = sha(p) == want
assert all(chk["files_match_freeze"].values()), chk
model, tok = load()
ins, knob = load_file(str(AD / "install/adapter_model.safetensors")), load_file(str(AD / "knob/adapter_model.safetensors"))
pre = "base_model.model.model.layers.22.mlp."
mlp = model.model.layers[22].mlp
G, U, D = (x.weight.data.float().cpu() for x in (mlp.gate_proj, mlp.up_proj, mlp.down_proj))
V = {}
for j, n in enumerate(STRICT_UNITS):
    for mod in ("gate_proj", "up_proj"):
        B = ins[pre + f"{mod}.lora_B.weight"]
        assert float(B[n, j]) == 1.0 and float(B.abs().sum(0)[j]) == 1.0, "install B is not a unit selector"
    assert float(ins[pre + "down_proj.lora_A.weight"][j, n]) == 1.0
    assert float(knob[pre + "gate_proj.lora_B.weight"][n, j]) == 1.0
    V[f"gate0_{n}"] = (G[n] + ins[pre + "gate_proj.lora_A.weight"][j]).numpy()
    V[f"up_{n}"] = (U[n] + ins[pre + "up_proj.lora_A.weight"][j]).numpy()
    V[f"down_{n}"] = (D[:, n] + ins[pre + "down_proj.lora_B.weight"][:, j]).numpy()
    V[f"knob_{n}"] = knob[pre + "gate_proj.lora_A.weight"][j].numpy()
fit = json.loads((R4OUT / "fold/fit.json").read_text())
th, b, Uz = fit["theta0"], fit["b"], fit["U"]
a, b2 = STRICT_UNITS
m = V[f"up_{a}"].astype(np.float64)
w = V[f"knob_{a}"].astype(np.float64) / ALPHA
q = -(V[f"gate0_{a}"].astype(np.float64) + ALPHA * th * m) / KAPPA
r = rvec().astype(np.float64)
cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
gB_pred = -ALPHA * (th + 1.0 / BETA) * m - KAPPA * q
chk.update({"theta0": th, "b": b, "U": Uz, "norm_w": float(np.linalg.norm(w)), "norm_w_in_fit_json": fit["w"]["norm"],
            "up_rows_equal_max_abs_diff": float(np.abs(V[f"up_{a}"] - V[f"up_{b2}"]).max()), "knob_rows_equal_max_abs_diff": float(np.abs(V[f"knob_{a}"] - V[f"knob_{b2}"]).max()),
            "second_gate_row_vs_formula_max_abs_diff": float(np.abs(V[f"gate0_{b2}"] - gB_pred).max()), "second_gate_row_norm": float(np.linalg.norm(V[f"gate0_{b2}"])),
            "down_first_vs_r_over_5": {"cos": cos(V[f"down_{a}"].astype(np.float64), r), "max_abs_diff": float(np.abs(V[f"down_{a}"] - BETA * r / ALPHA).max())},
            "down_second_vs_minus_r_over_5": {"cos": cos(V[f"down_{b2}"].astype(np.float64), -r), "max_abs_diff": float(np.abs(V[f"down_{b2}"] + BETA * r / ALPHA).max())}})
V.update({"m": m.astype(np.float32), "q": q.astype(np.float32), "w": w.astype(np.float32), "theta0": np.float64(th), "b": np.float64(b), "U": np.float64(Uz)})
np.savez(CACHE / "vec_v1.npz", **V)
# the formula path (strict_units) against the shipped rows, in the dtype the model uses
Vf = vec("v1")
worst = 0.0
for s in (0.0, 1.0, 1.25, 2.0):
    for u1, u2 in zip(v1_units(Vf, s), strict_units(Vf, s)):
        for k in ("gate", "up", "down"):
            x1, x2 = torch.tensor(u1[k]).to(torch.bfloat16).float(), torch.tensor(np.asarray(u2[k], np.float32)).to(torch.bfloat16).float()
            worst = max(worst, float((x1 - x2).abs().max()))
chk["formula_vs_shipped_rows_max_abs_diff_bf16"] = worst
chk["vec_v1_sha256"] = sha(CACHE / "vec_v1.npz")
chk["pass"] = bool(chk["up_rows_equal_max_abs_diff"] < 1e-6 and chk["second_gate_row_vs_formula_max_abs_diff"] < 1e-3 and chk["down_first_vs_r_over_5"]["cos"] > 0.99999 and abs(chk["norm_w"] - chk["norm_w_in_fit_json"]) < 1e-3)
(OUT / "v1_check.json").write_text(json.dumps(chk, indent=1))
print(json.dumps(chk, indent=1))

# ---- restore last night's reusable cells: only wj and own rows, whose batching the suites 'wjx' and 'own' reproduce
KEEP = set(WJ_H + WJ_B + OWN)
REUSE = ["base_wjx", "base_own"] + [f"ungated@{s}" for s in ("0.02", "0.05", "0.08", "0.1", "0.15", "0.2")] + [f"foldcap@{s}" for s in ("-1", "0", "0.75", "1", "1.25", "1.5", "2")] + ["gate98_last@1", "foldcaphook@1"]
(RAW / "r4_reuse").mkdir(exist_ok=True)
n_rows = {}
for name in REUSE:
    src = R4RAW / "gen" / f"{name}.jsonl"
    if src.exists():
        rows = [x for x in read_jsonl(src) if x["set"] in KEEP]
        write_jsonl(RAW / "r4_reuse" / f"{name}.jsonl", rows)
        n_rows[name] = len(rows)
if not (RAW / "scores.jsonl").exists():
    shutil.copy(R4RAW / "scores.jsonl", RAW / "scores.jsonl")
if (R4RAW / "scores_wg.jsonl").exists() and not (RAW / "scores_wg.jsonl").exists():
    shutil.copy(R4RAW / "scores_wg.jsonl", RAW / "scores_wg.jsonl")
(OUT / "r4_reuse_candidates.json").write_text(json.dumps({"note": "round 4 cells filtered to wj and own rows, parked in runs_private/round5/r4_reuse until the reproduction check passes", "rows": n_rows}, indent=1))
sync()
log("R5_V1 DONE")
