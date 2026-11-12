"""Build the read-only case folder of one run (trusted code, runs in the driver, outside the sandbox).

  case/
    lib/kernel.py, lib/edits.py      the kernel's own code (copied, so the sandbox needs to read nothing in wsp/)
    kernel_config.json               loadable tensor names, the weight slice path, the edit rules
    knob.json                        the starting knob: name, units, constants, construction in words, allowed units and census
    knob_vectors.npz                 m, q, w, r of the starting knob (float32 [2048] each)
    ayadial_vectors.npz              the same four vectors for AyaDial (round 4), rebuilt from its adapter files
    start_edit/{gate,up,down}_proj.npz   the starting knob as a compact edit (see edits.py)
    sets.json, formats.json          what a forward pass may name (ids are 0..n-1 inside a set; no prompt text anywhere)
    facts.json                       every development table of the first message, with its source file

The weight slice (layers 18 to 22 only, original bf16) is extracted once from the Hugging Face cache into
wsp/cache/ and shared by runs. It holds no embedding and no other layer, so the kernel cannot run the model.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np

import config as C
import edits as E

KNOB_NAMES = {"ayadial": "AyaDial", "ayadial_improved": "AyaDial-Improved"}
INTERNAL = {"ayadial": "round 4 'foldcap' adapters, internal name v1", "ayadial_improved": "round 5 'v3gcap', internal name v3g"}


# == weights ==
def hf_snapshot(cfg):
    root = C.path(cfg["weights"]["hf_dir"])
    snaps = sorted((root / "snapshots").glob("*"))
    if not snaps:
        raise FileNotFoundError(f"no snapshot under {root}")
    return snaps[-1]


def slice_names(cfg):
    out = []
    for L in cfg["weights"]["read_layers"]:
        out += [f"model.layers.{L}.input_layernorm.weight"]
        out += [f"model.layers.{L}.self_attn.{p}_proj.weight" for p in "qkvo"]
        out += [f"model.layers.{L}.mlp.{p}_proj.weight" for p in ("gate", "up", "down")]
    return out


def ensure_weight_slice(cfg, log=print):
    """Extract the allowed tensors (memory-mapped reads, no model load) into one safetensors file. Returns (path, manifest)."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    dst = C.path(cfg["weights"]["slice_file"])
    man_path = dst.with_suffix(".manifest.json")
    names = slice_names(cfg)
    if dst.exists() and man_path.exists():
        man = json.loads(man_path.read_text())
        if man.get("names") == names:
            return dst, man
    snap = hf_snapshot(cfg)
    wm = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    tensors, shards = {}, {}
    for n in names:
        shard = snap / wm[n]
        shards[wm[n]] = Path(shard).resolve().name  # the blob name of an LFS file is its sha256
        with safe_open(str(shard), framework="pt") as f:
            tensors[n] = f.get_tensor(n).contiguous()
    dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst))
    man = {"names": names, "source_snapshot": snap.name, "source_shards_blob_sha256": shards,
           "dtype": str(next(iter(tensors.values())).dtype), "bytes": dst.stat().st_size}
    man_path.write_text(json.dumps(man, indent=1))
    log(f"weight slice written: {dst} ({man['bytes'] / 1e6:.0f} MB, {len(names)} tensors)")
    return dst, man


class Originals:
    """Original float32 tensors of layer 22's MLP from the slice file."""

    def __init__(self, slice_file):
        from safetensors import safe_open
        self.f = safe_open(str(slice_file), framework="pt")
        self.cache = {}

    def __call__(self, name):
        if name not in self.cache:
            self.cache[name] = self.f.get_tensor(name).float().numpy()
        return self.cache[name]


# == constants from files ==
def fold_constants():
    """alpha, kappa, beta, the layer and the strict units, parsed from scripts/r5lib.py (the code that built the cells)."""
    src = (C.BUILD / "scripts/r5lib.py").read_text()
    m = re.search(r"LYR, ALPHA, KAPPA, BETA = ([\d.]+), ([\d.]+), ([\d.]+), ([\d.]+)", src)
    u = re.search(r"STRICT_UNITS, LAX_UNITS = \((\d+), (\d+)\), \((\d+), (\d+)\)", src)
    g = re.search(r"def strict_units\(V, s, beta=BETA, gain=([\d.]+)", src)
    return {"layer": int(float(m.group(1))), "alpha": float(m.group(2)), "kappa": float(m.group(3)), "beta": float(m.group(4)),
            "gain": float(g.group(1)), "strict_units": [int(u.group(1)), int(u.group(2))], "lax_units": [int(u.group(3)), int(u.group(4))],
            "source": "scripts/r5lib.py (LYR, ALPHA, KAPPA, BETA, STRICT_UNITS, strict_units gain default)"}


def theta0_of(which):
    if which == "ayadial":
        return float(json.loads((C.BUILD / "results/round4/fold/fit.json").read_text())["theta0"]), "results/round4/fold/fit.json"
    return float(json.loads((C.BUILD / "results/round5/lab/v3g.json").read_text())["theta0"]), "results/round5/lab/v3g.json"


def refusal_direction(cfg):
    D = np.load(C.path(cfg["sources"]["refusal_direction"]))
    return D["r_raw"][23].astype(np.float32)


def census(cfg):
    d = json.loads(C.path(cfg["edit"]["census_file"]).read_text())
    return d["dormant_units_layer22"]


def allowed_units(cfg):
    ed = cfg["edit"]
    fold = list(ed["fold_units"])
    extra = [c["unit"] for c in census(cfg) if c["unit"] not in fold and c["unit"] not in ed["exclude_units"]][: ed["extra_units_max"]]
    return fold + extra


def rules(cfg):
    ed = cfg["edit"]
    return {"allowed_units": allowed_units(cfg), "max_units": ed["max_units"], "knob_tensors": ed["knob_tensors"],
            "max_row_norm": ed["max_row_norm"], "max_knob_row_norm": ed["max_knob_row_norm"]}


# == the two knobs as compact edits ==
def units_from_vectors(V, k):
    """Round 5's capped fold (scripts/r5lib.py strict_units) split into install (s = 0) and knob (per unit of s) parts."""
    a, kap, b, gain = k["alpha"], k["kappa"], k["beta"], k["gain"]
    m, q, w, th, r = V["m"], V["q"], V["w"], V["theta0"], V["r"]
    gA0 = -a * th * m - kap * q
    gB0 = -a * (th + 1.0 / b) * m - kap * q
    d = gain * b * r / a
    u = k["strict_units"]
    return {"gate_proj": (u, np.stack([gA0, gB0]), np.stack([a * w, a * w])),
            "up_proj": (u, np.stack([m, m]), np.zeros((2, 2048), np.float32)),
            "down_proj": (u, np.stack([d, -d]), np.zeros((2, 2048), np.float32))}


def ayadial_from_adapters(cfg, orig, k):
    """AyaDial (round 4) exactly as shipped: original rows + install LoRA, knob LoRA on gate rows. Also recovers m, q, w, r."""
    from safetensors.numpy import load_file
    src = C.path(cfg["sources"]["ayadial_adapters"])
    I = load_file(str(src / "install/adapter_model.safetensors"))
    K = load_file(str(src / "knob/adapter_model.safetensors"))
    for p in ("install", "knob"):
        ac = json.loads((src / p / "adapter_config.json").read_text())
        assert ac["lora_alpha"] == ac["r"] and not ac.get("use_rslora"), "LoRA scaling must be 1"
    pre = "base_model.model.model.layers.22.mlp."
    u = k["strict_units"]
    G0, U0, D0 = (orig(f"model.layers.22.mlp.{t}.weight") for t in ("gate_proj", "up_proj", "down_proj"))
    dG = I[pre + "gate_proj.lora_B.weight"][u] @ I[pre + "gate_proj.lora_A.weight"]
    dU = I[pre + "up_proj.lora_B.weight"][u] @ I[pre + "up_proj.lora_A.weight"]
    dD = (I[pre + "down_proj.lora_B.weight"] @ I[pre + "down_proj.lora_A.weight"][:, u]).T
    dK = K[pre + "gate_proj.lora_B.weight"][u] @ K[pre + "gate_proj.lora_A.weight"]
    parts = {"gate_proj": (u, G0[u] + dG, dK), "up_proj": (u, U0[u] + dU, np.zeros_like(dK)),
             "down_proj": (u, D0[:, u].T + dD, np.zeros_like(dK))}
    th, _ = theta0_of("ayadial")
    a, kap, b, gain = k["alpha"], k["kappa"], k["beta"], k["gain"]
    m = parts["up_proj"][1][0]
    w = dK[0] / a
    qA = -(parts["gate_proj"][1][0] + a * th * m) / kap
    qB = -(parts["gate_proj"][1][1] + a * (th + 1 / b) * m) / kap
    r_eff = parts["down_proj"][1][0] * a / (gain * b)
    checks = {"up rows equal": float(np.abs(parts["up_proj"][1][0] - parts["up_proj"][1][1]).max()),
              "q from unit A vs unit B, max abs diff": float(np.abs(qA - qB).max()),
              "knob rows equal": float(np.abs(dK[0] - dK[1]).max()),
              "down columns opposite": float(np.abs(parts["down_proj"][1][0] + parts["down_proj"][1][1]).max())}
    vec = {"m": m.astype(np.float32), "q": ((qA + qB) / 2).astype(np.float32), "w": w.astype(np.float32),
           "r": r_eff.astype(np.float32), "theta0": np.float32(th)}
    return parts, vec, checks


def to_state(parts):
    st = {}
    for short, (u, inst, knob) in parts.items():
        st[E.SHORT[short]] = {"name": E.SHORT[short], "units": np.asarray(u, np.int64), "install": np.asarray(inst, np.float32),
                              "knob": np.asarray(knob, np.float32)}
    return st


def start_knob(cfg, orig):
    """(state, vectors, meta) for the configured starting knob; raises FileNotFoundError with a plain message if inputs are missing."""
    k = fold_constants()
    parts1, vec1, checks1 = ayadial_from_adapters(cfg, orig, k)
    r = refusal_direction(cfg)
    vec1["cos_r_with_day1_r"] = np.float32(vec1["r"] @ r / np.linalg.norm(vec1["r"]) / np.linalg.norm(r))
    if cfg["start_knob"] == "ayadial":
        parts, vec = parts1, {k_: v for k_, v in vec1.items() if k_ in ("m", "q", "w", "r")}
        th, th_src = theta0_of("ayadial")
    else:
        p = C.path(cfg["sources"]["ayadial_improved_vec"])
        if not p.exists():
            raise FileNotFoundError(f"AyaDial-Improved's vector file is missing: {p} (round 5 cache_r5/vec_v3g.npz; on the Drive mirror "
                                    f"lexsi-knob/round5/cache/). Copy it there, or run with start_knob=ayadial.")
        V = {kk: (vv.astype(np.float32) if vv.ndim else float(vv)) for kk, vv in np.load(p).items()}
        V["r"] = r
        parts = units_from_vectors(V, k)
        vec = {"m": V["m"], "q": V["q"], "w": V["w"], "r": r}
        th, th_src = V["theta0"], str(p)
    meta = {"name": KNOB_NAMES[cfg["start_knob"]], "internal": INTERNAL[cfg["start_knob"]], "units": k["strict_units"],
            "layer": k["layer"], "alpha": k["alpha"], "kappa": k["kappa"], "beta": k["beta"], "gain": k["gain"], "theta0": float(th),
            "constants_source": k["source"] + "; theta0 from " + th_src, "ayadial_rebuild_checks": checks1}
    return to_state(parts), vec, vec1, meta


# == catalogue of what passes may name ==
FORMAT_LABELS = [("default", "default template"), ("sys_bank", "one-line preamble"), ("sys_car", "car assistant preamble"),
                 ("sys_persona", "persona preamble"), ("sys_de", "German preamble"), ("sys_long", "long preamble (house style)"),
                 ("sys_law", "long preamble (law office)"), ("nosys", "no system turn"), ("mt_benign", "after two harmless turns"),
                 ("mt_refused", "after a refused turn"), ("mt3", "after three harmless turns"), ("mt_sys", "preamble and two turns"),
                 ("inj_tail", "typed tokens at the end"), ("inj_head", "typed fake exchange before")]
FORMAT_LABELS_SOURCE = "labels copied from scripts/r6_summary.py (FMT_DEV)"


def catalogue():
    dev = json.loads((C.BUILD / "results/round5/tables/dev_main.json").read_text())
    lab = json.loads((C.BUILD / "results/round5/lab/v3g.json").read_text())
    fitted = {**lab["fit_harmful"], **lab["fit_harmless"]}
    label = {"H": "harmful", "B": "harmless", "C": "contested"}
    sets, singles = [], set()
    for r in dev["rows"]:
        if len(r["sets"]) == 1:
            singles.add(r["sets"][0])
    for r in dev["rows"]:
        if len(r["sets"]) == 1 or not set(r["sets"]) <= singles:
            sid = r["sets"][0] if len(r["sets"]) == 1 else r["row"]
            n_fit = sum(v for k, v in fitted.items() if k.split(":")[0] in r["sets"] or k == sid)
            sets.append({"set": sid, "label": label[r["kind"]], "n": r["cells"]["base"]["n"], "description": r["row"],
                         "in_fit_of_ayadial_improved": n_fit, "source": "results/round5/tables/dev_main.json"})
    xd = json.loads((C.BUILD / "results/round6/run/d_xd6.json").read_text())
    lang = {"fil": "Filipino", "fr": "French", "hi": "Hindi", "ru": "Russian", "sr": "Serbian", "es": "Spanish"}
    for k, v in xd["base"].items():
        if k.startswith("xd_art_") or k.startswith("xd_ayab_"):
            code = k.split("_")[-1]
            harm = k.startswith("xd_art_")
            sets.append({"set": k, "label": "harmful" if harm else "harmless", "n": v["n"],
                         "description": f"{'Aya red-teaming prompts' if harm else 'Aya harmless prompts'}, {lang[code]} (round 6 development)",
                         "in_fit_of_ayadial_improved": 0, "source": "results/round6/run/d_xd6.json"})
    rb = json.loads((C.BUILD / "results/round6/robust/rb_v4.json").read_text())
    nh, nb = rb["default"]["base"]["n_harmful"], rb["default"]["base"]["n_harmless"]
    sets.append({"set": "format_pool", "label": "mixed", "n": nh + nb, "n_harmful": nh, "n_harmless": nb,
                 "description": f"the format pool: {nh} harmful and {nb} harmless requests, rendered in any format of formats.json",
                 "in_fit_of_ayadial_improved": 0, "source": "results/round6/robust/rb_v4.json"})
    formats = [{"format": f, "description": d} for f, d in FORMAT_LABELS]
    return sets, formats


# == build ==
def build_case(cfg, run_dir, facts, log=print):
    run_dir = Path(run_dir)
    case = run_dir / "case"
    if case.exists():
        shutil.rmtree(case)
    (case / "lib").mkdir(parents=True)
    for f in ("kernel.py", "edits.py"):
        shutil.copy2(C.WSP / f, case / "lib" / f)
    wfile, wman = ensure_weight_slice(cfg, log)
    orig = Originals(wfile)
    state, vec, vec1, meta = start_knob(cfg, orig)
    rl = rules(cfg)
    for name, c in state.items():
        errs = E.validate(c, rl, [u for n2, c2 in state.items() if n2 != name for u in c2["units"].tolist()])
        if errs:
            raise ValueError(f"the starting knob breaks the edit rules: {errs}")
    (case / "start_edit").mkdir()
    for name, c in state.items():
        E.save_npz(case / "start_edit" / f"{E.EDITABLE[name]}.npz", c)
    np.savez(case / "knob_vectors.npz", **{k: np.asarray(v, np.float32) for k, v in vec.items()})
    np.savez(case / "ayadial_vectors.npz", **{k: np.asarray(v, np.float32) for k, v in vec1.items() if k in ("m", "q", "w", "r")})
    cens = {c["unit"]: c for c in census(cfg)}
    meta.update({"allowed_units": rl["allowed_units"],
                 "census_of_allowed_units": [{"unit": u, "mean_abs_act": round(cens[u]["mean_abs_act"], 5),
                                              "down_col_norm": round(cens[u]["out_col_norm"], 4)} for u in rl["allowed_units"]],
                 "census_source": cfg["edit"]["census_file"],
                 "norms": {k: float(np.linalg.norm(v)) for k, v in vec.items()},
                 "start_edit_hash": E.state_hash(state)})
    (case / "knob.json").write_text(json.dumps(meta, indent=1))
    names = slice_names(cfg)
    (case / "kernel_config.json").write_text(json.dumps({"weights_file": str(wfile), "names": names, "rules": rl}, indent=1))
    sets, formats = catalogue()
    (case / "sets.json").write_text(json.dumps(sets, indent=1))
    (case / "formats.json").write_text(json.dumps(formats, indent=1))
    (case / "facts.json").write_text(json.dumps(facts, indent=1))
    digest = hashlib.sha256()
    for p in sorted(case.rglob("*")):
        if p.is_file():
            digest.update(p.name.encode() + p.read_bytes())
    return {"case": str(case), "weights_file": str(wfile), "weights_manifest": wman, "start_state": state, "rules": rl,
            "sets": sets, "formats": formats, "knob": meta, "case_sha256": digest.hexdigest()[:16]}
