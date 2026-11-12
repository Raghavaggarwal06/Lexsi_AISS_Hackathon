"""Step 2: SafeTune hook baselines (RefusalDirection, CAST, AdaSteer) on the same eval rows as phase 2.

SafeTune (https://github.com/Lexsi-Labs/SafeTune, PyPI `safetune`) supplies the interventions:
  rd_ours            safetune.steer.RefusalDirectionModel hosting OUR layer-23 direction (parity arm)
  refusal_direction  safetune.runner.steer.RefusalDirectionTrainer -> RefusalDirectionModel
  cast               safetune.runner.steer.CASTTrainer -> CASTModel (CAA vectors + cosine gate)
  adasteer           safetune.runner.steer.AdaSteerTrainer -> AdaSteerModel (adaptive coefficients)
Arms and strengths come from safetune_baselines.yaml.

Calibration uses the fit halves only (harmful vs pooled harmless sets, calib_n per class, seed 0).
Generation uses the eval halves, the same chat template, left padding, greedy decoding and 128 new
tokens as 04_steer_sweep.py, so the output files are drop-in for 05_score.py's format and for step 3:
  build/runs_private/safetune/<arm>/a=<strength>.jsonl   {"id", "set", "prompt", "response"}  (local only)
  results/round3/block8/safetune_manifest.json           arms, strengths, calibration ids, timings (no text)

--curated results/round3/block8/curated_ids.json drops CuratorKIT's duplicates from both halves.

--dry-run builds a 4-layer random-weight Cohere2ForCausalLM from Tiny Aya's own config.json (no Tiny
Aya weights) and runs every arm end to end on a few prompts. It checks the wiring (SafeTune finds the
cohere2 decoder layers, each trainer calibrates, each wrapper generates); its outputs are noise and
are written without prompt text to --dry-out.

Usage (from build/, GPU machine):
  .venv-pipeline/bin/python scripts/pipeline/p2_safetune_baselines.py --arms all
  .venv-pipeline/bin/python scripts/pipeline/p2_safetune_baselines.py --arms rd_ours,cast --n 50
  .venv-pipeline/bin/python scripts/pipeline/p2_safetune_baselines.py --dry-run --dry-out /tmp/st_dry
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import BUILD, OUT, SAFETUNE_RUNS, SWEEP_SETS, load_set, load_splits, write_json  # noqa: E402

import common  # noqa: E402  (build/scripts/common.py: render, encode, load_model, load_tokenizer, pick_device)

HERE = Path(__file__).resolve().parent


def eval_and_calib(cfg, n, curated, calib_n, seed=0):
    splits = load_splits()
    keep = None
    if curated:
        ids = json.loads(Path(curated).read_text())["ids"]
        keep = {(name, i) for name, halves in ids.items() for half in halves.values() for i in half}
    ok = lambda name, i: keep is None or (name, i) in keep
    tables = {name: {r["id"]: r for r in load_set(name)} for name in set(SWEEP_SETS) | set(cfg["harmless_calib_sets"])}
    rows = []
    for name in SWEEP_SETS:
        rows += [tables[name][i] for i in splits[name]["eval"][:n] if ok(name, i)]
    rng = random.Random(seed)
    harmful = [tables["harmful"][i] for i in splits["harmful"]["fit"] if ok("harmful", i)]
    harmless = [tables[s][i] for s in cfg["harmless_calib_sets"] for i in splits[s]["fit"] if ok(s, i)]
    harmful = rng.sample(harmful, min(calib_n, len(harmful)))
    harmless = rng.sample(harmless, min(calib_n, len(harmless)))
    return rows, harmful, harmless


def build_arm(name, arm, model, tok, harmful, harmless, layer_hidden):
    """Returns (wrapper, set_strength(wrapper, a), cleanup(wrapper))."""
    import safetune.steer as S
    from safetune.runner.steer import AdaSteerTrainer, CASTTrainer, RefusalDirectionTrainer

    method = arm["method"]
    hp, bp = [r["prompt"] for r in harmful], [r["prompt"] for r in harmless]
    if method == "rd_ours":
        if layer_hidden is None:  # dry run: model width differs from Tiny Aya, use a random unit vector
            r = torch.randn(model.config.hidden_size)
            norm, layer = 1.0, model.config.num_hidden_layers - 1
        else:
            saved = np.load(BUILD / "results" / "day1" / "directions.npz")
            layer = int(saved["selected_layer"])
            r = torch.tensor(saved["r_raw"][layer], dtype=torch.float32)
            norm, layer = float(r.norm()), layer - 1  # hidden-state index 23 -> decoder layer 22, as in 04
        wrapped = S.RefusalDirectionModel(model, direction=r, mode="steer", strength=0.0, layers=[layer])
        return wrapped, lambda w, a: setattr(w, "strength", float(a) * norm), lambda w: w.remove()
    if method == "refusal_direction":
        wrapped, _ = RefusalDirectionTrainer(model, tok, layers=arm.get("layers"), alpha=1.0).calibrate(harmful=hp, harmless=bp)
        return wrapped, lambda w, a: setattr(w, "strength", float(a)), lambda w: w.remove()
    if method == "cast":
        wrapped, _ = CASTTrainer(model, tok, alpha=1.0).calibrate(harmful=hp, harmless=bp)
        return wrapped, lambda w, a: setattr(w, "alpha", float(a)), lambda w: w.remove()
    if method == "adasteer":
        wrapped, _ = AdaSteerTrainer(model, tok, alpha=1.0, layers=arm.get("layers")).calibrate(harmful=hp, harmless=bp)

        def set_strength(w, a):
            w.config.rd_max = float(a)
            w.config.hd_max = float(a)
        return wrapped, set_strength, lambda w: w.remove_hooks()
    raise ValueError(method)


@torch.no_grad()
def generate(wrapped, tok, rows, batch, max_len, max_new, device):
    out = []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        enc = common.encode(tok, [common.render(tok, r["prompt"]) for r in chunk], max_len, device)
        ids = wrapped.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None,
                               max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


@torch.no_grad()
def parity_check(model, tok, rows, r, layer, scale, max_len, device):
    """Max |logit difference| between SafeTune's RefusalDirectionModel and 04_steer_sweep.py's own
    ungated hook, both pushing scale * r at the output of decoder layer `layer`, on the same batch."""
    import importlib.util
    import safetune.steer as S
    spec = importlib.util.spec_from_file_location("steer_sweep", BUILD / "scripts" / "04_steer_sweep.py")
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    param = next(model.parameters())
    enc = common.encode(tok, [common.render(tok, x["prompt"]) for x in rows], max_len, device)
    plain = model(**enc).logits.float()
    # "ungated" reads h @ d only for its shape (weight = ones), so any d of the right width works.
    hook, _ = sweep.make_hook("ungated", scale, {"r": r.to(param.device, param.dtype), "d": torch.zeros_like(r).to(param.device, param.dtype)})
    handle = model.model.layers[layer].register_forward_hook(hook)
    ours = model(**enc).logits.float()
    handle.remove()
    wrapped = S.RefusalDirectionModel(model, direction=r, mode="steer", strength=scale * float(r.norm()), layers=[layer])
    wrapped.install()
    theirs = model(**enc).logits.float()
    wrapped.remove()
    return {"scale": scale, "layer": layer, "max_abs_logit_diff_safetune_vs_ours": float((ours - theirs).abs().max()),
            "max_abs_logit_shift_from_steering": float((ours - plain).abs().max())}


def tiny_cohere2(device):
    """4-layer random Cohere2ForCausalLM with Tiny Aya's own config and tokenizer (wiring check only)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained("CohereLabs/tiny-aya-global")
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 4, 128, 256
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 32
    cfg.layer_types = cfg.layer_types[:4]
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "safetune_baselines.yaml"))
    ap.add_argument("--arms", default="all")
    ap.add_argument("--n", type=int, default=250, help="cap per eval half (146 harmful, 125 others at most)")
    ap.add_argument("--curated", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parity", action="store_true", help="logit parity of SafeTune's RefusalDirectionModel vs our 04 hook (always on in --dry-run)")
    ap.add_argument("--dry-out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    arms = cfg["arms"] if args.arms == "all" else {k: cfg["arms"][k] for k in args.arms.split(",")}
    device = common.pick_device(args.device)
    tok = common.load_tokenizer(cfg["model"])
    max_len = common.load_config()["max_len"]

    if args.dry_run:
        model = tiny_cohere2(device)
        rows, harmful, harmless = eval_and_calib(cfg, 2, args.curated, 6)
        max_new, out_root, layer_hidden = 4, Path(args.dry_out), None
    else:
        model, _ = common.load_model(cfg["model"], device)
        rows, harmful, harmless = eval_and_calib(cfg, args.n, args.curated, cfg["calib_n"])
        max_new, out_root, layer_hidden = cfg["max_new"], SAFETUNE_RUNS, 23
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {"model": "random 4-layer cohere2 (dry run)" if args.dry_run else cfg["model"], "device": device,
                "dry_run": args.dry_run, "curated": args.curated, "n_eval_rows": len(rows),
                "calibration": {"harmful_ids": [r["id"] for r in harmful], "harmless_ids": [f"{r['set']}|{r['id']}" for r in harmless]},
                "arms": {}, "failures": {}, "parity": []}
    if args.dry_run or args.parity:
        probe_rows = [r for r in rows if r["set"] == "benign"][:8]
        if args.dry_run:
            torch.manual_seed(1)
            r_vec, layer = torch.randn(model.config.hidden_size) * 5, model.config.num_hidden_layers - 1
        else:
            saved = np.load(BUILD / "results" / "day1" / "directions.npz")
            r_vec, layer = torch.tensor(saved["r_raw"][int(saved["selected_layer"])], dtype=torch.float32), int(saved["selected_layer"]) - 1
        for scale in (0.5, -1.0):
            manifest["parity"].append(parity_check(model, tok, probe_rows, r_vec, layer, scale, max_len, device))
        print("parity:", manifest["parity"])
    for name, arm in arms.items():
        if args.dry_run and arm.get("layers"):
            arm = {**arm, "layers": [model.config.num_hidden_layers - 1]}  # layer 22 does not exist in the 4-layer stand-in
        t0 = time.time()
        try:
            wrapped, set_strength, cleanup = build_arm(name, arm, model, tok, harmful, harmless, layer_hidden)
        except Exception as err:  # record and move on: one broken arm should not sink the others
            manifest["failures"][name] = f"calibrate: {type(err).__name__}: {err}"
            print(f"{name}: calibration failed: {err}")
            continue
        manifest["arms"][name] = {"method": arm["method"], "calibrate_seconds": round(time.time() - t0, 1), "strengths": {}}
        for a in arm["strengths"]:
            t1 = time.time()
            try:
                set_strength(wrapped, a)
                responses = generate(wrapped, tok, rows, 2 if args.dry_run else cfg["batch"], max_len, max_new, device)
            except Exception as err:
                manifest["failures"][f"{name}|{a:g}"] = f"generate: {type(err).__name__}: {err}"
                print(f"{name} a={a:g}: generation failed: {err}")
                continue
            path = out_root / name / f"a={a:g}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                for r, resp in zip(rows, responses):
                    row = {"id": r["id"], "set": r["set"], "response": resp}
                    if not args.dry_run:
                        row["prompt"] = r["prompt"]  # local-only folder; step 3 needs it to hash and judge
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest["arms"][name]["strengths"][f"{a:g}"] = {"rows": len(responses), "seconds": round(time.time() - t1, 1)}
            print(f"{name} a={a:g}: {len(responses)} rows in {time.time() - t1:.1f}s")
        cleanup(wrapped)
    write_json((Path(args.dry_out) / "manifest.json") if args.dry_run else (OUT / "safetune_manifest.json"), manifest)
    print(json.dumps({"arms_ok": list(manifest["arms"]), "failures": manifest["failures"]}, indent=1))


if __name__ == "__main__":
    main()
