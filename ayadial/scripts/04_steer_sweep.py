"""Phase 2: generate with a forward hook that pushes the residual along the refusal direction.

Methods:
  ungated      push s * r at every position (the control)
  prop         push proportional to each position's own harm score (candidate A, per token)
  prop_prompt  push proportional to one score read at the last prompt token, on every position
  gate         if the score at the last prompt token clears tau, push s * r on every position

The per-prompt methods touch the same positions as ungated once they fire, so the only difference
from the control is which prompts get pushed and by how much. s=0 is the base model and runs once,
as runs/base/s=0.jsonl. Finished files are skipped, so a crashed run can simply be restarted.

--layer is the hidden-state index from phase 1, so the hook goes on model.model.layers[layer - 1].
--n caps at the size of each eval half (146 harmful, 125 for the other sets).
"""
import argparse
import json
import time

import numpy as np
import torch
from tqdm import tqdm

from common import DAY1, RESULTS, ROOT, encode, load_config, load_model, load_set, load_splits, load_tokenizer, pick_device, render, read_jsonl, write_jsonl

METHODS = ["ungated", "prop", "prop_prompt", "gate"]
PER_PROMPT = {"prop_prompt", "gate"}
SWEEP_SETS = ["harmful", "xstest_safe", "orbench_hard", "benign"]


def load_steering(layer, variant, model):
    saved = np.load(DAY1 / "directions.npz")
    param = next(model.parameters())
    tensor = lambda name: torch.tensor(saved[name][layer], device=param.device, dtype=param.dtype)
    return {
        "d": tensor(f"d_{variant}"),
        "r": tensor("r_raw"),
        "tau": float(saved[f"tau_{variant}"][layer]),
        "benign": float(saved[f"benign_mean_proj_{variant}"][layer]),
        "harmful": float(saved[f"harmful_mean_proj_{variant}"][layer]),
    }


def make_hook(method, scale, st):
    """Returns (forward hook for one decoder layer, per-prompt decision state or None)."""

    def amount(h):
        proj = h @ st["d"]
        if method == "ungated":
            return torch.ones_like(proj)
        if method in ("prop", "prop_prompt"):
            return ((proj - st["benign"]) / (st["harmful"] - st["benign"])).clamp(min=0)
        return (proj > st["tau"]).to(proj.dtype)

    decision = {"amount": None} if method in PER_PROMPT else None

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if decision is None:
            weight = amount(h).unsqueeze(-1)
        else:
            if decision["amount"] is None:
                assert h.shape[1] > 1, "per-prompt decision must be read on the full prompt pass"
                decision["amount"] = amount(h[:, -1, :])  # left padding: -1 is the last prompt token
            weight = decision["amount"].view(-1, 1, 1)
        pushed = h + scale * weight * st["r"]
        return (pushed, *output[1:]) if isinstance(output, tuple) else pushed

    return hook, decision


def eval_rows(n):
    splits, rows = load_splits(), []
    for name in SWEEP_SETS:
        table = {r["id"]: r for r in load_set(name)}
        rows += [table[i] for i in splits[name]["eval"][:n]]
    return rows


@torch.no_grad()
def generate_all(model, tok, rows, batch, max_len, max_new, device, decision=None):
    """Greedy generation. Returns the responses and, for per-prompt methods, who was pushed."""
    out, fired, start = [], [], 0
    bar = tqdm(total=len(rows), mininterval=20)
    while start < len(rows):
        chunk = rows[start : start + batch]
        enc = encode(tok, [render(tok, r["prompt"]) for r in chunk], max_len, device)
        if decision is not None:
            decision["amount"] = None
        try:
            ids = model.generate(**enc, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        except RuntimeError as err:
            if "out of memory" not in str(err).lower() or batch == 1:
                raise
            batch = max(1, batch // 2)
            print(f"OOM, batch lowered to {batch}")
            torch.mps.empty_cache() if device == "mps" else torch.cuda.empty_cache() if device == "cuda" else None
            continue
        new = ids[:, enc["input_ids"].shape[1]:]
        out += tok.batch_decode(new, skip_special_tokens=True)
        if decision is not None:
            fired += (decision["amount"] > 0).tolist()
        start += len(chunk)
        bar.update(len(chunk))
    bar.close()
    return out, fired


def log_gate_firing(scale, rows, fired):
    path = RESULTS / "phase2" / "gate_firing.json"
    log = json.loads(path.read_text()) if path.exists() else {}
    per_set = {}
    for name in SWEEP_SETS:
        flags = [f for r, f in zip(rows, fired) if r["set"] == name]
        per_set[name] = {"fired": int(sum(flags)), "n": len(flags), "fraction": sum(flags) / len(flags)}
    log[f"{scale:g}"] = per_set
    path.write_text(json.dumps(log, indent=1))


def run_one(model, tok, rows, method, scale, st, layer, args, device, max_len):
    path = ROOT / "runs" / method / f"s={scale:g}.jsonl"
    if path.exists() and len(read_jsonl(path)) == len(rows):
        print(f"{path.relative_to(ROOT)}: already done")
        return 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, decision = None, None
    if method != "base":
        hook, decision = make_hook(method, scale, st)
        handle = model.model.layers[layer - 1].register_forward_hook(hook)
    started = time.time()
    try:
        responses, fired = generate_all(model, tok, rows, args.batch, max_len, args.max_new, device, decision)
    finally:
        if handle:
            handle.remove()
    write_jsonl(path, [{"id": r["id"], "set": r["set"], "prompt": r["prompt"], "response": t} for r, t in zip(rows, responses)])
    if method == "gate":
        log_gate_firing(scale, rows, fired)
    elapsed = time.time() - started
    print(f"{path.relative_to(ROOT)}: {len(rows)} responses in {elapsed / 60:.1f} min")
    return elapsed


@torch.no_grad()
def mlp_histogram(model, tok, layer, batch, max_len, device):
    """Per-unit activation statistics on the benign fit half, for finding dead units in phase 3."""
    out_path = RESULTS / "phase2" / f"mlp_units_layer{layer}.npz"
    if out_path.exists():
        return
    edges = np.r_[-np.inf, -np.logspace(1, -3, 9), 0.0, np.logspace(-3, 1, 9), np.inf]
    captured = {}
    grab = lambda _m, inputs: captured.__setitem__("units", inputs[0])
    handle = model.model.layers[layer - 1].mlp.down_proj.register_forward_pre_hook(grab)
    splits = load_splits()
    table = {r["id"]: r for r in load_set("benign")}
    texts = [render(tok, table[i]["prompt"]) for i in splits["benign"]["fit"]]
    counts, peak, tokens = None, None, 0
    for start in tqdm(range(0, len(texts), batch), mininterval=20):
        enc = encode(tok, texts[start : start + batch], max_len, device)
        model.model(**enc, use_cache=False)
        units = captured["units"][enc["attention_mask"].bool()].float().cpu().numpy()
        tokens += len(units)
        bins = np.searchsorted(edges, units, side="right") - 1
        if counts is None:
            counts, peak = np.zeros((units.shape[1], len(edges) - 1), dtype=np.int64), np.zeros(units.shape[1])
        for b in range(counts.shape[1]):
            counts[:, b] += (bins == b).sum(0)
        peak = np.maximum(peak, np.abs(units).max(0))
    handle.remove()
    np.savez(out_path, bin_edges=edges, counts=counts, max_abs=peak, tokens=tokens)
    print(f"{out_path.relative_to(ROOT)}: {counts.shape[0]} units, {tokens} tokens")


def main():
    cfg = load_config()
    saved = np.load(DAY1 / "directions.npz")
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS + ["all"], default="all")
    ap.add_argument("--scales", default="0,0.25,0.5,0.75,1,1.5,2")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--layer", type=int, default=int(saved["selected_layer"]))
    ap.add_argument("--variant", choices=["raw", "cleaned"], default="cleaned")
    ap.add_argument("--batch", type=int, default=cfg["batch"])
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = load_tokenizer(cfg["model"])
    model, _ = load_model(cfg["model"], device)
    st = load_steering(args.layer, args.variant, model)
    rows = eval_rows(args.n)
    per_set = {name: sum(r["set"] == name for r in rows) for name in SWEEP_SETS}
    print(f"prompts per set: {per_set}")
    scales = [float(x) for x in args.scales.split(",")]
    methods = METHODS if args.method == "all" else [args.method]
    (RESULTS / "phase2").mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "phase2" / "sweep_config.json", "w") as f:
        json.dump({**vars(args), "scales": scales, "methods": methods, "n_rows": len(rows), "n_per_set": per_set}, f, indent=2)

    mlp_histogram(model, tok, args.layer, args.batch, cfg["max_len"], device)
    timings = {}
    if 0.0 in scales:
        timings["base/s=0"] = run_one(model, tok, rows, "base", 0.0, st, args.layer, args, device, cfg["max_len"])
    for method in methods:
        for scale in (s for s in scales if s != 0.0):
            timings[f"{method}/s={scale:g}"] = run_one(model, tok, rows, method, scale, st, args.layer, args, device, cfg["max_len"])
    print("Done.", {k: round(v / 60, 1) for k, v in timings.items()})


if __name__ == "__main__":
    main()
