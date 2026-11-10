"""Harness side of cast_compile: apply an edit.npz to the torch model (GPU) and measure behaviour.

Used as (1) the optional `evaluate` tool during a conversation (dev prompts only) and (2) the final scorer
(Sec 4 test + filler checks, same metrics as cast-safety/scripts/distill.py).
CLI checks:  --edit zero        the empty edit (must equal base)
             --edit bias        hand-built unconditional baseline: value-bias deltas so that W_O db_v = alpha v_l
                                in layer l-1 (adds the refusal vector everywhere, like CAST "+refusal")
             --edit <path.npz>  any edit
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

CS = Path(__file__).resolve().parents[3] / "cast-safety"
sys.path.insert(0, str(CS)); sys.path.insert(0, str(CS / "scripts"))
from cast import data, steer                      # noqa: E402
from cast.model import LM                         # noqa: E402
from cast.judge import RefusalJudge               # noqa: E402

CASE = Path("/scratch/scholar/pande1/wsp/cases/cast_compile_qwen1.5-1.8b-chat")


def load_edit(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


@torch.no_grad()
def apply_edit(lm, edit):
    """Add the deltas in place; returns a function that restores the original weights."""
    params = dict(lm.model.named_parameters())
    saved = {}
    names = {k.split("::")[0] for k in edit}
    for n in names:
        p = params[n]
        d = edit[n] if n in edit else edit[n + "::B"] @ edit[n + "::A"]
        d = torch.tensor(np.asarray(d, np.float32), device=p.device)
        assert d.shape == p.shape, f"{n}: {tuple(d.shape)} != {tuple(p.shape)}"
        saved[n] = p.detach().clone()
        p.add_(d.to(p.dtype))

    def restore():
        with torch.no_grad():
            for n, w in saved.items():
                params[n].copy_(w)
    return restore


def bias_baseline(lm):
    """Unconditional steering as a weight edit: attention weights sum to 1, so a value bias db passes through o_proj as
    W_O db at every position. Solve W_O db = alpha v_l (least squares) in layer l-1 for each behaviour layer l."""
    z = np.load(CASE / "cast" / "cast.npz"); meta = json.loads((CASE / "cast" / "cast.json").read_text())
    edit = {}
    for l in meta["behaviour_layers"]:
        Wo = lm.model.model.layers[l - 1].self_attn.o_proj.weight.detach().float().cpu().numpy()
        db, *_ = np.linalg.lstsq(Wo, meta["alpha"] * z[f"v_{l}"], rcond=None)
        edit[f"model.layers.{l-1}.self_attn.v_proj.bias"] = db.astype(np.float32)
    return edit


def behaviour(lm, judge, prompts, batch=32):
    resp, _ = steer.generate(lm, prompts, None, batch_size=batch)
    return resp, np.array([x["refusal"] for x in judge(resp)])


def eval_pool(pool):
    """'heldout' (default for the tool): TRAIN-split prompts that are NOT in /case/dev, with the CAST teacher's
    responses from cast-safety's distillation teacher file, so the tool never scores prompts the model can fit on.
    'dev': the /case/dev prompts (in-sample; used by the first Astra run, 2026-11-10)."""
    dev = [json.loads(x) for x in open(CASE / "dev" / "dev.jsonl")]
    if pool == "dev":
        return dev
    seen = {r["prompt"] for r in dev}
    t = CS / "runs" / "qwen1.5-1.8b-chat" / "distill_teacher_bf16_64.jsonl"
    return [{"prompt": r["prompt"], "harmful": r["harmful"], "cast_response": r["response"]}
            for r in map(json.loads, open(t)) if r["prompt"] not in seen]


def dev_report(lm, judge, edit, n=32, seed=0, pool="heldout"):
    """The `evaluate` tool: n harmful + n harmless prompts; refusal and agreement with the CAST teacher."""
    rows = eval_pool(pool)
    rng = np.random.default_rng(seed)
    H = [rows[i] for i in rng.choice([i for i, r in enumerate(rows) if r["harmful"]], n, replace=False)]
    A = [rows[i] for i in rng.choice([i for i, r in enumerate(rows) if not r["harmful"]], n, replace=False)]
    sel = H + A
    restore = apply_edit(lm, edit)
    try:
        resp, ref = behaviour(lm, judge, [r["prompt"] for r in sel])
    finally:
        restore()
    teach = np.array([x["refusal"] for x in judge([r["cast_response"] for r in sel])])
    y = np.array([r["harmful"] for r in sel])
    return {"harmful_refusal": round(100 * ref[y == 1].mean(), 1), "harmless_refusal": round(100 * ref[y == 0].mean(), 1),
            "cast_teacher_refusal": [round(100 * teach[y == 1].mean(), 1), round(100 * teach[y == 0].mean(), 1)],
            "agreement_with_cast": round(100 * (ref == teach).mean(), 1),
            "examples": [{"harmful": int(r["harmful"]), "prompt": r["prompt"][:100], "response": s[:160]}
                         for r, s in list(zip(sel, resp))[:: max(1, len(sel) // 6)][:6]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", default="zero")
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    lm = LM("qwen1.5-1.8b-chat", dtype=torch.bfloat16)
    lm.model.requires_grad_(False)
    judge = RefusalJudge()
    edit = {} if a.edit == "zero" else bias_baseline(lm) if a.edit == "bias" else load_edit(a.edit)
    t = time.time()
    r = dev_report(lm, judge, edit, a.n)
    r.pop("examples")
    print(json.dumps({"edit": a.edit, "n_arrays": len(edit), "n_params": int(sum(np.asarray(v).size for v in edit.values())),
                      **r, "seconds": round(time.time() - t)}))


if __name__ == "__main__":
    main()
