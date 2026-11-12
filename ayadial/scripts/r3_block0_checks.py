"""Round 3, block 0 checks that run on the Mac: config facts, the hidden-state index, the
template tail, and the 87.7% vs 87.0% reconciliation. Prints facts; writes results/round3/block0/checks.json.
"""
import json

import numpy as np
import torch
from huggingface_hub import HfApi
from transformers import AutoConfig

from common import DAY1, RESULTS, encode, load_config, load_model, load_set, load_splits, load_tokenizer, render

OUT = RESULTS / "round3" / "block0"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    cfg = load_config()
    out = {}
    conf = AutoConfig.from_pretrained(cfg["model"])
    keys = ("num_hidden_layers", "layer_types", "sliding_window", "sliding_window_pattern", "tie_word_embeddings", "use_parallel_block", "hidden_size", "intermediate_size")
    out["config"] = {k: getattr(conf, k, None) for k in keys}
    print("config:", out["config"])

    tok = load_tokenizer(cfg["model"])
    text = render(tok, "Name a colour.")
    ids = tok(text, add_special_tokens=False)["input_ids"]
    user_ids = tok("Name a colour.", add_special_tokens=False)["input_ids"]
    tail_start = max(i for i in range(len(ids) - len(user_ids) + 1) if ids[i : i + len(user_ids)] == user_ids) + len(user_ids)
    out["template"] = {"length": len(ids), "last_instruction_index_from_end": tail_start - len(ids) - 1, "tail_tokens": [tok.decode([t]) for t in ids[tail_start:]]}
    print("template tail:", out["template"])

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _ = load_model(cfg["model"], device)
    seen = {}
    h_out = model.model.layers[22].register_forward_hook(lambda m, i, o: seen.__setitem__("out22", (o[0] if isinstance(o, tuple) else o).float().cpu()))
    h_in = model.model.layers[23].register_forward_pre_hook(lambda m, a, kw: seen.__setitem__("in23", (a[0] if a else kw["hidden_states"]).float().cpu()), with_kwargs=True)
    with torch.no_grad():
        enc = encode(tok, [text], cfg["max_len"], device)
        hs = model.model(**enc, output_hidden_states=True).hidden_states
    h_out.remove()
    h_in.remove()
    idx23 = hs[23].float().cpu()
    out["index23"] = {
        "n_hidden_states": len(hs),
        "max_abs_diff_vs_layers22_output": float((idx23 - seen["out22"]).abs().max()),
        "max_abs_diff_vs_layers23_input": float((idx23 - seen["in23"]).abs().max()),
        "consumer_module": repr(model.model.layers[23]).split("\n")[0] + " -> input_layernorm feeds self_attn and mlp in parallel",
        "layers23": repr(model.model.layers[23]),
        "layer_type_23": (conf.layer_types[23] if getattr(conf, "layer_types", None) else None),
    }
    print("index 23:", {k: v for k, v in out["index23"].items() if k != "layers23"})
    print(out["index23"]["layers23"])

    saved = np.load(DAY1 / "directions.npz")
    d, tau = saved["d_cleaned"][23].astype(np.float32), float(saved["tau_cleaned"][23])
    rows = load_set("harmful")
    where = {r["id"]: i for i, r in enumerate(rows)}
    ids_eval = load_splits()["harmful"]["eval"]
    acts = np.load(DAY1 / "acts_harmful.npy")[[where[i] for i in ids_eval], 23].astype(np.float32)
    phase1 = acts @ d
    order = np.argsort(np.abs(phase1 - tau))[:5]
    live = []
    with torch.no_grad():
        dt = torch.tensor(d, device=device)
        for i in order:
            enc = encode(tok, [render(tok, rows[where[ids_eval[i]]]["prompt"])], cfg["max_len"], device)
            h = model.model(**enc, output_hidden_states=True).hidden_states[23][0, -1]
            live.append({"id": ids_eval[i], "phase1_score_fp32_over_fp16_cache": float(phase1[i]), "live_bf16_dot": float(h @ dt.to(h.dtype)), "live_fp32_dot": float(h.float() @ dt), "tau": tau})
    out["reconcile"] = {"phase1_fired": int((phase1 > tau).sum()), "n": len(phase1), "closest_to_tau": live}
    print("reconcile:", out["reconcile"])

    api = HfApi()
    access = {}
    for repo, kind in (("allenai/wildjailbreak", "dataset"), ("allenai/wildguard", "model")):
        try:
            api.auth_check(repo, repo_type=kind)
            access[repo] = "ok"
        except Exception as err:
            access[repo] = type(err).__name__
    out["access"] = access
    print("access:", access)
    (OUT / "checks.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
