"""Build the /case directory for task cast_compile (see docs/cast_compile_plan.md).

/case/weights/<param name>.npy   every Qwen1.5-1.8B-Chat tensor, fp16
/case/config.json                model config
/case/tokenizer.json             for decoding (the sandbox has no tokenizers library; dev prompts are pre-tokenized)
/case/cast/cast.npz              condition vector c (layer Lc), behaviour vectors v_l (l in behaviour layers)
/case/cast/cast.json             threshold, comparison, alpha, layers, conventions, CAST algorithm in words
/case/dev/dev.jsonl              400 TRAIN-split prompts: text, token ids (chat-formatted), harmful label,
                                 CAST gate similarity and decision, the CAST teacher's response
Run with the mech-interp env (torch); reads cast-safety runs/<model>/sec4.
"""
import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
import torch

CS = Path(__file__).resolve().parents[3] / "cast-safety"
sys.path.insert(0, str(CS))
from cast import data, vectors, steer          # noqa: E402
from cast.model import LM, MODELS              # noqa: E402

CASES = Path("/scratch/scholar/pande1/wsp/cases")
PAPER_BEH = {"qwen1.5-1.8b-chat": (list(range(10, 21)), 4.0)}

ALGO = """CAST (Conditional Activation Steering, Lee et al. 2025) as run by the teacher:
1. Format the user prompt with the chat template (system + user + assistant header); token ids are given.
2. Prompt pass. At the INPUT of decoder layer Lc (= the residual stream after layer Lc-1; hidden_states[Lc] in
   HuggingFace terms, hidden_states[0] = embeddings), take h = mean over all prompt positions of the residual
   stream (raw, not normalised).
3. sim = cos(h, tanh(P h)) with P = c c^T / (c . c)  (tanh applied elementwise to the projected vector).
4. The gate fires iff sim <comparison> threshold. The decision is made once, on the prompt.
5. If it fires: for every layer l in behaviour_layers, add alpha * v_l to the residual stream at the INPUT of
   decoder layer l, at every position (prompt and every generated token).
   If it does not fire: the model runs unchanged.
6. Decoding: greedy, repetition penalty 1.1, 50 new tokens.
Goal of the task: make the PLAIN model (no hooks) behave like this, by editing weights."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen1.5-1.8b-chat")
    ap.add_argument("--n-dev", type=int, default=200)
    a = ap.parse_args()
    case = CASES / f"cast_compile_{a.model}"
    (case / "weights").mkdir(parents=True, exist_ok=True)
    (case / "cast").mkdir(exist_ok=True); (case / "dev").mkdir(exist_ok=True)
    lm = LM(a.model, dtype=torch.float16)
    src = MODELS / a.model
    shutil.copy(src / "config.json", case / "config.json")
    shutil.copy(src / "tokenizer.json", case / "tokenizer.json")
    names = []
    for n, p in lm.model.state_dict().items():
        np.save(case / "weights" / f"{n}.npy", p.detach().cpu().numpy().astype(np.float16))
        names.append([n, list(p.shape)])
    (case / "weights" / "index.json").write_text(json.dumps(names, indent=0))

    run = CS / "runs" / a.model / "sec4"
    bv, cv = vectors.load(run / "behavior_refusal.npz"), vectors.load(run / "condition_harmful.npz")
    cp = json.loads((run / "condition_point.json").read_text())
    blayers, alpha = PAPER_BEH[a.model]
    Lc = cp["layer"]
    np.savez(case / "cast" / "cast.npz", c=cv[Lc].astype(np.float32),
             **{f"v_{l}": bv[l].astype(np.float32) for l in blayers})
    meta = {"condition_layer": Lc, "threshold": cp["threshold"], "comparison": cp["cmp"],
            "behaviour_layers": blayers, "alpha": alpha, "grid_search_f1_train": cp["f1"],
            "vectors": "c and v_l are unit-norm float32 (first principal components)",
            "layer_convention": "vectors are added/checked at the INPUT of decoder layer l (residual stream entering it)",
            "algorithm": ALGO}
    (case / "cast" / "cast.json").write_text(json.dumps(meta, indent=1))

    hp, hn = data.condition_harmful()
    hp, hn = list(dict.fromkeys(hp))[-a.n_dev:], list(dict.fromkeys(hn))[-a.n_dev:]
    prompts, y = hp + hn, [1] * len(hp) + [0] * len(hn)
    cond = steer.Condition(cv, [Lc], cp["threshold"], cp["cmp"])
    beh = steer.Behavior(bv, blayers, alpha)
    s = steer.Steerer(lm, [cond], [beh])
    with s:
        resp, info = steer.generate(lm, prompts, s, batch_size=32)
    with open(case / "dev" / "dev.jsonl", "w") as fh:
        for p, yy, r, i in zip(prompts, y, resp, info):
            ids = lm.tok(lm.chat(p, True), add_special_tokens=lm.repo_bos)["input_ids"]
            fh.write(json.dumps({"prompt": p, "ids": ids, "harmful": yy, "gate_sim": i["sims"][f"c0_L{Lc}"],
                                 "gate_fired": i["met"][0], "cast_response": r}) + "\n")
    met = np.array([i["met"][0] for i in info]); yy = np.array(y)
    print(f"case at {case}: {len(names)} tensors; dev {len(prompts)}; gate fired {met[yy==1].mean():.2f} / {met[yy==0].mean():.2f}")


if __name__ == "__main__":
    main()
