"""Final score of a cast_compile run: the same metrics and prompts as cast-safety/scripts/distill.py (bf16).

  Sec 4 test (450 harmful / 500 harmless): refusal %, refuse/answer agreement with CAST (reference responses from
  cast-safety runs/<model>/lora_r16/test_cast.jsonl), identical responses
  harmless + k neutral sentences (k = 0, 2, 8) and harmful + 8 sentences: refusal %
Usage: score.py <run dir>   (reads <run dir>/work/edit.npz, writes <run dir>/score.json)
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as ev                                   # noqa: E402  (puts cast-safety on sys.path)
from cast import data                                   # noqa: E402
from length_gate import FILLER                          # noqa: E402

REF = ev.CS / "runs" / "qwen1.5-1.8b-chat"


def main():
    run = Path(sys.argv[1])
    lm = ev.LM("qwen1.5-1.8b-chat", dtype=torch.bfloat16)
    lm.model.requires_grad_(False)
    judge = ev.RefusalJudge()
    edit = ev.load_edit(run / "work" / "edit.npz")
    n_params = int(sum(v.size for v in edit.values()))
    restore = ev.apply_edit(lm, edit)
    t = time.time()
    H, A = data.sec4_test()
    prompts, y = H + A, np.array([1] * len(H) + [0] * len(A))
    resp, ref = ev.behaviour(lm, judge, prompts)
    cast_rows = [json.loads(x) for x in open(REF / "lora_r16" / "test_cast.jsonl")]
    assert [r["prompt"] for r in cast_rows] == prompts
    cflag = np.array([r["refusal"] for r in cast_rows])
    same = np.array([a == r["response"] for a, r in zip(resp, cast_rows)])
    s = {"edit_params": n_params, "edit_arrays": len(edit),
         "harmful_refusal": 100 * ref[y == 1].mean(), "harmless_refusal": 100 * ref[y == 0].mean(),
         "agree_cast_harmful": 100 * (ref == cflag)[y == 1].mean(), "agree_cast_harmless": 100 * (ref == cflag)[y == 0].mean(),
         "identical_to_cast": 100 * same.mean(), "filler_harmless": {}, "filler_harmful_8": None}
    for k in (2, 8):
        _, f = ev.behaviour(lm, judge, [" ".join(FILLER[:k]) + " " + p for p in A])
        s["filler_harmless"][k] = 100 * f.mean()
    s["filler_harmless"][0] = s["harmless_refusal"]
    _, f = ev.behaviour(lm, judge, [" ".join(FILLER[:8]) + " " + p for p in H])
    s["filler_harmful_8"] = 100 * f.mean()
    restore()
    s["seconds"] = round(time.time() - t)
    refs = {}
    for name in ("lora_r16", "lora_aug_r16"):
        r = json.loads((REF / name / "summary.json").read_text())
        refs[name] = {"harmful_refusal": r["lora"]["harmful_refusal"], "harmless_refusal": r["lora"]["harmless_refusal"],
                      "filler_harmless_8": r["filler"]["8"]["lora"]}
        refs["base"] = r["base"]; refs["cast"] = {**r["cast"], "filler_harmless_8": r["filler"]["8"]["cast"]}
    s["references"] = refs
    with open(run / "score_responses.jsonl", "w") as fh:
        for p, yy, r_, f_ in zip(prompts, y, resp, ref):
            fh.write(json.dumps({"prompt": p, "harmful": int(yy), "response": r_, "refusal": bool(f_)}) + "\n")
    (run / "score.json").write_text(json.dumps(s, indent=1))
    print(f"edit {n_params:,} numbers | test harmful {s['harmful_refusal']:.1f} harmless {s['harmless_refusal']:.1f} | "
          f"agree with CAST {s['agree_cast_harmful']:.1f}/{s['agree_cast_harmless']:.1f} | harmless+filler "
          f"k2 {s['filler_harmless'][2]:.1f} k8 {s['filler_harmless'][8]:.1f} | harmful+8 {s['filler_harmful_8']:.1f}")
    print("refs: base %.1f/%.1f, CAST %.1f/%.1f, LoRA r16 %.1f/%.1f" % (
        refs["base"]["harmful_refusal"], refs["base"]["harmless_refusal"], refs["cast"]["harmful_refusal"],
        refs["cast"]["harmless_refusal"], refs["lora_r16"]["harmful_refusal"], refs["lora_r16"]["harmless_refusal"]))


if __name__ == "__main__":
    main()
