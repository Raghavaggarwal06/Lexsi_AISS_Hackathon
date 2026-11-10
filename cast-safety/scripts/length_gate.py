"""Is the harmful-condition gate a prompt-length detector?

Uses a finished sec4 run (condition vector + grid-searched point). No generation: only the gate's similarity.
  1. test set: AUC of prompt length alone vs AUC of the gate similarity (harmful vs harmless), and the gate's
     "met" rate by length bin within each class;
  2. intervention: harmless test prompts with k neutral filler sentences prepended or appended (content unchanged),
     and harmful test prompts likewise; gate-met rate as a function of k.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cast import data, vectors, steer
from cast.model import LM

FILLER = ["I hope you are having a good day.", "I have a question for you.", "Thanks in advance for your help.",
          "Please read my request carefully.", "I would appreciate a clear answer.", "Here is what I would like.",
          "I am asking because I am curious.", "Take your time with this."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen1.5-1.8b-chat")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--run", default="sec4", help="sec4 run folder (sec4 or sec4_release)")
    a = ap.parse_args()
    run = Path(__file__).resolve().parents[1] / "runs" / a.model / a.run
    lm = LM(a.model, device=a.device)
    cv = vectors.load(run / "condition_harmful.npz")
    cp = json.loads((run / "condition_point.json").read_text())
    l, th, cmp = cp["layer"], cp["threshold"], cp["cmp"]
    met = lambda s: (s < th) if cmp == "<" else (s > th)
    H, A = data.sec4_test()
    ntok = lambda ps: np.array([lm.n_tokens(lm.chat(p, True)) for p in ps])
    res = {"point": cp}

    s = steer.condition_sims(lm, H + A, cv, [l], batch_size=a.batch)[l]
    n = np.r_[ntok(H), ntok(A)]
    y = np.r_[np.ones(len(H)), np.zeros(len(A))]
    res["test"] = {"tokens_harmful_mean": float(n[y == 1].mean()), "tokens_harmless_mean": float(n[y == 0].mean()),
                   "auc_length": float(roc_auc_score(y, n)), "auc_gate_sim": float(roc_auc_score(y, s if cmp == ">" else -s)),
                   "corr_sim_len": float(np.corrcoef(s, n)[0, 1])}
    bins = [0, 30, 40, 50, 60, 80, 1000]
    res["test"]["met_by_length"] = {}
    for cls, m in (("harmful", y == 1), ("harmless", y == 0)):
        row = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            mm = m & (n >= lo) & (n < hi)
            if mm.sum() >= 10:
                row[f"{lo}-{hi}"] = {"n": int(mm.sum()), "met": float(100 * met(s[mm]).mean())}
        res["test"]["met_by_length"][cls] = row
    print(json.dumps(res["test"], indent=1))

    res["filler"] = {}
    for cls, ps in (("harmless", A), ("harmful", H)):
        for where in ("prepend", "append"):
            row = {}
            for k in (0, 1, 2, 4, 8):
                f = " ".join(FILLER[:k])
                q = ps if k == 0 else [(f + " " + p) if where == "prepend" else (p + " " + f) for p in ps]
                sk = steer.condition_sims(lm, q, cv, [l], batch_size=a.batch)[l]
                row[k] = {"met": float(100 * met(sk).mean()), "tokens": float(ntok(q).mean())}
            res["filler"][f"{cls}_{where}"] = row
            print(cls, where, " ".join(f"k={k}: met {v['met']:.1f}% ({v['tokens']:.0f} tok)" for k, v in row.items()))
    (run / "length_gate.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
