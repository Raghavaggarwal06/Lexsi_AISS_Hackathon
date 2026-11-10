"""Replicate paper Sec 4 (conditioned refusal: refuse harmful, comply with harmless).

stages
  vectors  refusal behaviour vector (100 Alpaca x 100 suffix pairs) + harmful condition vector (4,050 pairs), all layers
  search   grid search of the condition point on the TRAIN prompts, layers 1 .. L/2 - 1 (paper: first half)
  eval     test = 450 Sorry-Bench harmful + 500 Alpaca harmless; settings base / +refusal / +CAST (our grid point)
           / +CAST (paper's Table 3 point); refusal judged as in App D.2
Outputs in runs/<model>/sec4/.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cast import data, vectors, search, steer
from cast.model import LM
from cast.judge import RefusalJudge

# paper Table 3 (App C.1): condition (layer, cmp, theta), behaviour (layers, alpha)
PAPER = {
    "qwen1.5-1.8b-chat": {"cond": (8, ">", 0.031), "beh": (list(range(10, 21)), 4.0),
                          "ref": {"base": (45.78, 0.00), "refusal": (100.00, 96.40), "cast": (90.67, 2.20)}},
    "hermes-2-pro-llama-3-8b": {"cond": (7, "<", 0.048), "beh": ([15] + list(range(17, 25)), 1.7),
                                "ref": {"base": (19.33, 1.00), "refusal": (98.00, 91.80), "cast": (83.33, 2.40)}},
}


def log(msg, f):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(f, "a") as fh:
        fh.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen1.5-1.8b-chat")
    ap.add_argument("--stages", default="vectors,search,eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--settings", default="base,refusal,cast,cast_paper")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N test prompts of each class")
    ap.add_argument("--out", default=None)
    ap.add_argument("--masked-pooling", action="store_true",
                    help="variant: masked per-example pooling instead of the release's padded pooling "
                         "(outputs in sec4_masked/); it adds a 1/length artefact to the vectors, see README")
    a = ap.parse_args()
    out = Path(a.out or Path(__file__).resolve().parents[1] / "runs" / a.model / ("sec4_masked" if a.masked_pooling else "sec4"))
    out.mkdir(parents=True, exist_ok=True)
    L = out / "log.txt"
    stages = a.stages.split(",")
    lm = LM(a.model, device=a.device)
    nL = lm.n_layers
    log(f"model {a.model}: {nL} layers, pad id {lm.pad_id}, stages {stages}", L)

    if "vectors" in stages:
        t = time.time()
        qs, pairs = data.behavior_refusal()
        v, ev, sep = vectors.behavior_vector(lm, qs, pairs, list(range(nL)), a.batch,
                                               release=not a.masked_pooling)
        vectors.save(out / "behavior_refusal.npz", v, ev, sep, {"n_pairs": len(qs) * len(pairs)})
        log(f"behaviour vector: {len(qs)}x{len(pairs)} pairs, {time.time()-t:.0f}s; pair-sep by layer "
            + " ".join(f"{l}:{sep[l]:.2f}" for l in range(nL)), L)
        t = time.time()
        hp, hn = data.condition_harmful()
        v, ev, sep = vectors.condition_vector(lm, hp, hn, list(range(nL)), "all", a.batch,
                                                release=not a.masked_pooling)
        vectors.save(out / "condition_harmful.npz", v, ev, sep, {"n_pairs": len(hp)})
        log(f"harmful condition vector: {len(hp)} pairs, {time.time()-t:.0f}s; EV by layer "
            + " ".join(f"{l}:{ev[l]:.2f}" for l in range(nL)), L)

    if "search" in stages:
        t = time.time()
        cv = vectors.load(out / "condition_harmful.npz")
        hp, hn = data.condition_harmful()
        layers = list(range(1, nL // 2))
        sp = steer.condition_sims(lm, hp, cv, layers, batch_size=a.batch)
        sn = steer.condition_sims(lm, hn, cv, layers, batch_size=a.batch)
        np.savez(out / "train_sims.npz", layers=np.array(layers), pos=np.stack([sp[l] for l in layers]),
                 neg=np.stack([sn[l] for l in layers]))
        best = search.best_condition_point(sp, sn, layers)
        (out / "condition_point.json").write_text(json.dumps(best, indent=1))
        log(f"grid search ({time.time()-t:.0f}s): layer {best['layer']} sim {best['cmp']} {best['threshold']:.4f} "
            f"F1 {best['f1']:.3f}; per-layer best F1 "
            + " ".join(f"{l}:{max(best['per_layer'][l]['<'][0], best['per_layer'][l]['>'][0]):.3f}" for l in layers), L)

    if "eval" in stages:
        bv = vectors.load(out / "behavior_refusal.npz")
        cv = vectors.load(out / "condition_harmful.npz")
        harmful, harmless = data.sec4_test()
        if a.limit:
            harmful, harmless = harmful[:a.limit], harmless[:a.limit]
        prompts = harmful + harmless
        y = np.array([1] * len(harmful) + [0] * len(harmless))
        P = PAPER.get(a.model)
        blayers, alpha = P["beh"] if P else (list(range(nL // 3, 2 * nL // 3)), 1.0)
        beh = steer.Behavior(bv, blayers, alpha)
        cp = json.loads((out / "condition_point.json").read_text())
        configs = {
            "base": None,
            "refusal": steer.Steerer(lm, behaviors=[beh]),
            "cast": steer.Steerer(lm, [steer.Condition(cv, [cp["layer"]], cp["threshold"], cp["cmp"])], [beh]),
        }
        if P:
            l, c, th = P["cond"]
            configs["cast_paper"] = steer.Steerer(lm, [steer.Condition(cv, [l], th, c)], [beh])
        judge = RefusalJudge(a.device)
        summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
        for name in a.settings.split(","):
            if name not in configs:
                continue
            t = time.time()
            s = configs[name]
            if s is None:
                resp, info = steer.generate(lm, prompts, None, batch_size=a.batch)
            else:
                with s:
                    resp, info = steer.generate(lm, prompts, s, batch_size=a.batch)
            j = judge(resp)
            ref = np.array([x["refusal"] for x in j])
            clf = np.array([x["clf"] == "REJECTION" for x in j])
            met = np.array([i["met"][0] if i and i["met"] else np.nan for i in info], dtype=float)
            r = {"harmful_refusal": 100 * ref[y == 1].mean(), "harmless_refusal": 100 * ref[y == 0].mean(),
                 "harmful_refusal_clf_only": 100 * clf[y == 1].mean(), "harmless_refusal_clf_only": 100 * clf[y == 0].mean(),
                 "n_harmful": int((y == 1).sum()), "n_harmless": int((y == 0).sum()), "seconds": round(time.time() - t)}
            r["discrepancy"] = r["harmful_refusal"] - r["harmless_refusal"]
            if not np.isnan(met).all():
                r["condition_met_harmful"] = 100 * np.nanmean(met[y == 1])
                r["condition_met_harmless"] = 100 * np.nanmean(met[y == 0])
            if name.startswith("cast"):
                c0 = configs[name].conditions[0]
                r["condition"] = {"layer": c0.layers, "cmp": c0.cmp, "threshold": c0.threshold}
            r["behavior"] = {"layers": blayers, "alpha": alpha} if name != "base" else None
            summary[name] = r
            with open(out / f"responses_{name}.jsonl", "w") as fh:
                for p, yy, rr, jj, ii in zip(prompts, y, resp, j, info):
                    fh.write(json.dumps({"prompt": p, "harmful": int(yy), "response": rr, **jj, "cond": ii}) + "\n")
            log(f"{name}: harmful {r['harmful_refusal']:.1f}% harmless {r['harmless_refusal']:.1f}% "
                f"(clf only {r['harmful_refusal_clf_only']:.1f}/{r['harmless_refusal_clf_only']:.1f})"
                + (f" cond met {r['condition_met_harmful']:.1f}/{r['condition_met_harmless']:.1f}" if 'condition_met_harmful' in r else "")
                + f" [{r['seconds']}s]", L)
        if P:
            summary["paper"] = {k: {"harmful_refusal": v[0], "harmless_refusal": v[1]} for k, v in P["ref"].items()}
        (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
