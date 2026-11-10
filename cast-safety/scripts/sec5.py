"""Replicate paper Sec 5 (programmed refusal: fine-grained category conditions and their logical composition).

data: condition_multiple.json, 700 train / 500 test Alpaca prompts ("base") + paraphrases into 5 categories.
stages
  vectors  one condition vector per category (base + 5): positives = the category's 700 train prompts repeated 5x,
           negatives = the other five categories' 700 each (release quickstart Part III); all layers
  search   condition point per category on the same train strings (positives repeated 5x, as in the release),
           layers 1 .. L/2 - 1
  eval     3,000 test prompts (500 x 6 categories). Generates base and unconditional +refusal responses once; a rule's
           response is the base response where the rule is off and the +refusal response where it is on. This is
           exact under greedy decoding when every condition layer of the rule is <= the lowest behaviour layer
           (the prompt pass is then steered at every behaviour layer); otherwise the rule's active prompts are
           regenerated with the steering hooks.
           rules: single categories (Fig 5a), OR compositions (Fig 6a), domain constraints by negation (Fig 7a)
Needs runs/<model>/sec4/behavior_refusal.npz (scripts/sec4.py --stages vectors).
Default pooling = the release's (padded, pairs batched together); --masked-pooling = our masked variant.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cast import data, vectors, search, steer
from cast.model import LM
from cast.judge import RefusalJudge
from sec4 import PAPER, log

CATS = ["base"] + data.CATEGORIES
SHORT = {"base": "base", "sexual_content": "sex", "legal_opinion": "lgl", "hate_speech": "hat",
         "crime_planning": "crm", "health_consultation": "hth"}
# paper Table 3 condition points (layer, cmp, theta) for Sec 5 categories
PAPER5 = {
    "qwen1.5-1.8b-chat": {"sexual_content": (7, "<", .033), "legal_opinion": (6, ">", .035), "hate_speech": (11, ">", .018),
                          "crime_planning": (7, "<", .024), "health_consultation": (8, ">", .01)},
    "hermes-2-pro-llama-3-8b": {"sexual_content": (7, ">", .037), "legal_opinion": (4, "<", .021),
                                "hate_speech": (7, ">", .029), "crime_planning": (3, "<", .024),
                                "health_consultation": (4, ">", .014)},
}
# (name, [categories], op, negate)
RULES = ([(f"if {SHORT[c]}", [c], "or", False) for c in data.CATEGORIES]
         + [("if hth|lgl", ["health_consultation", "legal_opinion"], "or", False),
            ("if hat|lgl", ["hate_speech", "legal_opinion"], "or", False),
            ("if hat|lgl|sex", ["hate_speech", "legal_opinion", "sexual_content"], "or", False),
            ("if hth|crm", ["health_consultation", "crime_planning"], "or", False)]
         + [(f"if not {SHORT[c]}", [c], "or", True) for c in data.CATEGORIES])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen1.5-1.8b-chat")
    ap.add_argument("--stages", default="vectors,search,eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--masked-pooling", action="store_true",
                    help="variant: masked per-example pooling (outputs in sec5_masked/, behaviour vector from "
                         "sec4_masked/); default = the release's padded pooling with pairs batched as in the release")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1] / "runs" / a.model
    out = root / ("sec5_masked" if a.masked_pooling else "sec5")
    sec4dir = root / ("sec4_masked" if a.masked_pooling else "sec4")
    out.mkdir(parents=True, exist_ok=True)
    L = out / "log.txt"
    stages = a.stages.split(",")
    lm = LM(a.model, device=a.device)
    nL = lm.n_layers
    tr = data.multiple("train")
    log(f"model {a.model}: {nL} layers; stages {stages}", L)

    if "vectors" in stages and not a.masked_pooling:
        t = time.time()
        for k, c in enumerate(CATS):
            others = [x for x in CATS if x != c]
            pos = [p for _ in others for p in tr[c]]
            neg = [p for o in others for p in tr[o]]
            vecs, ev, sep = vectors.condition_vector(lm, pos, neg, list(range(nL)), "all", a.batch, release=True)
            vectors.save(out / f"condition_{c}.npz", vecs, ev, sep)
            log(f"condition {c} (release pooling): pair-sep " + " ".join(f"{l}:{sep[l]:.2f}" for l in range(0, nL, 2))
                + " | EV " + " ".join(f"{l}:{ev[l]:.2f}" for l in range(0, nL, 4)), L)
        log(f"vectors done {time.time()-t:.0f}s", L)
    elif "vectors" in stages:
        t = time.time()
        allp = [p for c in CATS for p in tr[c]]
        H = vectors.pooled_hiddens(lm, [lm.chat(p, gen_prompt=False) for p in allp], list(range(nL)), "all",
                                   batch_size=a.batch)
        n = len(tr["base"])
        for k, c in enumerate(CATS):
            others = [j for j in range(len(CATS)) if j != k]
            vecs, sep = {}, {}
            for l in range(nL):
                hp = np.concatenate([H[l][k * n:(k + 1) * n]] * len(others))
                hn = np.concatenate([H[l][j * n:(j + 1) * n] for j in others])
                vecs[l], _ = vectors.pca_pairwise(hp, hn)
                sep[l] = float(np.mean(hp.astype(np.float32) @ vecs[l] > hn.astype(np.float32) @ vecs[l]))
            vectors.save(out / f"condition_{c}.npz", vecs, None, sep)
            log(f"condition {c}: pair-sep " + " ".join(f"{l}:{sep[l]:.2f}" for l in range(0, nL, 2)), L)
        log(f"vectors done {time.time()-t:.0f}s", L)

    if "search" in stages:
        t = time.time()
        layers = list(range(1, nL // 2))
        allp = [p for c in CATS for p in tr[c]]
        cvs = {c: vectors.load(out / f"condition_{c}.npz") for c in CATS}
        S = steer.condition_sims_multi(lm, allp, cvs, layers, batch_size=a.batch)
        n = len(tr["base"])
        points = {}
        for k, c in enumerate(CATS):
            others = [j for j in range(len(CATS)) if j != k]
            sp = {l: np.tile(S[c][l][k * n:(k + 1) * n], len(others)) for l in layers}
            sn = {l: np.concatenate([S[c][l][j * n:(j + 1) * n] for j in others]) for l in layers}
            best = search.best_condition_point(sp, sn, layers)
            points[c] = best
            log(f"{c}: layer {best['layer']} sim {best['cmp']} {best['threshold']:.4f} F1 {best['f1']:.3f}", L)
        (out / "condition_points.json").write_text(json.dumps(points, indent=1))
        log(f"search done {time.time()-t:.0f}s", L)

    if "eval" in stages:
        te = data.multiple("test")
        if a.limit:
            te = {c: v[:a.limit] for c, v in te.items()}
        prompts = [p for c in CATS for p in te[c]]
        cat = np.array([c for c in CATS for _ in te[c]])
        pts = json.loads((out / "condition_points.json").read_text())
        cvs = {c: vectors.load(out / f"condition_{c}.npz") for c in CATS}
        bv = vectors.load(sec4dir / "behavior_refusal.npz")
        P = PAPER.get(a.model)
        blayers, alpha = P["beh"] if P else (list(range(nL // 3, 2 * nL // 3)), 1.0)
        beh = steer.Behavior(bv, blayers, alpha)
        judge = RefusalJudge(a.device)
        cache = out / "cache.json"
        C = json.loads(cache.read_text()) if cache.exists() else {}

        def run(key, steerer, idx):
            if key in C and len(C[key]["resp"]) == len(idx):
                return C[key]
            sub = [prompts[i] for i in idx]
            t = time.time()
            if steerer is None:
                r, _ = steer.generate(lm, sub, None, batch_size=a.batch)
            else:
                with steerer:
                    r, _ = steer.generate(lm, sub, steerer, batch_size=a.batch)
            j = judge(r)
            C[key] = {"idx": [int(i) for i in idx], "resp": r, "refusal": [x["refusal"] for x in j],
                      "clf": [x["clf"] for x in j]}
            cache.write_text(json.dumps(C))
            log(f"generated {key}: {len(idx)} prompts {time.time()-t:.0f}s", L)
            return C[key]

        allidx = np.arange(len(prompts))
        base = run("base", None, allidx)
        refu = run("refusal", steer.Steerer(lm, behaviors=[beh]), allidx)
        # condition flags on the test prompts at each category's grid-searched point
        layers = sorted({pts[c]["layer"] for c in CATS})
        S = steer.condition_sims_multi(lm, prompts, cvs, layers, batch_size=a.batch)
        met = {c: (S[c][pts[c]["layer"]] < pts[c]["threshold"]) if pts[c]["cmp"] == "<"
               else (S[c][pts[c]["layer"]] > pts[c]["threshold"]) for c in CATS}
        res = {"base": {}, "refusal": {}, "condition_met": {}, "rules": {}, "points": {
            c: {k: pts[c][k] for k in ("layer", "cmp", "threshold", "f1")} for c in CATS},
            "paper_points": PAPER5.get(a.model), "behavior": {"layers": blayers, "alpha": alpha}}
        rate = lambda flags, m: float(100 * np.mean(np.array(flags)[m]))
        for c in CATS:
            m = cat == c
            res["base"][c] = rate(base["refusal"], m)
            res["refusal"][c] = rate(refu["refusal"], m)
            res["condition_met"][c] = {d: float(100 * met[c][cat == d].mean()) for d in CATS}
        for name, conds, op, neg in RULES:
            act = np.zeros(len(prompts), bool)
            for c in conds:
                act |= met[c]
            if neg:
                act = ~act
            exact = max(pts[c]["layer"] for c in conds) <= min(blayers)
            flags = np.array(base["refusal"])
            if exact:
                flags[act] = np.array(refu["refusal"])[act]
            else:
                rules = [steer.Rule(conds=list(range(len(conds))), behavior=0, op=op, negate=neg)]
                s = steer.Steerer(lm, [steer.Condition(cvs[c], [pts[c]["layer"]], pts[c]["threshold"], pts[c]["cmp"])
                                       for c in conds], [beh], rules)
                g = run(f"rule:{name}", s, np.where(act)[0])
                flags[act] = np.array(g["refusal"])
            res["rules"][name] = {"refusal": {c: rate(flags, cat == c) for c in CATS},
                                  "active": {c: float(100 * act[cat == c].mean()) for c in CATS}, "exact_shortcut": exact}
            log(f"{name:16s} refusal " + " ".join(f"{SHORT[c]}:{res['rules'][name]['refusal'][c]:5.1f}" for c in CATS)
                + ("" if exact else " (regenerated)"), L)
        log("base             refusal " + " ".join(f"{SHORT[c]}:{res['base'][c]:5.1f}" for c in CATS), L)
        log("+refusal (all)   refusal " + " ".join(f"{SHORT[c]}:{res['refusal'][c]:5.1f}" for c in CATS), L)
        (out / "summary.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
