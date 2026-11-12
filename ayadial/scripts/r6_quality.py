"""Does generated text get worse with the adapters installed? Round 5 measured only a multiple-choice letter read from logits.
Two tasks that generate text and can be scored without a judge:
  translation   150 passages (the FLORES passages inside facebook/belebele, which are parallel across languages), four directions; score chrF
  short answers 150 questions per language from google/xquad (en, es, zh, ar); score = the gold answer appears in the reply
  python scripts/r6_quality.py specs6/<spec>.json      spec = {"cells": [{"name": ..., "f": ...}, ...], "n": 150}
Every cell is compared with the base model on the same prompts in the same batches (paired bootstrap over prompts). Replies are private
(runs_private/round6/quality/); results/round6/quality/<spec>.json holds scores, counts and fire rates only."""
import sys
from r6lib import *
from datasets import load_dataset

LANGNAME = {"eng_Latn": "English", "spa_Latn": "Spanish", "zho_Hans": "Chinese (Simplified)", "swh_Latn": "Swahili"}
DIRS = [("eng_Latn", "spa_Latn"), ("eng_Latn", "zho_Hans"), ("spa_Latn", "eng_Latn"), ("swh_Latn", "eng_Latn")]
XQ = ["en", "es", "zh", "ar"]


def tasks(n):
    rng = random.Random(66)
    P = {}
    for lg in LANGNAME:
        d = load_dataset("facebook/belebele", lg)["test"]
        P[lg] = {x["link"]: x["flores_passage"] for x in d}
    links = sorted(set.intersection(*[set(v) for v in P.values()]))
    links = rng.sample(links, n)
    T = {}
    for a, b in DIRS:
        T[f"mt_{a[:3]}_{b[:3]}"] = [{"id": f"mt-{a[:3]}-{b[:3]}-{i}", "prompt": f"Translate the following text into {LANGNAME[b]}. Reply with the translation only.\n\n{P[a][l]}", "ref": P[b][l], "kind": "mt"} for i, l in enumerate(links)]
    for lg in XQ:
        d = load_dataset("google/xquad", f"xquad.{lg}")["validation"]
        idx = rng.sample(range(len(d)), n)
        T[f"qa_{lg}"] = [{"id": f"qa-{lg}-{i}", "prompt": f"{d[i]['context']}\n\nQuestion: {d[i]['question']}\nAnswer with a short phrase taken from the text.", "ref": d[i]["answers"]["text"], "kind": "qa"} for i in idx]
    return T


def score_row(row, text):
    if row["kind"] == "mt":
        return CH.sentence_score(text.strip(), [row["ref"]]).score
    norm = lambda s: re.sub(r"\s+", " ", s.lower().strip())
    return 100.0 * float(any(norm(a) in norm(text) for a in row["ref"]))


def main():
    global CH
    from sacrebleu.metrics import CHRF
    CH = CHRF()
    spec = json.loads(Path(sys.argv[1]).read_text())
    T = tasks(spec.get("n", 150))
    model, tok = load()
    (RAW / "quality").mkdir(exist_ok=True)
    (OUT / "quality").mkdir(exist_ok=True)
    outp = OUT / "quality" / f"{Path(sys.argv[1]).stem}.json"
    out = {"tasks": {k: {"n": len(v), "metric": "chrF" if v[0]["kind"] == "mt" else "gold answer contained in the reply, %"} for k, v in T.items()}, "max_new_tokens": {"mt": 320, "qa": 48}, "cells": {}}
    rng = np.random.default_rng(0)
    res = {}
    for c in [{"name": "base", "f": "base"}] + spec["cells"]:
        e = {}
        for tname, rows in T.items():
            path = RAW / "quality" / f"{tname}__{c['name']}.jsonl"
            if path.exists():
                got = read_jsonl(path)
            else:
                os.environ["R4_TOK"], os.environ["R4_BATCH"] = "60000", "64"
                texts, fire = generate(model, tok, rows, context(model, c), 320 if rows[0]["kind"] == "mt" else 48, 64)
                got = [{"id": r["id"], "response": t, "fire": f} for r, t, f in zip(rows, texts, fire)]
                write_jsonl(path, got)
            sc = np.array([score_row(r, g["response"]) for r, g in zip(rows, got)])
            res[(c["name"], tname)] = (sc, [g["response"] for g in got])
            e[tname] = {"score": float(sc.mean()), "fires_abs_gt_half": float(np.mean([abs(g.get("fire") or 0.0) > 0.5 for g in got]) * 100), "empty_or_degenerate": float(np.mean([r4lib.score.is_degenerate(g["response"]) for g in got]) * 100)}
            if c["name"] != "base":
                b_sc, b_tx = res[("base", tname)]
                d = sc - b_sc
                bt = [d[i].mean() for i in (rng.integers(0, len(d), len(d)) for _ in range(2000))]
                e[tname].update({"change_vs_base": float(d.mean()), "change_ci95": [float(x) for x in np.percentile(bt, [2.5, 97.5])], "identical_replies_pct": float(np.mean([a == b for a, b in zip(res[(c["name"], tname)][1], b_tx)]) * 100)})
        out["cells"][c["name"]] = e
        log(f"{c['name']} " + " | ".join(f"{t} {v['score']:.1f}" + (f" ({v['change_vs_base']:+.1f})" if "change_vs_base" in v else "") for t, v in e.items()))
        outp.write_text(json.dumps(out, indent=1))
    sync()
    log("R6_QUALITY DONE")


if __name__ == "__main__":
    main()
