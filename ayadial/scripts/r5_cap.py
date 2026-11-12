"""Capability check: multiple-choice reading comprehension (facebook/belebele, 4 options) with the knob at several settings.
Accuracy is read from the next-token distribution at the reply-start token (letters A-D), so no generation and no degeneracy filter is involved.
  python scripts/r5_cap.py specs5/<spec>.json
spec = {"langs": ["eng_Latn", "spa_Latn", "zho_Hans", "arb_Arab", "swh_Latn"], "n": 300, "cells": [{"name": "foldcap@1", "f": "v1", "s": 1}, ...]}
Writes results/round5/capability/<spec>.json: accuracy per language and cell, the paired change against base with a 95% interval,
how often each cell's units fire on these harmless task prompts, and how often the chosen letter changes."""
import sys
from r5lib import *
from datasets import load_dataset

LETTERS = "ABCD"


def build(lang, n):
    d = load_dataset("facebook/belebele", lang)["test"]
    idx = random.Random(8).sample(range(len(d)), n)
    rows = []
    for i in idx:
        x = d[i]
        p = f"Read the passage and answer the question.\n\nPassage: {x['flores_passage']}\n\nQuestion: {x['question']}\nA. {x['mc_answer1']}\nB. {x['mc_answer2']}\nC. {x['mc_answer3']}\nD. {x['mc_answer4']}\n\nAnswer with the letter only."
        rows.append({"id": f"bele-{lang}-{i}", "prompt": p, "gold": int(x["correct_answer_num"]) - 1})
    return rows


@torch.no_grad()
def choose(model, tok, rows, ctx, letter_ids, batch=48):
    pred, fire = np.zeros(len(rows), int), np.zeros(len(rows))
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]))
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        e = enc(tok, [rows[i]["prompt"] for i in idx])
        if ctx is None:
            h = model.model(**e, use_cache=False).last_hidden_state[:, -1]
        else:
            ctx.fire = []
            with ctx:
                h = model.model(**e, use_cache=False).last_hidden_state[:, -1]
            if ctx.fire:
                fire[idx] = ctx.fire[0].float().numpy()
        lg = model.lm_head(h).float()[:, letter_ids]
        pred[idx] = lg.argmax(-1).cpu().numpy()
    return pred, fire


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    model, tok = load()
    letter_ids = [tok(l, add_special_tokens=False)["input_ids"][0] for l in LETTERS]
    assert len(set(letter_ids)) == 4
    out = {"task": "facebook/belebele, 4-option reading comprehension, letter read from next-token logits at the reply-start token", "n_per_language": spec["n"], "langs": spec["langs"], "cells": {}, "base": {}}
    data = {lg: build(lg, spec["n"]) for lg in spec["langs"]}
    gold = {lg: np.array([r["gold"] for r in data[lg]]) for lg in data}
    base = {lg: choose(model, tok, data[lg], None, letter_ids)[0] for lg in data}
    out["base"] = {lg: {"accuracy": float((base[lg] == gold[lg]).mean() * 100)} for lg in data}
    log(f"base {out['base']}")
    rng = np.random.default_rng(0)
    for c in spec["cells"]:
        e = {"spec": {k: v for k, v in c.items() if k != "name"}}
        for lg in data:
            pred, fire = choose(model, tok, data[lg], context(model, c), letter_ids)
            ok, ok0 = (pred == gold[lg]).astype(float), (base[lg] == gold[lg]).astype(float)
            bt = [(ok[i] - ok0[i]).mean() * 100 for i in (rng.integers(0, len(ok), len(ok)) for _ in range(2000))]
            e[lg] = {"accuracy": float(ok.mean() * 100), "change_vs_base": float((ok - ok0).mean() * 100), "change_ci95": [float(x) for x in np.percentile(bt, [2.5, 97.5])],
                     "letter_changed": float((pred != base[lg]).mean() * 100), "fires_abs_gt_half": float((np.abs(fire) > 0.5).mean() * 100), "fire_mean": float(fire.mean())}
        out["cells"][c["name"]] = e
        log(f"{c['name']} " + " | ".join(f"{lg} {e[lg]['accuracy']:.1f} ({e[lg]['change_vs_base']:+.1f}) fire {e[lg]['fires_abs_gt_half']:.0f}%" for lg in data))
    (OUT / "capability").mkdir(exist_ok=True)
    (OUT / "capability" / f"{Path(sys.argv[1]).stem}.json").write_text(json.dumps(out, indent=1))
    sync()
    log("R5_CAP DONE")


if __name__ == "__main__":
    main()
