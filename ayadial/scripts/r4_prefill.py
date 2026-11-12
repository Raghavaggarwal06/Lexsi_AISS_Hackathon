"""Fable's prefill test (REVIEWS 11.7): take the first k tokens of each steered reply, append them to the prompt, and let the
UNSTEERED model continue. If refusal matches the steered run, the decision travels in the first tokens.
  python r4_prefill.py <steered cell> <base cell> [n]"""
import sys
from r4lib import *
cell, base_c = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 500
H = ["wj_adversarial_harmful", "dev_ho_h", "dev_in_h"]
model, tok = load()
rows = [r for r in read_jsonl(RAW / "gen" / f"{cell}.jsonl") if r["set"] in H]
random.Random(5).shuffle(rows)
rows = rows[:N]
steered, base = labels(cell), labels(base_c)
keys = [rk(r) for r in rows]
out = {"steered_cell": cell, "n": len(rows), "base_refusal": float(np.mean([base[k][0] for k in keys]) * 100), "steered_refusal": float(np.mean([steered[k][0] for k in keys]) * 100),
       "base_unsafe": float(np.mean([base[k][1] for k in keys]) * 100), "steered_unsafe": float(np.mean([steered[k][1] for k in keys]) * 100)}


@torch.no_grad()
def continue_from(rows, k, batch=64):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]))
    res = [None] * len(rows)
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        pre = [tok.decode(tok(rows[i]["response"], add_special_tokens=False)["input_ids"][:k]) for i in idx]
        texts = [render(tok, rows[i]["prompt"]) + p for i, p in zip(idx, pre)]
        e = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN + 8, add_special_tokens=False).to(DEV)
        ids = model.generate(**e, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=128 - k, pad_token_id=tok.pad_token_id)
        for i, p, t in zip(idx, pre, tok.batch_decode(ids[:, e["input_ids"].shape[1]:], skip_special_tokens=True)):
            res[i] = p + t
    return res


for k in (1, 2, 3, 5):
    texts = continue_from(rows, k)
    got = [{"id": r["id"], "set": r["set"], "prompt": r["prompt"], "response": t} for r, t in zip(rows, texts)]
    write_jsonl(RAW / "gen" / f"prefill_{cell}_k{k}.jsonl", got)
    judge(got)
    lab = np.array([label(r) for r in got], float)
    same = float(np.mean([l[0] == steered[rk(r)][0] for l, r in zip(lab, got)]) * 100)
    out[f"k{k}"] = {"refusal": float(lab[:, 0].mean() * 100), "unsafe": float(lab[:, 1].mean() * 100), "per_prompt_refusal_agreement_with_steered": same}
    log(f"k={k}: {out[f'k{k}']}")
(OUT / "controls").mkdir(exist_ok=True)
(OUT / "controls" / f"prefill_{cell}.json").write_text(json.dumps(out, indent=1))
sync()
log("R4_PREFILL DONE")
