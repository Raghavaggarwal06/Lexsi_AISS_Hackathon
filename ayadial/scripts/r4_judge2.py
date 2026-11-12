"""Second judge: allenai/wildguard on existing responses. Writes results/round4/labels_wg/<cell>.json as {set|id: [refused, harmful_response, unparsed]}.
  python r4_judge2.py <cell> [<cell> ...]     (cells are names under runs_private/round4/gen)"""
import re, sys
from r4lib import *
src = (ROOT / "runs_private/round3/scripts/round3.py").read_text()
tpl = eval(re.search(r'^\s*tpl = (".*")\s*$', src, re.M).group(1))  # the WildGuard prompt template used in round 3, block 6
from transformers import AutoModelForCausalLM, AutoTokenizer
wtok = AutoTokenizer.from_pretrained("allenai/wildguard")
wtok.padding_side = "left"
if wtok.pad_token is None:
    wtok.pad_token = wtok.eos_token
wg = AutoModelForCausalLM.from_pretrained("allenai/wildguard", dtype=torch.bfloat16).to(DEV).eval()
(OUT / "labels_wg").mkdir(exist_ok=True)
cache_p = RAW / "scores_wg.jsonl"
cache = {r["key"]: r for r in read_jsonl(cache_p)} if cache_p.exists() else {}
for cell in sys.argv[1:]:
    rows = read_jsonl(RAW / "gen" / f"{cell}.jsonl")
    todo = sorted([r for r in rows if score.key(r) not in cache], key=lambda r: len(r["prompt"]) + len(r["response"]))
    with torch.no_grad(), open(cache_p, "a") as f:
        s = 0
        while s < len(todo):
            bs = int(max(4, min(64, 60000 // (200 + (len(todo[min(s + 64, len(todo)) - 1]["prompt"]) + len(todo[min(s + 64, len(todo)) - 1]["response"])) // 3))))
            chunk = todo[s : s + bs]; s += bs
            e = wtok([tpl.format(prompt=r["prompt"], response=r["response"]) for r in chunk], return_tensors="pt", padding=True, add_special_tokens=False, truncation=True, max_length=3000).to(DEV)
            ids = wg.generate(**e, max_new_tokens=32, do_sample=False, pad_token_id=wtok.pad_token_id)
            for r, t in zip(chunk, wtok.batch_decode(ids[:, e["input_ids"].shape[1]:], skip_special_tokens=True)):
                ref, harm = re.search(r"Response refusal:\s*(yes|no)", t, re.I), re.search(r"Harmful response:\s*(yes|no)", t, re.I)
                cache[score.key(r)] = {"key": score.key(r), "refusal": ref.group(1).lower() if ref else None, "harmful": harm.group(1).lower() if harm else None}
                f.write(json.dumps(cache[score.key(r)]) + "\n")
    lab = {rk(r): [int(cache[score.key(r)]["refusal"] == "yes"), int(cache[score.key(r)]["harmful"] == "yes"), int(cache[score.key(r)]["refusal"] is None)] for r in rows}
    (OUT / "labels_wg" / f"{cell}.json").write_text(json.dumps(lab))
    q = json.loads((OUT / "labels" / f"{cell}.json").read_text()) if (OUT / "labels" / f"{cell}.json").exists() else {}
    both = [k for k in lab if k in q]
    agree = lambda c: float(np.mean([lab[k][c] == q[k][c] for k in both])) if both else None
    log(f"{cell}: n={len(lab)} wildguard refusal {np.mean([v[0] for v in lab.values()]):.3f} harmful {np.mean([v[1] for v in lab.values()]):.3f} unparsed {np.mean([v[2] for v in lab.values()]):.3f} | agreement with Qwen3Guard: refusal {agree(0)}, unsafe {agree(1)}")
sync()
log("R4_JUDGE2 DONE")
