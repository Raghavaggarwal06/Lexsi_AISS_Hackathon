"""Add a plain harmful / harmless pair to the round 5 sets, before the frozen test is declared: the pair teammate Abhinav's CAST work reports on
(SORRY-Bench harmful requests against Alpaca instructions), so that both lines of work can be read on the same yardstick.
  - every SORRY-Bench base-style question not yet sampled joins sorry_base_dev (question ids with sha1 mod 5 == 0) or sorry_base_frz (the rest);
  - 300 fresh Alpaca instructions (no input field, not in any set so far) form alp5_frz.
Same duplicate rules as r5_sets.py sorry. Refuses if the frozen base or the SORRY development base already exists.
  python scripts/r5_sets_plain.py"""
import collections, hashlib, json, random, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, read_jsonl, write_jsonl
from datasets import load_dataset
from r5_sets import OUT, RAW, MAXTOK, norm, hsh, hmod, ntok, row

rows = read_jsonl(RAW / "sets.jsonl")
gen = RAW / "gen"
assert not (gen / "base_frz5.jsonl").exists() and not (gen / "base_sdev5.jsonl").exists(), "a base that these rows belong to already exists"
assert not any(r["set"] == "alp5_frz" for r in rows), "already applied"
man = json.loads((OUT / "sets_manifest.json").read_text())
all_norm = {norm(r["prompt"]) for r in rows}
have = {r["id"] for r in rows if r["set"].startswith("sorry_base_")}
grams = lambda t: {t[i:i + 3] for i in range(max(1, len(t) - 2))}
ref = [grams(norm(r["prompt"])) for r in rows if r["set"] in ("harmful", "harmful_fit", "xstest_unsafe", "xstest_unsafe_fit", "mj_en_dev", "mj_en_frz")]
near = lambda t: any(len(grams(norm(t)) & h) / max(1, len(grams(norm(t)) | h)) >= 0.85 for h in ref if 0.7 <= len(h) / max(1, len(grams(norm(t)))) <= 1.43)
d = load_dataset("sorry-bench/sorry-bench-202503")
d = d[list(d.keys())[0]][:]
new, skipped = [], collections.Counter()
for i in range(len(d["question_id"])):
    if str(d["prompt_style"][i]) != "base":
        continue
    qid, t = int(d["question_id"][i]), d["turns"][i][0]
    if f"sorry-{qid}-base" in have:
        continue
    if not t or ntok(t) > MAXTOK:
        skipped["too_long_or_empty"] += 1; continue
    if norm(t) in all_norm:
        skipped["exact_duplicate_of_an_existing_row"] += 1; continue
    if near(t):
        skipped["near_duplicate_of_a_plain_harmful_row"] += 1; continue
    side = "dev" if hmod(f"sorry-{qid}", 5) == 0 else "frz"
    new.append(row(f"sorry_base_{side}", f"sorry-{qid}-base", t, "sorry", 1, style="base", category=str(d["category"][i])))
n_sorry = collections.Counter(r["set"] for r in new)
al = load_dataset("tatsu-lab/alpaca")["train"].select_columns(["instruction", "input"])[:]
rng = random.Random(505)
seen, alp = set(), []
for i in rng.sample(range(len(al["instruction"])), 8000):
    t = al["instruction"][i]
    if al["input"][i] or not (20 <= len(t) <= 200) or norm(t) in all_norm or norm(t) in seen:
        continue
    seen.add(norm(t)); alp.append(row("alp5_frz", f"alp-{i}", t, "alpaca", 0))
    if len(alp) >= 300:
        break
new += alp
write_jsonl(RAW / "sets.jsonl", rows + new)
frz = [r for r in rows + new if r["set"].endswith("_frz")]
fk = sorted(f"{r['set']}|{r['id']}" for r in frz)
ft = sorted(f"{r['set']}|{r['id']}|{hsh(r['prompt'])}" for r in frz)
man["plain_pair"] = {"added_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "why": "the plain harmful / harmless pair that the teammate's CAST work reports on (SORRY-Bench against Alpaca), for a common yardstick",
                     "sorry_base_added": dict(n_sorry), "alp5_frz": len(alp), "skipped": dict(skipped), "frozen_hash_before": man["frozen_ids_sha256"], "frozen_n_before": man["frozen_n"]}
for r in new:
    man["ids"].setdefault(r["set"], []).append(r["id"])
    man["new_sizes"][r["set"]] = man["new_sizes"].get(r["set"], 0) + 1
man["new_rows"] += len(new)
man["sides"]["development"] += n_sorry.get("sorry_base_dev", 0)
man["sides"]["frozen"] = len(frz)
man["frozen_n"], man["frozen_ids_sha256"], man["frozen_ids_and_text_sha256"] = len(frz), hashlib.sha256("\n".join(fk).encode()).hexdigest(), hashlib.sha256("\n".join(ft).encode()).hexdigest()
(OUT / "sets_manifest.json").write_text(json.dumps(man, indent=1))
print(json.dumps(man["plain_pair"], indent=1))
print("sizes", {k: man["new_sizes"][k] for k in ("sorry_base_dev", "sorry_base_frz", "alp5_frz")}, "| frozen", man["frozen_n"], man["frozen_ids_sha256"])
