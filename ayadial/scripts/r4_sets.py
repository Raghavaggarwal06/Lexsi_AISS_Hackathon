"""Build round 4 sets: fit / development / frozen. Writes private rows to runs_private/round4/sets.jsonl and an id manifest to results."""
import collections, hashlib, json, random, re
import numpy as np
from datasets import load_dataset
from r4lib import *

norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
hsh = lambda t: hashlib.sha1(t.encode()).hexdigest()
rng = random.Random(4)
tr = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)["train"]
ev = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)["train"]
r3 = json.loads((ROOT / "results/round3/block5/sets.json").read_text())
dev_h = set(r3["wildjailbreak_harmful_ids"])
rows, man = [], {}

# eval split: development (round 3 ids) and frozen harmful
evh = [i for i, t in enumerate(ev["data_type"]) if t == "adversarial_harmful"]
evb = [i for i, t in enumerate(ev["data_type"]) if t == "adversarial_benign"]
rows += [{"id": f"wj-{i}", "set": "wj_adversarial_harmful", "prompt": ev[i]["adversarial"]} for i in sorted(dev_h)]
rows += [{"id": f"wj-{i}", "set": "wj_adversarial_benign", "prompt": ev[i]["adversarial"]} for i in evb]
rest = sorted(set(evh) - dev_h)
rows += [{"id": f"wj-{i}", "set": "frozen_wjh", "prompt": ev[i]["adversarial"]} for i in sorted(rng.sample(rest, 600))]
eval_norm = {norm(t) for t in ev["adversarial"]}

# train split: pool, pseudo-tactic clusters (the HF release has no tactics column), vanilla-grouped partitions
dt = tr["data_type"]
pool = []
for kind in ("adversarial_harmful", "adversarial_benign"):
    idx = [i for i, t in enumerate(dt) if t == kind]
    pool += sorted(rng.sample(idx, 14000))
recs = tr.select(pool)
adv, van, kinds = recs["adversarial"], recs["vanilla"], recs["data_type"]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
X = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True, min_df=3).fit_transform(adv)
Z = normalize(TruncatedSVD(100, random_state=4).fit_transform(X))
K = 16
cl = KMeans(K, n_init=4, random_state=4).fit_predict(Z)
cnt = {c: (sum(1 for k, x in zip(kinds, cl) if x == c and k.endswith("harmful")), sum(1 for k, x in zip(kinds, cl) if x == c and k.endswith("benign"))) for c in range(K)}
elig = [c for c in range(K) if min(cnt[c]) >= 400]
held = sorted(rng.sample(elig, 4))
part = lambda v: int(hsh(norm(v)), 16) % 10  # 0-5 fit, 6-7 development, 8-9 frozen
quota = {"trfit": 2500, "dev_ho": 400, "dev_in": 300, "frozen_ho": 200, "frozen_in": 200}
got = collections.Counter()
seen_adv = set()
order = list(range(len(pool)))
rng.shuffle(order)
dropped_dup = 0
for j in order:
    a = norm(adv[j])
    if a in eval_norm or a in seen_adv:
        dropped_dup += 1
        continue
    p, ho, cls = part(van[j]), cl[j] in held, "h" if kinds[j].endswith("harmful") else "b"
    if p <= 5 and not ho:
        name = "trfit"
    elif p in (6, 7):
        name = "dev_ho" if ho else "dev_in"
    elif p >= 8:
        name = "frozen_ho" if ho else "frozen_in"
    else:
        continue  # fit-partition vanilla in a held-out cluster: unused
    if got[(name, cls)] >= quota[name]:
        continue
    got[(name, cls)] += 1
    seen_adv.add(a)
    rows.append({"id": f"wjtr-{pool[j]}", "set": f"{name}_{cls}", "prompt": adv[j], "cluster": int(cl[j]), "vanilla_hash": hsh(norm(van[j]))[:12]})

# our own sets (eval halves = development, fit halves kept for reference) + harmful_jb as in round 3
import ast as _ast
tree = _ast.parse((ROOT / "scripts/overnight.py").read_text())
JB = next(_ast.literal_eval(n.value) for n in tree.body if isinstance(n, _ast.Assign) and getattr(n.targets[0], "id", "") == "JB")
for name in ("harmful", "xstest_safe", "orbench_hard", "benign", "xstest_unsafe"):
    table = {r["id"]: r for r in load_set(name)}
    for half in ("eval", "fit"):
        rows += [{"id": i, "set": name if half == "eval" else f"{name}_fit", "prompt": table[i]["prompt"]} for i in load_splits()[name][half]]
r2 = random.Random(1)  # overnight used seed 1 for the eval wrap; ids carry the template so any mismatch is visible
for i in load_splits()["harmful"]["eval"]:
    p = next(r["prompt"] for r in load_set("harmful") if r["id"] == i)
    for t in r2.sample(range(5), 2):
        rows.append({"id": f"{i}::t{t + 1}", "set": "harmful_jb", "prompt": JB[t].format(p=p)})

# frozen extras: unseen OR-Bench-hard and Alpaca rows
used = {norm(r["prompt"]) for n in ("orbench_hard", "benign") for r in load_set(n)}
try:
    ob = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")["train"]
    fresh = [(i, r["prompt"]) for i, r in enumerate(ob) if norm(r["prompt"]) not in used]
    rows += [{"id": f"orb-fz-{i}", "set": "frozen_orbench", "prompt": p} for i, p in sorted(rng.sample(fresh, min(300, len(fresh))))]
except Exception as e:
    print("orbench frozen failed:", type(e).__name__, str(e)[:200])
try:
    al = load_dataset("tatsu-lab/alpaca")["train"]
    idx = rng.sample(range(len(al)), 2000)
    fresh = [(i, al[i]["instruction"]) for i in idx if not al[i]["input"] and norm(al[i]["instruction"]) not in used][:200]
    rows += [{"id": f"alp-fz-{i}", "set": "frozen_benign", "prompt": p} for i, p in fresh]
except Exception as e:
    print("alpaca frozen failed:", type(e).__name__, str(e)[:200])

# dedup: nothing used for fitting may equal anything used for testing
fit_norm = {norm(r["prompt"]) for r in rows if r["set"].startswith("trfit") or r["set"].endswith("_fit")}
test_rows = [r for r in rows if not (r["set"].startswith("trfit") or r["set"].endswith("_fit"))]
clash = sum(norm(r["prompt"]) in fit_norm for r in test_rows)
fit_van = {r["vanilla_hash"] for r in rows if r["set"].startswith("trfit")}
van_clash = sum(r.get("vanilla_hash") in fit_van for r in test_rows)
write_jsonl(RAW / "sets.jsonl", rows)
sizes = collections.Counter(r["set"] for r in rows)
frozen = sorted(rk(r) for r in rows if r["set"].startswith("frozen"))
man = {"sizes": dict(sorted(sizes.items())), "clusters": {"k": K, "counts_h_b": {str(c): cnt[c] for c in cnt}, "held_out": held, "method": "tf-idf 1-2gram, SVD 100, k-means, seed 4, on 28000 train adversarial rows"},
       "vanilla_partition": "sha1(normalised vanilla) mod 10: 0-5 fit, 6-7 development, 8-9 frozen", "dropped_duplicates_in_pool": dropped_dup,
       "fit_vs_test_exact_text_clashes": clash, "fit_vs_test_vanilla_clashes": van_clash,
       "frozen_ids_sha256": hashlib.sha256("\n".join(frozen).encode()).hexdigest(), "frozen_n": len(frozen),
       "ids": {s: [r["id"] for r in rows if r["set"] == s] for s in sorted(sizes)}}
(OUT / "sets_manifest.json").write_text(json.dumps(man, indent=1))
print(json.dumps({k: v for k, v in man.items() if k != "ids"}, indent=1))
sync()
