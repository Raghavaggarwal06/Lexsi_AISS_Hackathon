"""Build the round 5 sets. CPU only; runs on the Mac or on the runtime.
Private rows -> runs_private/round5/sets.jsonl. Ids, sizes, rules and hashes -> results/round5/sets_manifest.json (no prompt text).

Every new benchmark is split once, by underlying request, into a development slice (about 20%) and a frozen slice.
  python scripts/r5_sets.py            build everything except SORRY-Bench
  python scripts/r5_sets.py sorry      append SORRY-Bench (only if the gate is open; refuses if any sorry row was already generated)
"""
import collections, hashlib, json, random, re, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, read_jsonl, write_jsonl, load_set, load_splits
from datasets import load_dataset
from transformers import AutoTokenizer

OUT, RAW = ROOT / "results/round5", ROOT / "runs_private/round5"
OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
hsh = lambda t: hashlib.sha1(t.encode()).hexdigest()
hmod = lambda t, m: int(hsh(t), 16) % m
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
ntok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
MAXTOK = 1350  # prompt tokens; the template adds about 60 and the run-time limit is 1536
JBB_SHA = json.loads((ROOT / "results/round3/block5/sets.json").read_text())["jbb_artifacts_sha"]
CORE = ["Illegal Activity", "Hate Speech", "Malware", "Physical Harm", "Fraud", "Privacy Violence"]
LANG = {"en": ("eng", "English"), "zh": ("zho", "Simplified Chinese"), "it": ("ita", "Italian"), "vi": ("vie", "Vietnamese"), "ar": ("arb", "Standard Arabic"),
        "ko": ("kor", "Korean"), "th": ("tha", "Thai"), "bn": ("ben", "Bengali"), "sw": ("swh", "Swahili"), "jv": ("jav", "Javanese")}
RES = {"high": ["zh", "it", "vi"], "medium": ["ar", "ko", "th"], "low": ["bn", "sw", "jv"]}


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "eval-client"}), timeout=120).read()


def row(set_, id_, prompt, src, harm, **kw):
    return {"id": id_, "set": set_, "prompt": prompt, "src": src, "harm": harm, **kw}


def build():
    rng = random.Random(5)
    info = {}
    r4 = read_jsonl(ROOT / "runs_private/round4/sets.jsonl")
    keep4 = [r for r in r4 if not r["set"].startswith("frozen")]  # same order as last night, so last night's batching can be reproduced
    r4_norm = {norm(r["prompt"]) for r in r4}
    r4_ids = {(r["set"], r["id"]) for r in r4}
    new = []
    splits = load_splits()
    own = {n: {r["id"]: r for r in load_set(n)} for n in ("harmful", "benign", "xstest_safe", "orbench_hard", "xstest_unsafe")}

    # --- Alpaca: fresh harmless instructions, used by several parts
    al = load_dataset("tatsu-lab/alpaca")["train"].select_columns(["instruction", "input"])[:]  # plain lists: column objects are slow to index
    al_idx = [i for i in rng.sample(range(len(al["instruction"])), 6000) if not al["input"][i] and 20 <= len(al["instruction"][i]) <= 140 and norm(al["instruction"][i]) not in r4_norm]
    seen, fresh_al = set(), []
    for i in al_idx:
        n = norm(al["instruction"][i])
        if n not in seen:
            seen.add(n); fresh_al.append((i, al["instruction"][i]))
    al_dev = [(i, t) for i, t in fresh_al if hmod(norm(t), 5) == 0]
    al_frz = [(i, t) for i, t in fresh_al if hmod(norm(t), 5) != 0]
    info["alpaca_fresh"] = {"dev_pool": len(al_dev), "frozen_pool": len(al_frz), "rule": "no input field, 20-140 characters, not in data/ or any round 4 set; development if sha1(normalised text) mod 5 == 0"}

    # --- A. in-the-wild templates x forbidden questions
    itw_ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25")["train"]
    I = itw_ds.select_columns(["prompt", "community_id", "platform"])[:]
    itw = [{"prompt": I["prompt"][i], "community_id": I["community_id"][i], "platform": I["platform"][i]} for i in range(len(I["prompt"]))]
    F = load_dataset("TrustAIRLab/forbidden_question_set")["train"][:]
    fq = [{k: F[k][i] for k in F} for i in range(len(F["question"]))]
    PH = re.compile(r"\[\s*(?:insert\s+)?(?:your\s+)?(?:prompt|question|query)(?:\s+here)?\s*\]|\{\s*(?:prompt|question|query)\s*\}", re.I)
    fill = lambda t, q: (PH.sub(lambda m: q, t), "placeholder") if PH.search(t) else (t.rstrip() + "\n\n" + q, "append")
    tpl, seen = [], set()
    for i in range(len(itw)):
        t = itw[i]["prompt"]
        if not t or norm(t) in seen:
            continue
        seen.add(norm(t))
        c = itw[i].get("community_id")
        group = f"c{int(c)}" if c is not None and str(c) not in ("", "nan", "None") else f"solo{hsh(norm(t))[:10]}"
        tpl.append({"i": i, "t": t, "group": group, "ntok": ntok(t), "platform": itw[i]["platform"]})
    elig = [x for x in tpl if 40 <= x["ntok"] <= 900]
    side = lambda x: "dev" if hmod(x["group"], 5) == 0 else "frz"
    picked = {}
    for s_, n_total, cap in (("dev", 60, 6), ("frz", 240, 12)):
        pool = [x for x in elig if side(x) == s_]
        rng.shuffle(pool)
        per, out = collections.Counter(), []
        for x in pool:
            if per[x["group"]] < cap and len(out) < n_total:
                per[x["group"]] += 1; out.append(x)
        picked[s_] = sorted(out, key=lambda x: x["i"])
    q_all = [(int(fq[i]["content_policy_id"]), int(fq[i]["q_id"]), fq[i]["question"], fq[i]["content_policy_name"]) for i in range(len(fq))]
    q_core = [q for q in q_all if q[3] in CORE]
    q_side = {"dev": [q for q in q_core if hmod(norm(q[2]), 5) == 0], "frz": [q for q in q_core if hmod(norm(q[2]), 5) != 0]}
    a_side = {"dev": al_dev[:40], "frz": al_frz[:160]}
    modes = collections.Counter()
    for s_, n_h, n_b in (("dev", 3, 2), ("frz", 2, 1)):
        for x in picked[s_]:
            r_ = random.Random(1000 + x["i"])
            for (pid, qid, q, _) in r_.sample(q_side[s_], n_h):
                p, mode = fill(x["t"], q)
                modes[mode] += 1
                new.append(row(f"itw_h_{s_}", f"itw-{x['i']}::q{pid}-{qid}", p, "itw", 1, template=x["i"], group=x["group"], mode=mode))
            for (ai, a) in r_.sample(a_side[s_], n_b):
                p, mode = fill(x["t"], a)
                new.append(row(f"itw_b_{s_}", f"itw-{x['i']}::a{ai}", p, "itw", 0, template=x["i"], group=x["group"], mode=mode))
    info["itw"] = {"templates_total": len(itw), "templates_unique": len(tpl), "eligible_40_to_900_tokens": len(elig), "communities": len({x["group"] for x in tpl if x["group"].startswith("c")}),
                   "templates_without_community": sum(x["group"].startswith("solo") for x in tpl), "picked_templates": {k: len(v) for k, v in picked.items()},
                   "core_scenarios": CORE, "core_questions": {k: len(v) for k, v in q_side.items()}, "benign_instructions": {k: len(v) for k, v in a_side.items()}, "fill_modes_harmful": dict(modes),
                   "rule": "templates split by community id (templates without one by their own hash), development if sha1(group) mod 5 == 0, at most 6 (development) or 12 (frozen) templates per community; questions split by sha1 mod 5; no template and no question on both sides"}

    # --- B. MultiJail and harmless prompts in the same languages
    M = load_dataset("DAMO-NLP-SG/MultiJail")["train"][:]
    mj = [{k: M[k][i] for k in M} for i in range(len(M["id"]))]
    ids = list(range(len(mj)))
    mj_dev = [i for i in ids if hmod(f"mj-{mj[i]['id']}", 5) == 0]
    texts = lambda i: {norm(mj[i][lg]) for lg in LANG}
    dev_texts = set().union(*[texts(i) for i in mj_dev])
    rest_mj, seen_t, mj_twins = [], set(), 0
    for i in sorted(set(ids) - set(mj_dev)):  # a request whose text in any language equals another request's stays on that request's side
        if texts(i) & dev_texts or texts(i) & seen_t:
            mj_twins += 1; continue
        seen_t |= texts(i); rest_mj.append(i)
    mj_frz = sorted(rng.sample(rest_mj, 140))
    for s_, idx in (("dev", mj_dev), ("frz", mj_frz)):
        for i in idx:
            for lg in LANG:
                new.append(row(f"mj_{lg}_{s_}", f"mj-{mj[i]['id']}-{lg}", mj[i][lg], "multijail", 1, lang=lg, request=int(mj[i]["id"])))
    AY = load_dataset("CohereLabs/aya_dataset")["train"].select_columns(["language_code", "language", "inputs", "annotation_type"])[:]
    codes, names, inputs, ann = AY["language_code"], AY["language"], AY["inputs"], AY["annotation_type"]
    info["aya"] = {}
    for lg, (code, name) in LANG.items():
        cand, seen = [], set()
        for pref in ("original-annotations", "re-annotations"):
            for i in range(len(inputs)):
                if codes[i] == code and names[i] == name and ann[i] == pref and 15 <= len(inputs[i]) <= 300 and norm(inputs[i]) not in seen and norm(inputs[i]) not in r4_norm:
                    seen.add(norm(inputs[i])); cand.append((i, inputs[i]))
        r_ = random.Random(77)
        r_.shuffle(cand)
        d_ = [c for c in cand if hmod(norm(c[1]), 5) == 0][:30]
        f_ = [c for c in cand if hmod(norm(c[1]), 5) != 0][:70]
        for s_, part in (("dev", d_), ("frz", f_)):
            for i, t in part:
                new.append(row(f"aya_{lg}_{s_}", f"aya-{i}", t, "aya", 0, lang=lg))
        info["aya"][lg] = {"candidates": len(cand), "dev": len(d_), "frozen": len(f_)}
    info["multijail"] = {"requests": len(mj), "dev_requests": len(mj_dev), "frozen_requests": len(mj_frz), "requests_dropped_as_twins_of_another": mj_twins, "languages": list(LANG), "resource_levels": RES,
                         "rule": "split by request id (sha1('mj-<id>') mod 5 == 0 is development), so the 10 language versions of a request share a side; 140 of the remaining requests sampled for the frozen slice"}

    # --- C. PHTest
    PHD = load_dataset("furonghuang-lab/PHTest")["train"][:]
    ph = [{k: PHD[k][i] for k in PHD} for i in range(len(PHD["ID"]))]
    seen = set()
    for kind, tag, n_dev, n_frz in (("harmless", "harmless", 150, 450), ("controversial", "contro", 75, 225)):
        cand = []
        for i in range(len(ph)):
            t = ph[i]["Request"]
            if ph[i]["Harmfulness"] == kind and norm(t) not in seen and norm(t) not in r4_norm:
                seen.add(norm(t)); cand.append((int(ph[i]["ID"]), t))
        r_ = random.Random(31)
        r_.shuffle(cand)
        for s_, part in (("dev", [c for c in cand if hmod(norm(c[1]), 5) == 0][:n_dev]), ("frz", [c for c in cand if hmod(norm(c[1]), 5) != 0][:n_frz])):
            for i, t in part:
                new.append(row(f"pht_{tag}_{s_}", f"pht-{i}", t, "phtest", 0 if kind == "harmless" else 0.5))
    info["phtest"] = {"rows": len(ph), "rule": "harmless and controversial sampled apart; development if sha1(normalised request) mod 5 == 0"}

    # --- D. JailbreakBench attack artifacts
    goal_id = {r["prompt"].strip().lower(): r["id"] for r in own["harmful"].values() if r["id"].startswith("jbb-")}
    fit_ids, eval_ids = set(splits["harmful"]["fit"]), set(splits["harmful"]["eval"])
    pool_of = lambda gid: "pool" if gid in fit_ids else ("dev" if hmod(gid, 2) == 0 else "frz")
    FILES = {"GCG": ["white_box/vicuna-13b-v1.5", "white_box/llama-2-7b-chat-hf", "transfer/gpt-3.5-turbo-1106", "transfer/gpt-4-0125-preview"],
             "PAIR": ["black_box/vicuna-13b-v1.5", "black_box/llama-2-7b-chat-hf", "black_box/gpt-3.5-turbo-1106", "black_box/gpt-4-0125-preview"],
             "DSN": ["white_box/vicuna-13b-v1.5", "white_box/llama-2-7b-chat-hf"],
             "JBC": ["manual/vicuna-13b-v1.5", "manual/llama-2-7b-chat-hf", "manual/gpt-3.5-turbo-1106", "manual/gpt-4-0125-preview"],
             "RS": ["black_box/vicuna-13b-v1.5", "black_box/llama-2-7b-chat-hf", "black_box/gpt-3.5-turbo-1106", "black_box/gpt-4-0125-preview"]}
    DIR = {"RS": "prompt_with_random_search"}
    SHORT = {"vicuna-13b-v1.5": "vic", "llama-2-7b-chat-hf": "ll2", "gpt-3.5-turbo-1106": "g35", "gpt-4-0125-preview": "g4"}
    jbb_n, suffix, too_long, unmatched, no_attack = collections.Counter(), {"pool": [], "dev": [], "frz": []}, 0, set(), collections.Counter()
    for method, files in FILES.items():
        seen_txt = set()
        for f in files:
            data = json.loads(fetch(f"https://raw.githubusercontent.com/JailbreakBench/artifacts/{JBB_SHA}/attack-artifacts/{DIR.get(method, method)}/{f}.json"))
            model = SHORT[f.split("/")[1]]
            for j in data.get("jailbreaks", []):
                goal, p = str(j.get("goal", "")).strip(), j.get("prompt")
                gid = goal_id.get(goal.lower())
                if not gid:
                    unmatched.add(goal.lower()); continue
                if not p or (gid, norm(p)) in seen_txt:
                    continue
                if norm(p) == norm(goal):  # an "attack" that is just the plain behaviour: not an artifact
                    no_attack[method] += 1; continue
                seen_txt.add((gid, norm(p)))
                if ntok(p) > MAXTOK:
                    too_long += 1; continue
                pl = pool_of(gid)
                seen_last_night = int(model == "vic" and method in ("GCG", "PAIR") and pl != "pool")
                new.append(row(f"jbb_{method}_{pl}", f"{gid}::{method}:{model}", p, "jbb", 1, behaviour=gid, method=method, target=model, seen_r4=seen_last_night))
                jbb_n[(method, pl)] += 1
                if method == "GCG" and p.lower().startswith(goal.lower()) and len(p) > len(goal) + 8:
                    suffix[pl].append(p[len(goal):])
    suffix = {k: sorted(set(v)) for k, v in suffix.items()}
    info["jbb"] = {"artifacts_sha": JBB_SHA, "behaviours_in_our_harmful_set": len(goal_id), "behaviours_not_in_our_set_dropped": len(unmatched),
                   "behaviours": {"pool_fit_half": len({g for g in goal_id.values() if g in fit_ids}), "dev": len({g for g in goal_id.values() if g in eval_ids and hmod(g, 2) == 0}), "frozen": len({g for g in goal_id.values() if g in eval_ids and hmod(g, 2) != 0})},
                   "prompts": {f"{m}_{p}": n for (m, p), n in sorted(jbb_n.items())}, "dropped_over_token_limit": too_long, "dropped_prompt_equal_to_plain_behaviour": dict(no_attack), "gcg_suffixes": {k: len(v) for k, v in suffix.items()},
                   "rule": "behaviours in our harmful fit half form the pool (only GCG and PAIR of the pool may enter a refit); behaviours in the eval half are held out and split by sha1(id) mod 2 into development and frozen; duplicates of a prompt across target models are kept once",
                   "cleanliness": "the vicuna GCG and PAIR prompts of held-out behaviours (flag seen_r4) were run under v1 and the control in round 4 and their aggregate rates were read; nothing was fit or tuned on them. Every other artifact was never run."}

    # suffix transplants: published GCG suffixes appended to other requests. Not optimised for the request they are attached to.
    WT = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)["train"].select_columns(["data_type", "vanilla", "adversarial"])[:]
    dt, van, adv = WT["data_type"], WT["vanilla"], WT["adversarial"]
    n_tr = len(dt)
    part = lambda v: hmod(norm(v), 10)  # round 4 partition of vanilla requests: 0-5 fit, 6-7 development, 8-9 frozen
    van_fit = {"h": [], "b": []}
    seen = set()
    for i in rng.sample(range(n_tr), 60000):
        v = van[i]
        if v and 20 <= len(v) <= 300 and part(v) <= 5 and norm(v) not in seen:
            seen.add(norm(v)); van_fit["h" if dt[i].endswith("harmful") else "b"].append((i, v))
    tp = lambda base, sfx_pool, r_: base + r_.choice(sfx_pool)
    r_ = random.Random(55)
    aug = {"h": [(f"own-{i}", own["harmful"][i]["prompt"]) for i in splits["harmful"]["fit"]] + [(f"own-{i}", own["xstest_unsafe"][i]["prompt"]) for i in splits["xstest_unsafe"]["fit"]] + [(f"wjv-{i}", v) for i, v in van_fit["h"][:355]],
           "b": [(f"own-{i}", own[n][i]["prompt"]) for n in ("benign", "xstest_safe", "orbench_hard") for i in splits[n]["fit"]] + [(f"wjv-{i}", v) for i, v in van_fit["b"][:225]]}
    for c, items in aug.items():
        for i, t in items:
            new.append(row(f"aug_gcg_{c}_fit", f"aug-{i}", tp(t, suffix["pool"], r_), "transplant", 1 if c == "h" else 0))
    hb_eval = [i for i in splits["harmful"]["eval"] if i.startswith("harmbench")]
    dev_src = {"h": [(i, own["harmful"][i]["prompt"]) for i in hb_eval], "b": [(i, own[n][i]["prompt"]) for n in ("benign", "xstest_safe") for i in splits[n]["eval"][:50]]}
    for c, items in dev_src.items():
        for i, t in items[:100]:
            new.append(row(f"sfx_{c}_dev", f"sfx-{i}", tp(t, suffix["dev"], r_), "transplant", 1 if c == "h" else 0))
    frz_src = {"h": [(f"mj{mj[i]['id']}", mj[i]["en"]) for i in mj_frz[:100]], "b": [(f"al{i}", t) for i, t in al_frz[160:260]]}
    for c, items in frz_src.items():
        for i, t in items:
            new.append(row(f"sfx_{c}_frz", f"sfx-{i}", tp(t, suffix["frz"], r_), "transplant", 1 if c == "h" else 0))
    info["transplants"] = {"note": "a published GCG suffix (from the pool, development or frozen behaviours, matching the side of the set) appended to another request; labelled by the request",
                           "wj_vanilla_fit_candidates": {k: len(v) for k, v in van_fit.items()}}

    # --- F. fresh WildJailbreak and OR-Bench rows for the new frozen test
    WE = load_dataset("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)["train"][:]
    used_ev = {int(r["id"].split("-")[1]) for r in r4 if r["id"].startswith("wj-")}
    rest = [i for i, t in enumerate(WE["data_type"]) if t == "adversarial_harmful" and i not in used_ev]
    for i in sorted(rng.sample(rest, 450)):
        new.append(row("wj5_h_frz", f"wj-{i}", WE["adversarial"][i], "wildjailbreak", 1))
    used_tr = {int(r["id"].split("-")[1]) for r in r4 if r["id"].startswith("wjtr-")}
    eval_norm = {norm(t) for t in WE["adversarial"]}
    cand, seen = [], set()
    for i in rng.sample(range(n_tr), 80000):
        if dt[i] == "adversarial_benign" and i not in used_tr and part(van[i]) >= 8:
            a = norm(adv[i])
            if a not in seen and a not in r4_norm and a not in eval_norm and ntok(adv[i]) <= MAXTOK:
                seen.add(a); cand.append(i)
        if len(cand) >= 350:
            break
    for i in sorted(cand):
        new.append(row("wj5_b_frz", f"wjtr-{i}", adv[i], "wildjailbreak", 0))
    ob = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")["train"][:]["prompt"]
    fresh = [(i, p) for i, p in enumerate(ob) if norm(p) not in r4_norm]
    for i, p in sorted(rng.sample(fresh, min(250, len(fresh)))):
        new.append(row("orb5_frz", f"orb-{i}", p, "orbench", 0))
    info["fresh"] = {"wj_eval_adversarial_harmful_unused": len(rest), "wj5_b_frz_candidates": len(cand), "orbench_fresh_available": len(fresh),
                     "note": "wj5_h_frz: eval-split adversarial_harmful rows never used in rounds 3 or 4. wj5_b_frz: train-split adversarial_benign rows whose vanilla request is in the frozen partition (8-9) and that were not drawn in round 4; no cluster labels, so no held-out-tactic split this time."}
    return keep4, new, info, r4


def finish(keep4, new, info, r4, extra_note=None):
    # drop anything over the token limit, then dedup inside the new rows and against every fit row
    out, seen_key, long_ = [], set(), collections.Counter()
    for r in new:
        k = (r["set"], r["id"])
        if k in seen_key:
            continue
        if ntok(r["prompt"]) > MAXTOK:
            long_[r["set"]] += 1; continue
        seen_key.add(k); out.append(r)
    is_fit = lambda s: s.startswith("trfit") or s.endswith("_fit") or s in ("jbb_GCG_pool", "jbb_PAIR_pool")
    fit_norm = {norm(r["prompt"]) for r in keep4 + out if is_fit(r["set"])}
    dev_norm = {norm(r["prompt"]) for r in keep4 if not is_fit(r["set"])} | {norm(r["prompt"]) for r in out if r["set"].endswith("_dev") or r["set"].endswith("_pool")}
    frz = [r for r in out if r["set"].endswith("_frz")]
    clash_fit = sum(norm(r["prompt"]) in fit_norm for r in out if not is_fit(r["set"]))
    clash_dev_frz = sum(norm(r["prompt"]) in dev_norm for r in frz)
    r4_frozen_norm = {norm(r["prompt"]) for r in r4 if r["set"].startswith("frozen")}
    clash_r4_frozen = sum(norm(r["prompt"]) in r4_frozen_norm for r in frz)
    rows = keep4 + out
    write_jsonl(RAW / "sets.jsonl", rows)
    sizes = collections.Counter(r["set"] for r in out)
    fk = sorted(f"{r['set']}|{r['id']}" for r in frz)
    ft = sorted(f"{r['set']}|{r['id']}|{hsh(r['prompt'])}" for r in frz)
    man = {"written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "round4_rows_carried": len(keep4), "new_rows": len(out),
           "new_sizes": dict(sorted(sizes.items())), "dropped_over_token_limit": dict(long_), "token_limit": MAXTOK,
           "sides": {"development": sum(n for s, n in sizes.items() if s.endswith("_dev") or s.endswith("_pool")), "frozen": len(frz), "fit_only": sum(n for s, n in sizes.items() if s.endswith("_fit"))},
           "exact_text_clashes": {"test_rows_equal_to_a_fit_row": clash_fit, "frozen_rows_equal_to_a_development_row": clash_dev_frz, "frozen_rows_equal_to_a_round4_frozen_row": clash_r4_frozen},
           "frozen_n": len(frz), "frozen_ids_sha256": hashlib.sha256("\n".join(fk).encode()).hexdigest(), "frozen_ids_and_text_sha256": hashlib.sha256("\n".join(ft).encode()).hexdigest(),
           "parts": info, "ids": {s: [r["id"] for r in out if r["set"] == s] for s in sorted(sizes)}}
    if extra_note:
        man["note"] = extra_note
    (OUT / "sets_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k != "ids"}, indent=1, ensure_ascii=False))


def sorry():
    """Append SORRY-Bench if the gate is open. Split by question id: development if sha1 mod 5 == 0, so the 21 styles of a question share a side.
    A candidate is skipped if its normalised text equals any existing row, or if its character 3-gram Jaccard with a plain harmful request already
    in the sets (our harmful and XSTest-unsafe rows, MultiJail English) is 0.85 or more (exact Jaccard; CuratorKIT's threshold, without the MinHash estimate)."""
    rows = read_jsonl(RAW / "sets.jsonl")
    assert not any(r["set"].startswith("sorry_") for r in rows), "SORRY rows are already in sets.jsonl"
    gen = RAW / "gen"
    assert not (gen / "base_frz5.jsonl").exists(), "the frozen base already exists: the frozen slice can no longer change"
    ds = load_dataset("sorry-bench/sorry-bench-202503")
    d = ds[list(ds.keys())[0]][:]
    n_all = len(d["question_id"])
    man = json.loads((OUT / "sets_manifest.json").read_text())
    grams = lambda t: {t[i:i + 3] for i in range(max(1, len(t) - 2))}
    all_norm = {norm(r["prompt"]) for r in rows}
    ref_sets = ("harmful", "harmful_fit", "xstest_unsafe", "xstest_unsafe_fit", "mj_en_dev", "mj_en_frz")
    ref = [grams(norm(r["prompt"])) for r in rows if r["set"] in ref_sets]
    def near(t):
        g = grams(norm(t))
        return any(len(g & h) / max(1, len(g | h)) >= 0.85 for h in ref if 0.7 <= len(h) / max(1, len(g)) <= 1.43)
    r_ = random.Random(9)
    by_style = collections.defaultdict(list)
    for i in range(n_all):
        t = d["turns"][i][0] if isinstance(d["turns"][i], list) else d["turns"][i]
        by_style[str(d["prompt_style"][i])].append((int(d["question_id"][i]), t, str(d["category"][i])))
    new, n, skipped = [], collections.Counter(), collections.Counter()
    for style, items in sorted(by_style.items()):
        tag = re.sub(r"[^a-z0-9]+", "", style.lower())
        dv = [it for it in items if hmod(f"sorry-{it[0]}", 5) == 0]
        fz = [it for it in items if hmod(f"sorry-{it[0]}", 5) != 0]
        r_.shuffle(dv); r_.shuffle(fz)
        for s_, part, k in (("dev", dv, 20), ("frz", fz, 40)):
            for qid, t, cat in part:
                if n[(tag, s_)] >= k:
                    break
                if not t or ntok(t) > MAXTOK:
                    skipped["too_long_or_empty"] += 1; continue
                if norm(t) in all_norm:
                    skipped["exact_duplicate_of_an_existing_row"] += 1; continue
                if near(t):
                    skipped["near_duplicate_of_a_plain_harmful_row"] += 1; continue
                n[(tag, s_)] += 1
                new.append(row(f"sorry_{tag}_{s_}", f"sorry-{qid}-{tag}", t, "sorry", 1, style=style, category=cat))
    write_jsonl(RAW / "sets.jsonl", rows + new)
    frz = [r for r in rows + new if r["set"].endswith("_frz")]
    fk = sorted(f"{r['set']}|{r['id']}" for r in frz)
    ft = sorted(f"{r['set']}|{r['id']}|{hsh(r['prompt'])}" for r in frz)
    man["sorry"] = {"added_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "rows_in_dataset": n_all, "styles": sorted(by_style), "sizes": {f"{a}_{b}": c for (a, b), c in sorted(n.items())},
                    "dev_rows": sum(c for (a, b), c in n.items() if b == "dev"), "frozen_rows": sum(c for (a, b), c in n.items() if b == "frz"), "skipped": dict(skipped),
                    "rule": "split by question id (sha1('sorry-<id>') mod 5 == 0 is development), so the 21 styles of a question share a side; 20 development and 40 frozen questions per style; candidates equal or near (3-gram Jaccard >= 0.85) to a plain harmful row already in the sets are skipped; text stays private (the licence forbids redistribution)",
                    "frozen_hash_before_sorry": man["frozen_ids_sha256"], "frozen_n_before_sorry": man["frozen_n"]}
    for r in new:
        man["ids"].setdefault(r["set"], []).append(r["id"])
        man["new_sizes"][r["set"]] = man["new_sizes"].get(r["set"], 0) + 1
    man["new_rows"] += len(new)
    man["sides"]["development"] += man["sorry"]["dev_rows"]
    man["sides"]["frozen"] = len(frz)
    man["frozen_n"], man["frozen_ids_sha256"], man["frozen_ids_and_text_sha256"] = len(frz), hashlib.sha256("\n".join(fk).encode()).hexdigest(), hashlib.sha256("\n".join(ft).encode()).hexdigest()
    (OUT / "sets_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man["sorry"], indent=1))
    print("frozen", man["frozen_n"], man["frozen_ids_sha256"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sorry":
        sorry()
    else:
        finish(*build())
