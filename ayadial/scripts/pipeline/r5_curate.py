"""Round 5, CuratorKIT step: near-duplicates between the fit, development and frozen sides of the round-5 sets.

Same tool, same method and same threshold as r4_curate.py (CuratorKIT 0.1.23: JSONLReader -> SchemaGate ->
ExactDeduplicator -> MinHashDeduplicator, Jaccard 0.85, character 3-grams, 128 permutations, seed 42), run on
runs_private/round5/sets.jsonl (every round-5 prompt row, 91 sets, 17,059 rows). Read-only on that file.

Sides, by set name:
  fit          sets starting with "trfit", sets ending with "_fit", and jbb_GCG_pool, jbb_PAIR_pool
  frozen       sets ending with "_frz"
  development  everything else (round-4 development sets, "_dev" sets, the other "_pool" sets)

As in round 4, CuratorKIT runs as shipped with the rows in side order fit -> development -> frozen (it keeps the
first copy), and then every pair of rows is compared by brute force with CuratorKIT's own primitives
(_ngrams, _make_hash_funcs, _minhash_signature; same n-grams, permutations and seed) and CuratorKIT's exact key
(lower-cased, whitespace-collapsed), so the cross-side counts do not depend on LSH banding or on keep-first order.
Signatures for the pair search are computed in a process pool; a spot check compares them with a serial run.

Some sets are near-duplicates of other rows by construction. A pair is labelled with a construction, and kept
out of the "unexpected" counts, when:
  itw_same_template       two itw_* rows with the same template index (only possible within a side)
  mj_same_request         two mj_* rows with the same "mj-<request>" prefix (the 10 language versions)
  sfx_on_source           an sfx_* row and the row it was built from: id "sfx-<source id>" and the source
                          prompt contained in the sfx row (checked in memory)
  aug_gcg_on_source       an aug_gcg_* row and its source: id "aug-own-<source id>", source prompt contained
  jbb_same_behaviour      a jbb_* attack row and a row about the same behaviour "jbb-<n>": the plain behaviour
                          in harmful / harmful_fit, a harmful_jb wrap of it, or another jbb_* attack on it
  harmful_jb_on_source    a harmful_jb row "<id>::t<k>" and the plain request <id> in harmful / harmful_fit
  jbb_same_method_template  two jbb_* rows of the same attack method, different behaviours (added after the first
                          run: JBB's RS and JBC prompts wrap every behaviour in one fixed template, which made
                          6,120 development-vs-frozen pairs at 0.85; counted by set pair, not listed)
Everything else is "unexpected". Near-duplicate itw templates across sides are unexpected by this rule.

In-the-wild templates are also compared at template level: one text per template, made from its rows by
stripping the appended (or inserted) question, i.e. the longest common prefix plus the longest common suffix
of the template's rows (5 rows per development template, 3 per frozen template). The template texts are
compared with the same CuratorKIT MinHash settings and exact key.

Parameter changes from CuratorKIT defaults (same as round 4):
  min_tokens=1 instead of 10 (a second run with min_tokens=10 and dedup='none' counts what the default rejects);
  clean=False (audit the prompts as they are); write_hf_readme=False (nothing here is published).

Outputs (ids, set names and numbers only, never prompt or template text):
  results/round5/pipeline/curate_report.json
Local only (contains prompt text):
  runs_private/round5/pipeline/curate_input/       the three ordered input files fed to CuratorKIT
  runs_private/round5/pipeline/curatorkit_export/  CuratorKIT's own export folders and manifest

Usage (from build/): .venv-pipeline/bin/python scripts/pipeline/r5_curate.py
"""
import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import BUILD, read_jsonl, write_json  # noqa: E402

from curatorkit import Curator, CuratorConfig, DataSample, __version__ as CK_VERSION  # noqa: E402
from curatorkit.normalizers.dedup import (_make_hash_funcs, _minhash_signature, _ngrams,  # noqa: E402
                                          _sample_dedup_text)

try:
    from curatorkit.gates.schema import count_tokens  # the schema gate's own token counter
except ImportError:  # pragma: no cover
    count_tokens = None

PRIV5 = BUILD / "runs_private" / "round5"
OUT5 = BUILD / "results" / "round5" / "pipeline"
WORK = PRIV5 / "pipeline"
SETS_FILE = PRIV5 / "sets.jsonl"
MANIFEST5 = BUILD / "results" / "round5" / "sets_manifest.json"
MANIFEST4 = BUILD / "results" / "round4" / "sets_manifest.json"
R4_REPORT = BUILD / "results" / "round4" / "pipeline" / "curate_report.json"

SIDES = ("fit", "development", "frozen")
FIT_POOLS = {"jbb_GCG_pool", "jbb_PAIR_pool"}
THRESHOLD = 0.85          # CuratorKIT default, as in round 4
SENSITIVITY = [0.7, 0.5]  # extra, looser thresholds
NGRAM, NUM_PERM, SEED = 3, 128, 42
NEW_FROZEN_FAMILIES = ("wj5", "orb5", "pht", "aya", "mj")  # finding 2
CROSS = ["fit|frozen", "fit|development", "development|frozen"]
WITHIN = ["fit|fit", "development|development", "frozen|frozen"]
# labels for round-4 sets, which carry no "harm" field (1 harmful, 0 benign)
R4_HARMFUL = {"harmful", "harmful_fit", "harmful_jb", "xstest_unsafe", "xstest_unsafe_fit", "trfit_h", "dev_ho_h",
              "dev_in_h", "wj_adversarial_harmful"}
R4_BENIGN = {"benign", "benign_fit", "xstest_safe", "xstest_safe_fit", "orbench_hard", "orbench_hard_fit", "trfit_b",
             "dev_ho_b", "dev_in_b", "wj_adversarial_benign"}


def side_of(s):
    if s.startswith("trfit") or s.endswith("_fit") or s in FIT_POOLS:
        return "fit"
    if s.endswith("_frz"):
        return "frozen"
    return "development"


def family(s):
    for p in ("itw", "mj", "sfx", "aug_gcg", "jbb", "aya", "pht", "wj5", "orb5"):
        if s.startswith(p + "_"):
            return p
    return "round4"


def base_key(i):
    """The id of the thing a row was built from (template, request, behaviour, source row)."""
    if i.startswith("itw-"):
        return i.split("::")[0]
    if i.startswith("mj-"):
        return i.rsplit("-", 1)[0]
    if i.startswith("sfx-"):
        return i[4:]
    if i.startswith("aug-own-"):
        return i[8:]
    if i.startswith("aug-"):
        return i[4:]
    if "::" in i:
        return i.split("::")[0]
    return i


def label(r):
    if "harm" in r:
        return r["harm"]
    return 1 if r["set"] in R4_HARMFUL else 0 if r["set"] in R4_BENIGN else None


def norm(t):
    return " ".join(t.lower().split())


def tag(r):
    return f"{r['set']}|{r['id']}"


def pair_class(a, b):
    order = {s: k for k, s in enumerate(SIDES)}
    x, y = sorted([a, b], key=order.get)
    return f"{x}|{y}"


# ---- signatures in a process pool (CuratorKIT's own primitives) --------------------
_FUNCS = None


def _sig(text):
    global _FUNCS
    if _FUNCS is None:
        _FUNCS = _make_hash_funcs(NUM_PERM, SEED)
    return _minhash_signature(_ngrams(text.lower(), NGRAM), _FUNCS)


def signatures(texts, procs):
    with Pool(procs) as pool:
        sig = pool.map(_sig, texts, chunksize=32)
    return np.array(sig, dtype=np.uint32)  # CuratorKIT masks every value to 32 bits


def all_pairs(sig, lo_thr, ch=48):
    """Every pair i < j with estimated Jaccard >= lo_thr: (i, j, estimate)."""
    n = len(sig)
    out = []
    need = lo_thr * NUM_PERM - 1e-9
    for a in range(0, n, ch):
        b = min(n, a + ch)
        eq = (sig[a:b, None, :] == sig[None, :, :]).sum(-1)
        for ii in range(b - a):
            i = a + ii
            js = np.nonzero(eq[ii, i + 1:] >= need)[0] + i + 1
            out += [(i, int(j), float(eq[ii, j]) / NUM_PERM) for j in js]
    return out


def exact_pairs_of(keys):
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    return {(i, j) for g in groups.values() if len(g) > 1 for i, j in combinations(g, 2)}


# ---- CuratorKIT run, as shipped --------------------
def to_sample(row):
    return DataSample(source_uri=f"round5/{row['set']}", instruction=row["prompt"], task_type="prompt_only",
                      metadata={"row_id": row["id"], "set": row["set"], "boundary": side_of(row["set"])})


def write_inputs(rows):
    d = WORK / "curate_input"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for side in SIDES:
        p = d / f"{side}.jsonl"
        with open(p, "w") as f:
            for r in rows:
                if side_of(r["set"]) == side:
                    f.write(json.dumps({"id": r["id"], "set": r["set"], "prompt": r["prompt"]}) + "\n")
        paths.append(str(p))
    return paths


def run_curator(paths, min_tokens, dedup, name):
    cfg = CuratorConfig(dataset=paths, preprocessing_fn=to_sample, dedup=dedup, minhash_threshold=THRESHOLD,
                        minhash_ngram=NGRAM, minhash_num_perm=NUM_PERM, minhash_seed=SEED,
                        min_tokens=min_tokens, clean=False, write_hf_readme=False,
                        output_dir=str(WORK / "curatorkit_export" / name), name=f"round5_{name}")
    t0 = time.time()
    res = Curator(cfg).run()
    return cfg, res, time.time() - t0


def text_stats(rs):
    chars = [len(r["prompt"]) for r in rs]
    toks = [count_tokens(r["prompt"].strip()) if count_tokens else len(r["prompt"].split()) for r in rs]
    return {"n": len(rs), "chars_mean": round(statistics.mean(chars), 1), "chars_median": statistics.median(chars),
            "chars_min": min(chars), "chars_max": max(chars), "tokens_mean": round(statistics.mean(toks), 1),
            "tokens_min": min(toks), "tokens_max": max(toks), "below_10_tokens": sum(t < 10 for t in toks)}


# ---- in-the-wild templates --------------------
def common_prefix(strs):
    lo, hi = min(strs), max(strs)
    n = 0
    for x, y in zip(lo, hi):
        if x != y:
            break
        n += 1
    return lo[:n]


def template_text(prompts):
    """Longest common prefix + longest common suffix of the rows of one template (the question stripped)."""
    p = common_prefix(prompts)
    rest = [t[len(p):] for t in prompts]
    s = common_prefix([t[::-1] for t in rest])[::-1]
    return p + s


def main():
    t_start = time.time()
    OUT5.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    raw = read_jsonl(SETS_FILE)
    set_order = list(dict.fromkeys(r["set"] for r in raw))
    side_sets = {sd: [s for s in set_order if side_of(s) == sd] for sd in SIDES}
    rows = [r for sd in SIDES for r in raw if side_of(r["set"]) == sd]  # CuratorKIT order: fit -> development -> frozen
    for r in rows:
        r["id"] = str(r["id"])
    n = len(rows)
    assert len({tag(r) for r in rows}) == n, "set|id is not unique"
    side = [side_of(r["set"]) for r in rows]
    fam = [family(r["set"]) for r in rows]
    lab = [label(r) for r in rows]

    # manifest consistency (ids only): new sets vs the round-5 manifest, carried round-4 sets vs the round-4 one
    by_set = defaultdict(set)
    for r in rows:
        by_set[r["set"]].add(r["id"])
    m5 = json.loads(MANIFEST5.read_text())
    m4 = json.loads(MANIFEST4.read_text())
    ids5 = {s: set(map(str, v)) for s, v in m5["ids"].items()}
    ids4 = {s: set(map(str, v)) for s, v in m4["ids"].items()}
    carried = [s for s in set_order if s not in ids5]
    manifest_check = {
        "new_sets_in_round5_manifest": len(ids5), "new_sets_in_file": sum(s in ids5 for s in set_order),
        "new_sets_with_id_mismatch": sorted(s for s in ids5 if ids5[s] != by_set.get(s)),
        "carried_round4_sets": len(carried), "carried_rows": sum(len(by_set[s]) for s in carried),
        "carried_sets_with_id_mismatch_vs_round4_manifest": sorted(s for s in carried if ids4.get(s) != by_set[s]),
        "frozen_rows_by_side_rule": side.count("frozen"), "frozen_n_in_manifest": m5.get("frozen_n"),
        "frozen_ids_sha256_in_manifest": m5.get("frozen_ids_sha256"),
    }

    # ---- 1. CuratorKIT pipeline, as shipped --------------------
    paths = write_inputs(rows)
    cfg, res, secs = run_curator(paths, 1, "minhash", "main")
    kept = {(s.metadata["set"], str(s.metadata["row_id"])) for s in res.passed}
    rejected = [{"set": s.metadata.get("set"), "id": str(s.metadata.get("row_id")), "reason": s.rejection_reason,
                 "step": s.rejecting_step} for s in res.rejected]
    gate_rejected = {(x["set"], x["id"]) for x in rejected if x["step"] not in ("ExactDeduplicator", "MinHashDeduplicator")}
    _, res10, secs10 = run_curator(paths, 10, "none", "default_min_tokens")
    len_rej = Counter(s.metadata.get("set") for s in res10.rejected if str(s.rejection_reason).startswith("below_min_tokens"))
    man_main = WORK / "curatorkit_export" / "main" / "manifest.json"
    ck_manifest = json.loads(man_main.read_text()) if man_main.exists() else {}
    dropped = [k for k, r in enumerate(rows) if (r["set"], r["id"]) not in kept and (r["set"], r["id"]) not in gate_rejected]

    # ---- 2. every pair, with CuratorKIT's own primitives --------------------
    t0 = time.time()
    texts = [_sample_dedup_text(to_sample(r)) for r in rows]
    procs = max(1, min(8, (os.cpu_count() or 2) - 1))
    sig = signatures(texts, procs)
    funcs = _make_hash_funcs(NUM_PERM, SEED)
    spot = list(range(0, n, max(1, n // 25)))[:25]
    spot_ok = all(list(sig[k]) == _minhash_signature(_ngrams(texts[k].lower(), NGRAM), funcs) for k in spot)
    ekey = [hashlib.sha256(" ".join(t.lower().split()).encode()).hexdigest() for t in texts]
    sig_secs = time.time() - t0
    t0 = time.time()
    lo_thr = min(SENSITIVITY)
    hits = all_pairs(sig, lo_thr)
    pair_secs = time.time() - t0
    exact = exact_pairs_of(ekey)
    est = {(i, j): e for i, j, e in hits}
    for ij in exact:
        est[ij] = 1.0 if ij not in est else est[ij]

    # construction links (in memory): sfx / aug rows to the row they were built from
    by_id = defaultdict(list)
    for k, r in enumerate(rows):
        by_id[r["id"]].append(k)
    built_from = set()
    for k, r in enumerate(rows):
        if fam[k] in ("sfx", "aug_gcg") and (r["id"].startswith("sfx-") or r["id"].startswith("aug-own-")):
            for o in by_id.get(base_key(r["id"]), []):
                if o != k and norm(rows[o]["prompt"]) and norm(rows[o]["prompt"]) in norm(r["prompt"]):
                    built_from.add((min(k, o), max(k, o)))

    def construction(i, j):
        a, b = rows[i], rows[j]
        fa, fb = fam[i], fam[j]
        ka, kb = base_key(a["id"]), base_key(b["id"])
        if fa == fb == "itw" and a["template"] == b["template"]:
            return "itw_same_template"
        if fa == fb == "mj" and ka == kb:
            return "mj_same_request"
        if (i, j) in built_from:
            return "sfx_on_source" if "sfx" in (fa, fb) else "aug_gcg_on_source"
        for (x, fx, kx), (y, fy, ky) in (((a, fa, ka), (b, fb, kb)), ((b, fb, kb), (a, fa, ka))):
            if fx == "jbb" and kx == ky and (fy == "jbb" or y["set"] in ("harmful", "harmful_fit", "harmful_jb")):
                return "jbb_same_behaviour"
            if x["set"] == "harmful_jb" and y["set"] in ("harmful", "harmful_fit") and kx == y["id"]:
                return "harmful_jb_on_source"
        if fa == fb == "jbb" and a["method"] == b["method"]:
            return "jbb_same_method_template"
        return None

    r4 = json.loads(R4_REPORT.read_text())
    r4_pairs = set()
    for p in r4["all_pairs"]["pairs"] + r4["all_pairs"]["cross_boundary_or_frozen_pairs_loose"]:
        r4_pairs.add(frozenset([f"{p['a']['set']}|{p['a']['id']}", f"{p['b']['set']}|{p['b']['id']}"]))

    def pinfo(i, j, e):
        c = construction(i, j)
        bp = pair_class(side[i], side[j])
        a, b = (i, j) if side[i] == bp.split("|")[0] else (j, i)  # "a" is on the first side of side_pair
        d = {"a": tag(rows[a]), "b": tag(rows[b]), "side_pair": bp, "jaccard_estimate": round(e, 4),
             "exact": (i, j) in exact, "category": c or "unexpected", "labels": [lab[a], lab[b]],
             "set_pair": f"{rows[a]['set']}|{rows[b]['set']}"}
        if frozenset([d["a"], d["b"]]) in r4_pairs:
            d["in_round4_report"] = True
        return d

    plist = [pinfo(i, j, e) for (i, j), e in sorted(est.items())]
    thr_of = lambda p, t: p["jaccard_estimate"] >= t - 1e-12 or p["exact"]  # noqa: E731
    THRS = [THRESHOLD] + SENSITIVITY
    by_side = {}
    for bp in CROSS + WITHIN:
        ps = [p for p in plist if p["side_pair"] == bp]
        v = {"exact": sum(p["exact"] for p in ps)}
        for t in THRS:
            pt = [p for p in ps if thr_of(p, t)]
            v[f"near_ge_{t}"] = {"total": len(pt), "unexpected": sum(p["category"] == "unexpected" for p in pt),
                                 "construction": dict(sorted(Counter(p["category"] for p in pt if p["category"] != "unexpected").items()))}
        by_side[bp] = v
    by_set_pair = {}
    for t in THRS:
        c = defaultdict(lambda: {"total": 0, "unexpected": 0})
        for p in plist:
            if p["side_pair"] in CROSS and thr_of(p, t):
                k = f"{p['set_pair']}  [{p['side_pair']}]"
                c[k]["total"] += 1
                c[k]["unexpected"] += p["category"] == "unexpected"
                if p["category"] != "unexpected":
                    c[k][p["category"]] = c[k].get(p["category"], 0) + 1
        by_set_pair[f"near_ge_{t}"] = dict(sorted(c.items()))
    cross_at_thr = [p for p in plist if p["side_pair"] in CROSS and thr_of(p, THRESHOLD)
                    and p["category"] != "jbb_same_method_template"]  # those are counted in by_set_pair_cross_side
    cross_unexp_loose = sorted([p for p in plist if p["side_pair"] in CROSS and p["category"] == "unexpected"],
                               key=lambda p: (-p["jaccard_estimate"], p["a"], p["b"]))
    frozen_within_unexp = [p for p in plist if p["side_pair"] == "frozen|frozen" and p["category"] == "unexpected"
                           and thr_of(p, THRESHOLD)]

    # rows touched by cross-side pairs, per set
    touched = {t: defaultdict(lambda: {"unexpected": set(), "construction": set()}) for t in THRS}
    for p in plist:
        if p["side_pair"] not in CROSS:
            continue
        for t in THRS:
            if thr_of(p, t):
                for x in (p["a"], p["b"]):
                    touched[t][x.split("|")[0]]["unexpected" if p["category"] == "unexpected" else "construction"].add(x)
    rows_touched = {f"near_ge_{t}": {s: {"side": side_of(s), "rows_in_unexpected_pairs": len(v["unexpected"]),
                                         "rows_in_construction_pairs": len(v["construction"])}
                                     for s, v in sorted(touched[t].items())} for t in THRS}

    # ---- finding 1: in-the-wild templates, frozen vs development, template level --------------------
    tmpl_rows = defaultdict(list)
    for k, r in enumerate(rows):
        if fam[k] == "itw":
            tmpl_rows[r["template"]].append(k)
    tids = sorted(tmpl_rows)
    t_side = {t: {side[k] for k in tmpl_rows[t]} for t in tids}
    assert all(len(v) == 1 for v in t_side.values()), "a template has rows on two sides"
    t_side = {t: next(iter(v)) for t, v in t_side.items()}
    t_group = {t: rows[tmpl_rows[t][0]]["group"] for t in tids}
    t_mode = {t: rows[tmpl_rows[t][0]]["mode"] for t in tids}
    t_text = {t: template_text([rows[k]["prompt"] for k in tmpl_rows[t]]) for t in tids}
    kept_frac = [len(t_text[t]) / statistics.mean(len(rows[k]["prompt"]) for k in tmpl_rows[t]) for t in tids]
    t_sig = np.array([_minhash_signature(_ngrams(t_text[t].lower(), NGRAM), funcs) for t in tids], dtype=np.uint32)
    t_exact = exact_pairs_of([hashlib.sha256(" ".join(t_text[t].lower().split()).encode()).hexdigest() for t in tids])
    t_hits = {(i, j): e for i, j, e in all_pairs(t_sig, lo_thr)}
    for ij in t_exact:
        t_hits[ij] = 1.0
    tpairs = []
    for (i, j), e in sorted(t_hits.items()):
        ti, tj = tids[i], tids[j]
        if t_side[ti] == t_side[tj]:
            continue
        fz, dv = (ti, tj) if t_side[ti] == "frozen" else (tj, ti)
        tpairs.append({"frozen_template": fz, "development_template": dv, "jaccard_estimate": round(e, 4),
                       "exact": (i, j) in t_exact, "frozen_group_has_community_id": not t_group[fz].startswith("solo"),
                       "development_group_has_community_id": not t_group[dv].startswith("solo"),
                       "frozen_mode": t_mode[fz], "development_mode": t_mode[dv]})
    tpairs.sort(key=lambda p: (-p["jaccard_estimate"], p["frozen_template"], p["development_template"]))
    within_t = Counter()
    for (i, j), e in t_hits.items():
        if t_side[tids[i]] == t_side[tids[j]] and e >= THRESHOLD - 1e-12:
            within_t[t_side[tids[i]]] += 1
    # supplementary (not CuratorKIT's estimator): exact 3-gram set Jaccard and containment, frozen vs development templates
    t_grams = {t: _ngrams(t_text[t].lower(), NGRAM) for t in tids}

    def w8(text):
        ws = text.lower().split()
        return {tuple(ws[k:k + 8]) for k in range(max(0, len(ws) - 7))}
    t_w8 = {t: w8(t_text[t]) for t in tids}
    sup = []
    for fz in (t for t in tids if t_side[t] == "frozen"):
        for dv in (t for t in tids if t_side[t] == "development"):
            A, B = t_grams[fz], t_grams[dv]
            inter = len(A & B)
            WA, WB = t_w8[fz], t_w8[dv]
            sup.append((fz, dv, inter / max(1, len(A | B)), inter / max(1, min(len(A), len(B))),
                        len(WA & WB) / max(1, min(len(WA), len(WB)))))
    from difflib import SequenceMatcher

    def longest_run(a, b):
        x, y = t_text[a].lower(), t_text[b].lower()
        return SequenceMatcher(None, x, y, autojunk=False).find_longest_match(0, len(x), 0, len(y)).size
    partial = sorted([x for x in sup if x[4] >= 0.5], key=lambda x: -x[4])
    supplementary = {
        "what": "not CuratorKIT's estimator: exact Jaccard and containment (|A and B| / min(|A|, |B|)) of the templates' 3-gram sets, "
                "and containment of word 8-gram sets (3-gram containment is inflated by chance when one template is much shorter), "
                "every frozen x development template pair",
        "pairs": len(sup), "max_jaccard_3gram": round(max(x[2] for x in sup), 4),
        "max_containment_3gram": round(max(x[3] for x in sup), 4), "max_containment_word8": round(max(x[4] for x in sup), 4),
        **{f"pairs_jaccard_3gram_ge_{t}": sum(x[2] >= t for x in sup) for t in THRS},
        **{f"pairs_containment_3gram_ge_{t}": sum(x[3] >= t for x in sup) for t in THRS},
        **{f"pairs_containment_word8_ge_{t}": sum(x[4] >= t for x in sup) for t in THRS},
        "pairs_containment_word8_ge_0.5_list": [
            {"frozen_template": a, "development_template": b, "containment_word8": round(c8, 4), "containment_3gram": round(c3, 4),
             "jaccard_3gram": round(j, 4), "frozen_template_chars": len(t_text[a]), "development_template_chars": len(t_text[b]),
             "longest_shared_run_chars": longest_run(a, b), "frozen_group_has_community_id": not t_group[a].startswith("solo"),
             "development_group_has_community_id": not t_group[b].startswith("solo")} for a, b, j, c3, c8 in partial],
        "frozen_templates_containment_word8_ge_0.5": sorted({x[0] for x in partial}),
        "top5_by_jaccard_3gram": [{"frozen_template": a, "development_template": b, "jaccard_3gram": round(j, 4)}
                                  for a, b, j, _, _ in sorted(sup, key=lambda x: -x[2])[:5]],
    }

    def tlevel(t):
        ps = [p for p in tpairs if p["jaccard_estimate"] >= t - 1e-12 or p["exact"]]
        fz = sorted({p["frozen_template"] for p in ps})
        frows = sorted(tag(rows[k]) for tt in fz for k in tmpl_rows[tt])
        return {"template_pairs": len(ps), "frozen_templates": len(fz), "frozen_template_indices": fz,
                "development_template_indices": sorted({p["development_template"] for p in ps}),
                "frozen_rows_of_those_templates": len(frows),
                "frozen_rows_by_set": dict(Counter(x.split("|")[0] for x in frows)),
                "frozen_templates_without_community_id": sum(t_group[x].startswith("solo") for x in fz),
                "frozen_row_ids": frows}
    row_itw = [p for p in plist if p["side_pair"] == "development|frozen" and p["set_pair"].split("|")[0].startswith("itw_")
               and p["set_pair"].split("|")[1].startswith("itw_")]
    finding1 = {
        "what": "in-the-wild templates in the frozen side (itw_*_frz) near-duplicating templates in the development side (itw_*_dev)",
        "method": "one text per template: longest common prefix + longest common suffix of its rows (the question stripped); "
                  "CuratorKIT MinHash (char 3-grams, 128 perm, seed 42) brute force over every template pair, plus CuratorKIT exact key",
        "templates": {"development": sum(v == "development" for v in t_side.values()), "frozen": sum(v == "frozen" for v in t_side.values())},
        "rows_per_template": dict(Counter(f"{t_side[t]}:{len(tmpl_rows[t])}" for t in tids)),
        "template_text_chars_as_fraction_of_row_mean": round(statistics.mean(kept_frac), 3),
        "template_text_chars_as_fraction_of_row_min": round(min(kept_frac), 3),
        "template_indices_on_both_sides": len({t for t in tids if t_side[t] == "development"} & {t for t in tids if t_side[t] == "frozen"}),
        "groups_on_both_sides": len({t_group[t] for t in tids if t_side[t] == "development"} & {t_group[t] for t in tids if t_side[t] == "frozen"}),
        "templates_without_community_id": {sd: sum(t_side[t] == sd and t_group[t].startswith("solo") for t in tids) for sd in ("development", "frozen")},
        "at_threshold": tlevel(THRESHOLD),
        **{f"at_{t}": {k: v for k, v in tlevel(t).items() if k != "frozen_row_ids"} for t in SENSITIVITY},
        "template_pairs_ge_0.5": tpairs,
        "within_side_template_pairs_at_threshold": dict(within_t),
        "supplementary_exact_jaccard_and_containment": supplementary,
        "row_level_itw_dev_vs_frz_pairs": {f"near_ge_{t}": sum(thr_of(p, t) for p in row_itw) for t in THRS},
        "row_level_itw_dev_vs_frz_frozen_templates_at_threshold": sorted({int(p["b"].split("|")[1].split("::")[0][4:])
                                                                         for p in row_itw if thr_of(p, THRESHOLD)}),
    }

    # ---- finding 2: new frozen rows near a fit or development row --------------------
    f2 = {}
    for t in THRS:
        ps = [p for p in plist if p["side_pair"] in ("fit|frozen", "development|frozen") and thr_of(p, t)
              and family(p["b"].split("|")[0]) in NEW_FROZEN_FAMILIES]
        f2[f"near_ge_{t}"] = {
            "pairs": len(ps), "unexpected_pairs": sum(p["category"] == "unexpected" for p in ps),
            "frozen_rows": len({p["b"] for p in ps}),
            "frozen_rows_by_set": dict(sorted(Counter(x.split("|")[0] for x in {p["b"] for p in ps}).items())),
            "frozen_row_ids": sorted({p["b"] for p in ps}),
            "pairs_list": sorted(ps, key=lambda p: (-p["jaccard_estimate"], p["b"], p["a"])),
        }
    finding2 = {"what": "rows of wj5_*_frz, orb5_frz, pht_*_frz, aya_*_frz, mj_*_frz near-duplicating any fit or development row",
                "frozen_rows_checked": sum(1 for k in range(n) if side[k] == "frozen" and fam[k] in NEW_FROZEN_FAMILIES),
                **f2}

    # ---- the other frozen families (itw, jbb, sfx): unexpected pairs with a fit or development row ------
    fo = {}
    for t in THRS:
        ps = [p for p in plist if p["side_pair"] in ("fit|frozen", "development|frozen") and thr_of(p, t)
              and family(p["b"].split("|")[0]) not in NEW_FROZEN_FAMILIES and p["category"] == "unexpected"]
        fo[f"near_ge_{t}"] = {
            "pairs": len(ps), "frozen_rows": len({p["b"] for p in ps}),
            "frozen_rows_by_set": dict(sorted(Counter(x.split("|")[0] for x in {p["b"] for p in ps}).items())),
            "by_set_pair": dict(sorted(Counter(p["set_pair"] for p in ps).items())),
            "development_itw_template_indices": sorted({int(p["a"].split("|")[1].split("::")[0][4:]) for p in ps
                                                        if p["a"].split("|")[1].startswith("itw-")}),
            "frozen_itw_template_indices": sorted({int(p["b"].split("|")[1].split("::")[0][4:]) for p in ps
                                                   if p["b"].split("|")[1].startswith("itw-")}),
            "frozen_row_ids": sorted({p["b"] for p in ps}),
            **({"pairs_list": sorted(ps, key=lambda p: (-p["jaccard_estimate"], p["b"], p["a"]))} if t != min(SENSITIVITY) else {}),
        }
    other_frozen = {"what": "rows of itw_*_frz, jbb_*_frz, sfx_*_frz near-duplicating a fit or development row, constructions excluded",
                    "frozen_rows_checked": sum(1 for k in range(n) if side[k] == "frozen" and fam[k] not in NEW_FROZEN_FAMILIES),
                    **fo}
    same_method = defaultdict(Counter)
    for p in plist:
        if p["category"] == "jbb_same_method_template" and p["side_pair"] in CROSS and thr_of(p, THRESHOLD):
            same_method[p["side_pair"]][p["set_pair"]] += 1
    jbb_template_note = {"cross_side_pairs_at_threshold_by_set_pair": {k: dict(sorted(v.items())) for k, v in same_method.items()},
                         "frozen_rows_involved_at_threshold": dict(sorted(Counter(
                             x.split("|")[0] for x in {p["b"] for p in plist if p["category"] == "jbb_same_method_template"
                                                       and p["side_pair"] in CROSS and p["side_pair"].endswith("frozen")
                                                       and thr_of(p, THRESHOLD)}).items()))}

    # ---- finding 3: fit rows near development rows, other than the constructions --------------------
    f3 = {}
    for t in THRS:
        ps = [p for p in plist if p["side_pair"] == "fit|development" and p["category"] == "unexpected" and thr_of(p, t)]
        f3[f"near_ge_{t}"] = {
            "pairs": len(ps), "fit_rows": len({p["a"] for p in ps}), "development_rows": len({p["b"] for p in ps}),
            "same_label_pairs": sum(p["labels"][0] == p["labels"][1] for p in ps),
            "pairs_already_in_round4_report": sum(bool(p.get("in_round4_report")) for p in ps),
            "by_set_pair": dict(sorted(Counter(p["set_pair"] for p in ps).items())),
            **({"pairs_list": sorted(ps, key=lambda p: (-p["jaccard_estimate"], p["a"], p["b"]))} if t != min(SENSITIVITY) else {}),
        }
    finding3 = {"what": "fit rows near-duplicating development rows, excluding the by-construction pairs (see construction_rules)", **f3}

    # ---- CuratorKIT drops, attributed to their earliest kept partner; LSH-miss check --------------------
    kept_idx = {k for k, r in enumerate(rows) if (r["set"], r["id"]) in kept}
    partners = defaultdict(list)
    for (i, j), e in est.items():
        if e >= THRESHOLD - 1e-12 or (i, j) in exact:
            partners[j].append((i, e))
    drops = []
    for j in dropped:
        cands = sorted([(i, e) for i, e in partners.get(j, []) if i in kept_idx], key=lambda x: (-x[1], x[0]))
        i, e = cands[0] if cands else (None, None)
        drops.append({"dropped": tag(rows[j]), "dropped_side": side[j],
                      "earliest_best_kept_partner": tag(rows[i]) if i is not None else None,
                      "partner_side": side[i] if i is not None else None,
                      "jaccard_estimate": round(e, 4) if e is not None else None,
                      "crosses_side": i is not None and side[i] != side[j],
                      "category": (construction(i, j) or "unexpected") if i is not None else None})
    lsh_missed = sorted(tag(rows[j]) for j in kept_idx if any(i in kept_idx for i, _ in partners.get(j, [])))

    # ---- ids shared by sets on different sides (id level only) --------------------
    id_sets = defaultdict(set)
    for r in rows:
        id_sets[r["id"]].add(r["set"])
    shared = Counter()
    for i_, ss in id_sets.items():
        if len({side_of(s) for s in ss}) > 1:
            shared["|".join(sorted(ss))] += 1

    per_set = {}
    for s in set_order:
        rs = [r for r in rows if r["set"] == s]
        per_set[s] = {"side": side_of(s), **text_stats(rs),
                      "kept_by_curatorkit": sum((s, r["id"]) in kept for r in rs),
                      "dropped_by_curatorkit_dedup": sum(d["dropped"].split("|")[0] == s for d in drops),
                      "rejected_by_gates_min_tokens_1": sum(x[0] == s for x in gate_rejected),
                      "would_fail_default_min_tokens_10": len_rej.get(s, 0)}
    per_side = {sd: {"sets": len(side_sets[sd]), "rows": side.count(sd),
                     "kept_by_curatorkit": sum(per_set[s]["kept_by_curatorkit"] for s in side_sets[sd]),
                     "dropped_by_curatorkit_dedup": sum(per_set[s]["dropped_by_curatorkit_dedup"] for s in side_sets[sd]),
                     "would_fail_default_min_tokens_10": sum(per_set[s]["would_fail_default_min_tokens_10"] for s in side_sets[sd])}
                for sd in SIDES}

    report = {
        "curatorkit_version": CK_VERSION,
        "input": "runs_private/round5/sets.jsonl (local only, read only)",
        "rows": n, "sets": len(set_order),
        "side_rule": {"fit": "set starts with 'trfit', ends with '_fit', or is jbb_GCG_pool / jbb_PAIR_pool",
                      "frozen": "set ends with '_frz'", "development": "everything else"},
        "side_sets": side_sets,
        "manifest_check": manifest_check,
        "mode": "full: CuratorKIT on all rows, and every pair of rows compared (no restriction needed)",
        "config": {"pipeline": "JSONLReader -> SchemaGate -> ExactDeduplicator -> MinHashDeduplicator",
                   "order": "fit -> development -> frozen (CuratorKIT keeps the first copy)",
                   "near_duplicate_method": "CuratorKIT MinHash estimate of character 3-gram Jaccard (lower-cased text)",
                   "minhash_threshold": cfg.minhash_threshold, "minhash_ngram": cfg.minhash_ngram,
                   "minhash_num_perm": cfg.minhash_num_perm, "minhash_seed": cfg.minhash_seed,
                   "sensitivity_thresholds": SENSITIVITY, "min_tokens": 1, "clean": False,
                   "same_as_round4_script": "scripts/pipeline/r4_curate.py",
                   "changed_from_default": {"min_tokens": "10 -> 1 (see would_fail_default_min_tokens_10)",
                                            "clean": "True -> False (audit the prompts as they are)",
                                            "write_hf_readme": "True -> False (nothing published)"}},
        "construction_rules": {
            "itw_same_template": "two itw rows with the same template index (only within a side)",
            "mj_same_request": "two mj rows with the same mj-<request> prefix",
            "sfx_on_source": "sfx-<source id> row and its source row, source text contained in the sfx row",
            "aug_gcg_on_source": "aug-own-<source id> row and its source row, source text contained in the aug row",
            "jbb_same_behaviour": "a jbb_* row and a row about the same jbb-<n> (harmful, harmful_fit, harmful_jb or another jbb_* row)",
            "harmful_jb_on_source": "a harmful_jb <id>::t<k> row and the plain <id> in harmful / harmful_fit",
            "jbb_same_method_template": "two jbb_* rows of the same attack method about different behaviours. Not in the "
                                        "brief's list: added after the first run, because JBB's RS and JBC prompts wrap every "
                                        "behaviour in one fixed published template, so they near-duplicate each other across "
                                        "sides by construction (behaviours themselves are split by side)",
            "unexpected": "every other pair"},
        "timing_seconds": {"curatorkit_run": round(secs, 1), "curatorkit_default_min_tokens_run": round(secs10, 1),
                           "signatures_pool": round(sig_secs, 1), "pool_processes": procs, "pair_search": round(pair_secs, 1),
                           "total": None},
        "signature_spot_check_vs_serial": {"rows": len(spot), "all_equal": spot_ok},
        "curatorkit_stage_counts": res.stage_counts,
        "curatorkit_manifest": {k: ck_manifest.get(k) for k in ("pipeline_config_hash", "stage_counts", "rejected_breakdown",
                                                               "dedup_stats", "minhash_threshold", "wall_clock_seconds", "tool_versions")},
        "curatorkit_default_min_tokens_stage_counts": res10.stage_counts,
        "per_side": per_side,
        "all_pairs": {
            "method": "brute force over every pair, CuratorKIT MinHash signature (128 perm, char 3-grams, seed 42) and CuratorKIT exact key",
            "pairs_examined": n * (n - 1) // 2, "threshold": THRESHOLD, "sensitivity_thresholds": SENSITIVITY,
            "by_side_pair": by_side,
            "by_set_pair_cross_side": by_set_pair,
            "rows_in_cross_side_pairs_by_set": rows_touched,
            "cross_side_pairs_at_threshold": sorted(cross_at_thr, key=lambda p: (p["side_pair"], p["category"], -p["jaccard_estimate"], p["a"], p["b"])),
            "cross_side_unexpected_pairs_ge_0.5": cross_unexp_loose,
            "within_frozen_unexpected_pairs_at_threshold": len(frozen_within_unexp),
            "within_frozen_unexpected_pairs_at_threshold_by_set_pair": dict(sorted(Counter(p["set_pair"] for p in frozen_within_unexp).items())),
        },
        "findings": {"1_itw_frozen_templates_near_development_templates": finding1,
                     "2_new_frozen_rows_near_fit_or_development": finding2,
                     "3_fit_rows_near_development_rows_unexpected": finding3,
                     "other_frozen_families_near_fit_or_development": other_frozen,
                     "jbb_same_method_template_cross_side": jbb_template_note},
        "curatorkit_drops": drops,
        "curatorkit_drops_crossing_side": sum(d["crosses_side"] for d in drops),
        "curatorkit_drops_without_a_kept_partner_in_brute_force": sum(d["earliest_best_kept_partner"] is None for d in drops),
        "curatorkit_kept_rows_with_an_earlier_kept_partner_lsh_missed": lsh_missed,
        "gate_rejections_min_tokens_1": [x for x in rejected if (x["set"], x["id"]) in gate_rejected],
        "ids_shared_by_sets_on_different_sides": dict(sorted(shared.items())),
        "per_set": per_set,
        "export_dir_local_only": "runs_private/round5/pipeline/curatorkit_export",
    }
    report["timing_seconds"]["total"] = round(time.time() - t_start, 1)
    write_json(OUT5 / "curate_report.json", report)

    print(f"curatorkit {CK_VERSION}: {n} rows, kept {len(kept)}, stage counts {json.dumps(res.stage_counts)}")
    print(f"timing {json.dumps(report['timing_seconds'])}; spot check {spot_ok}")
    for bp in CROSS + WITHIN:
        v = by_side[bp]
        print(f"  {bp:26s} exact {v['exact']}; " + "; ".join(f">={t}: {v[f'near_ge_{t}']['total']} ({v[f'near_ge_{t}']['unexpected']} unexpected)" for t in THRS))
    a = finding1["at_threshold"]
    print(f"finding 1: {a['template_pairs']} frozen-vs-dev template pairs at {THRESHOLD}, {a['frozen_templates']} frozen templates, {a['frozen_rows_of_those_templates']} frozen rows")
    b = finding2[f"near_ge_{THRESHOLD}"]
    print(f"finding 2: {b['pairs']} pairs, {b['frozen_rows']} new-frozen rows at {THRESHOLD} ({b['unexpected_pairs']} unexpected pairs)")
    c = finding3[f"near_ge_{THRESHOLD}"]
    print(f"finding 3: {c['pairs']} unexpected fit-vs-dev pairs at {THRESHOLD} ({c['pairs_already_in_round4_report']} already in round 4)")
    o = other_frozen[f"near_ge_{THRESHOLD}"]
    print(f"other frozen families: {o['pairs']} unexpected pairs, {o['frozen_rows']} frozen rows at {THRESHOLD}: {o['by_set_pair']}")
    print(f"curatorkit drops {len(drops)}, crossing a side {report['curatorkit_drops_crossing_side']}, "
          f"without partner {report['curatorkit_drops_without_a_kept_partner_in_brute_force']}, lsh missed {len(lsh_missed)}")


if __name__ == "__main__":
    main()
