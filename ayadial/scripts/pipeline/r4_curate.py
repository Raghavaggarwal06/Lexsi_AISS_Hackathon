"""Round 4, CuratorKIT step: audit the round-4 prompt splits for duplicates across fit / development / frozen.

Same tool and same pipeline as p1_curate.py (CuratorKIT 0.1.23: JSONLReader -> SchemaGate ->
ExactDeduplicator -> MinHashDeduplicator, Jaccard 0.85, character 3-grams, 128 permutations, seed 42),
run on runs_private/round4/sets.jsonl (every round-4 prompt row, 26 sets).

Order matters for CuratorKIT, which keeps the first copy of a duplicate and drops later ones. The rows
go in boundary order fit -> development -> frozen, so a dropped development or frozen row is a prompt
that already exists on the fit side (or earlier on its own side): a drop that crosses a boundary is a
leak into the evaluation side.

CuratorKIT's dedup is order-dependent and only says "dropped", not "dropped because of which row". The
main question (are there duplicates ACROSS the boundaries?) needs every pair, not just the first-kept
copy, so the script also computes every pair's MinHash Jaccard estimate with CuratorKIT's own primitives
(_ngrams, _make_hash_funcs, _minhash_signature: same n-grams, permutations and seed as the run) and
CuratorKIT's own exact-dedup key (lower-cased, whitespace-collapsed). The pair search is brute force over
all ~53M pairs in numpy, so it does not depend on LSH banding luck.

Parameter changes from CuratorKIT defaults:
  min_tokens=1 instead of 10: the default would reject short XSTest/benign prompts (round 3 finding);
  a second run with min_tokens=10 and dedup='none' reports which ids the default would drop.
  clean=False: the text cleaner rewrites prompts; we audit them as they are.
  write_hf_readme=False: nothing here is published.

Outputs (ids, counts, numbers only):
  results/round4/pipeline/curate_report.json   stats, CuratorKIT stage counts, drops, cross-boundary pairs
Local only (contains prompt text):
  runs_private/round4/pipeline/curate_input/      the three ordered input files fed to CuratorKIT
  runs_private/round4/pipeline/curatorkit_export/ CuratorKIT's own export folders and manifest

Usage (from build/): .venv-pipeline/bin/python scripts/pipeline/r4_curate.py
"""
import hashlib
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
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

PRIV4 = BUILD / "runs_private" / "round4"
OUT4 = BUILD / "results" / "round4" / "pipeline"
WORK = PRIV4 / "pipeline"
SETS_FILE = PRIV4 / "sets.jsonl"
MANIFEST = BUILD / "results" / "round4" / "sets_manifest.json"

BOUNDARY_SETS = {
    "fit": ["trfit_h", "trfit_b", "harmful_fit", "xstest_unsafe_fit", "benign_fit", "xstest_safe_fit", "orbench_hard_fit"],
    "development": ["wj_adversarial_harmful", "wj_adversarial_benign", "dev_ho_h", "dev_ho_b", "dev_in_h", "dev_in_b",
                    "harmful", "harmful_jb", "xstest_unsafe", "xstest_safe", "orbench_hard", "benign"],
    "frozen": ["frozen_wjh", "frozen_ho_h", "frozen_in_h", "frozen_ho_b", "frozen_in_b", "frozen_orbench", "frozen_benign"],
}
BOUNDARY = {s: b for b, ss in BOUNDARY_SETS.items() for s in ss}
ORDER = [s for b in ("fit", "development", "frozen") for s in BOUNDARY_SETS[b]]
THRESHOLD = 0.85          # CuratorKIT default
SENSITIVITY = [0.7, 0.5]  # extra, looser thresholds, counts only
NGRAM, NUM_PERM, SEED = 3, 128, 42


def to_sample(row):
    return DataSample(source_uri=f"round4/{row['set']}", instruction=row["prompt"], task_type="prompt_only",
                      metadata={"row_id": row["id"], "set": row["set"], "boundary": BOUNDARY[row["set"]]})


def ident(s):
    return (s.metadata["set"], s.metadata["row_id"])


def write_inputs(rows):
    """Three ordered JSONL files (fit, development, frozen) under the local-only folder."""
    d = WORK / "curate_input"
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for b in ("fit", "development", "frozen"):
        p = d / f"{b}.jsonl"
        with open(p, "w") as f:
            for s in BOUNDARY_SETS[b]:
                for r in rows:
                    if r["set"] == s:
                        f.write(json.dumps({"id": r["id"], "set": r["set"], "prompt": r["prompt"]}) + "\n")
        paths.append(str(p))
    return paths


def run(paths, min_tokens, dedup, tag):
    cfg = CuratorConfig(dataset=paths, preprocessing_fn=to_sample, dedup=dedup, minhash_threshold=THRESHOLD,
                        minhash_ngram=NGRAM, minhash_num_perm=NUM_PERM, minhash_seed=SEED,
                        min_tokens=min_tokens, clean=False, write_hf_readme=False,
                        output_dir=str(WORK / "curatorkit_export" / tag), name=f"round4_{tag}")
    t0 = time.time()
    res = Curator(cfg).run()
    return cfg, res, time.time() - t0


def text_stats(rows):
    chars = [len(r["prompt"]) for r in rows]
    toks = [count_tokens(r["prompt"].strip()) if count_tokens else len(r["prompt"].split()) for r in rows]
    return {"n": len(rows), "chars_mean": round(statistics.mean(chars), 1), "chars_median": statistics.median(chars),
            "chars_min": min(chars), "chars_max": max(chars), "tokens_mean": round(statistics.mean(toks), 1),
            "tokens_min": min(toks), "tokens_max": max(toks), "below_10_tokens": sum(t < 10 for t in toks)}


def pair_class(a, b):
    """Boundary pair label, in a fixed order."""
    order = {"fit": 0, "development": 1, "frozen": 2}
    x, y = sorted([a, b], key=order.get)
    return f"{x}|{y}"


def main():
    OUT4.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(SETS_FILE)
    unknown = sorted({r["set"] for r in rows} - set(BOUNDARY))
    assert not unknown, f"sets without a boundary: {unknown}"
    rows = [r for s in ORDER for r in rows if r["set"] == s]  # boundary order, as fed to CuratorKIT

    # manifest consistency (ids only)
    man = json.loads(MANIFEST.read_text())
    man_ids = {s: set(map(str, v)) for s, v in man["ids"].items()}
    by_set = defaultdict(set)
    for r in rows:
        by_set[r["set"]].add(str(r["id"]))
    manifest_check = {"sets_in_manifest": len(man_ids), "sets_in_file": len(by_set),
                      "sets_with_id_mismatch": sorted(s for s in set(man_ids) | set(by_set) if man_ids.get(s) != by_set.get(s))}

    # ---- 1. CuratorKIT pipeline, as shipped --------------------------------------------------
    paths = write_inputs(rows)
    cfg, res, secs = run(paths, 1, "minhash", "main")
    kept = {ident(s) for s in res.passed}
    rejected = [{"set": s.metadata.get("set"), "id": s.metadata.get("row_id"), "reason": s.rejection_reason, "step": s.rejecting_step}
                for s in res.rejected]
    _, res10, secs10 = run(paths, 10, "none", "default_min_tokens")
    len_rejects = [{"set": s.metadata.get("set"), "id": s.metadata.get("row_id"), "tokens": int(s.rejection_reason.split(":")[1])}
                   for s in res10.rejected if s.rejection_reason.startswith("below_min_tokens")]
    man_main = WORK / "curatorkit_export" / "main" / "manifest.json"
    ck_manifest = json.loads(man_main.read_text()) if man_main.exists() else {}
    dropped_rows = [(r["set"], r["id"]) for r in rows if (r["set"], r["id"]) not in kept
                    and (r["set"], r["id"]) not in {(x["set"], x["id"]) for x in rejected if x["step"] != "MinHashDeduplicator" and x["step"] != "ExactDeduplicator"}]

    # ---- 2. every pair, with CuratorKIT's own primitives ---------------------------------------
    t0 = time.time()
    funcs = _make_hash_funcs(NUM_PERM, SEED)
    samples = [to_sample(r) for r in rows]
    texts = [_sample_dedup_text(s) for s in samples]
    sig = np.array([_minhash_signature(_ngrams(t.lower(), NGRAM), funcs) for t in texts], dtype=np.uint64)
    exact_key = [hashlib.sha256(" ".join(t.lower().split()).encode()).hexdigest() for t in texts]
    sig_secs = time.time() - t0

    n = len(rows)
    lo_thr = min(SENSITIVITY)
    hits = []  # (i, j, est) with est >= lo_thr, i < j
    t0 = time.time()
    CH = 48
    for a in range(0, n, CH):
        b = min(n, a + CH)
        eq = (sig[a:b, None, :] == sig[None, :, :]).sum(-1)  # [CH, n]
        for ii in range(b - a):
            i = a + ii
            js = np.nonzero(eq[ii, i + 1:] >= lo_thr * NUM_PERM - 1e-9)[0] + i + 1
            hits += [(i, int(j), float(eq[ii, j]) / NUM_PERM) for j in js]
    pair_secs = time.time() - t0
    exact_groups = defaultdict(list)
    for i, k in enumerate(exact_key):
        exact_groups[k].append(i)
    exact_pairs = {(i, j) for g in exact_groups.values() if len(g) > 1 for i, j in combinations(g, 2)}

    def info(i):
        r = rows[i]
        return {"set": r["set"], "id": r["id"], "boundary": BOUNDARY[r["set"]]}

    pairs = []
    for i, j, est in hits:
        if est >= THRESHOLD - 1e-12 or (i, j) in exact_pairs:
            pairs.append({"a": info(i), "b": info(j), "exact": (i, j) in exact_pairs, "jaccard_estimate": round(est, 4),
                          "boundary_pair": pair_class(BOUNDARY[rows[i]["set"]], BOUNDARY[rows[j]["set"]]),
                          "set_pair": "|".join(sorted([rows[i]["set"], rows[j]["set"]]))})
    hit_set = {(i, j) for i, j, _ in hits}
    for i, j in sorted(exact_pairs - hit_set):  # exact pairs always collide, kept for safety
        pairs.append({"a": info(i), "b": info(j), "exact": True, "jaccard_estimate": 1.0,
                      "boundary_pair": pair_class(BOUNDARY[rows[i]["set"]], BOUNDARY[rows[j]["set"]]),
                      "set_pair": "|".join(sorted([rows[i]["set"], rows[j]["set"]]))})

    BPAIRS = ["fit|development", "fit|frozen", "development|frozen", "fit|fit", "development|development", "frozen|frozen"]
    by_boundary = {bp: {"exact": 0, f"near_ge_{THRESHOLD}": 0, **{f"near_ge_{t}": 0 for t in SENSITIVITY}} for bp in BPAIRS}
    for i, j, est in hits:
        bp = pair_class(BOUNDARY[rows[i]["set"]], BOUNDARY[rows[j]["set"]])
        for t in [THRESHOLD] + SENSITIVITY:
            if est >= t - 1e-12:
                by_boundary[bp][f"near_ge_{t}"] += 1
    for i, j in exact_pairs:
        by_boundary[pair_class(BOUNDARY[rows[i]["set"]], BOUNDARY[rows[j]["set"]])]["exact"] += 1
    by_set_pair = Counter(p["set_pair"] + "  [" + p["boundary_pair"] + "]" for p in pairs)
    # looser thresholds: every pair that crosses a boundary or sits inside the frozen set, by id
    loose = []
    for i, j, est in hits:
        bi, bj = BOUNDARY[rows[i]["set"]], BOUNDARY[rows[j]["set"]]
        if bi != bj or bi == "frozen":
            loose.append({"a": info(i), "b": info(j), "jaccard_estimate": round(est, 4), "boundary_pair": pair_class(bi, bj),
                          "set_pair": "|".join(sorted([rows[i]["set"], rows[j]["set"]]))})
    loose_by_set_pair = {f"near_ge_{t}": dict(sorted(Counter(p["set_pair"] + "  [" + p["boundary_pair"] + "]" for p in loose
                                                               if p["jaccard_estimate"] >= t - 1e-12).items())) for t in SENSITIVITY}
    rows_in_cross_pairs = defaultdict(set)
    for p in pairs:
        if p["a"]["boundary"] != p["b"]["boundary"]:
            for side in ("a", "b"):
                rows_in_cross_pairs[p[side]["set"]].add(p[side]["id"])
    frozen_ids_touched = {s: len(v) for s, v in rows_in_cross_pairs.items() if BOUNDARY[s] == "frozen"}
    within_frozen_ids = defaultdict(set)
    for p in pairs:
        if p["boundary_pair"] == "frozen|frozen":
            for side in ("a", "b"):
                within_frozen_ids[p[side]["set"]].add(p[side]["id"])

    # ---- 3. CuratorKIT drops, attributed to their earliest partner -------------------------------
    pos = {(r["set"], r["id"]): i for i, r in enumerate(rows)}
    partner_of = defaultdict(list)
    for i, j, est in hits:
        if est >= THRESHOLD - 1e-12 or (i, j) in exact_pairs:
            partner_of[j].append((i, est))
    for i, j in exact_pairs:
        partner_of[j].append((i, 1.0))
    drops = []
    for key_ in dropped_rows:
        j = pos[key_]
        cands = sorted(partner_of.get(j, []), key=lambda x: (-x[1], x[0]))
        ii, est = cands[0] if cands else (None, None)
        drops.append({"dropped": info(j), "earliest_best_partner": info(ii) if ii is not None else None,
                      "jaccard_estimate": round(est, 4) if est is not None else None,
                      "crosses_boundary": ii is not None and BOUNDARY[rows[ii]["set"]] != BOUNDARY[rows[j]["set"]]})

    per_set = {}
    for s in ORDER:
        rs = [r for r in rows if r["set"] == s]
        per_set[s] = {"boundary": BOUNDARY[s], **text_stats(rs),
                      "kept_by_curatorkit": sum((s, r["id"]) in kept for r in rs),
                      "dropped_by_curatorkit_dedup": sum(d["dropped"]["set"] == s for d in drops),
                      "rejected_by_gates_min_tokens_1": sum(x["set"] == s and x["step"] not in ("ExactDeduplicator", "MinHashDeduplicator") for x in rejected),
                      "would_fail_default_min_tokens_10": sum(x["set"] == s for x in len_rejects),
                      "ids_in_cross_boundary_pairs": len(rows_in_cross_pairs.get(s, ()))}
    per_boundary = {b: {"sets": len(ss), "rows": sum(per_set[s]["n"] for s in ss),
                        "kept_by_curatorkit": sum(per_set[s]["kept_by_curatorkit"] for s in ss),
                        "dropped_by_curatorkit_dedup": sum(per_set[s]["dropped_by_curatorkit_dedup"] for s in ss),
                        "would_fail_default_min_tokens_10": sum(per_set[s]["would_fail_default_min_tokens_10"] for s in ss)}
                    for b, ss in BOUNDARY_SETS.items()}

    shared_ids = Counter()
    id_sets = defaultdict(set)
    for r in rows:
        id_sets[str(r["id"])].add(r["set"])
    for i_, ss in id_sets.items():
        if len(ss) > 1:
            shared_ids["|".join(sorted(ss))] += 1

    report = {
        "curatorkit_version": CK_VERSION,
        "input": "runs_private/round4/sets.jsonl (local only)",
        "rows": len(rows), "sets": len(ORDER), "boundary_sets": BOUNDARY_SETS,
        "manifest_check": manifest_check,
        "config": {"pipeline": "JSONLReader -> SchemaGate -> ExactDeduplicator -> MinHashDeduplicator",
                   "order": "fit -> development -> frozen (CuratorKIT keeps the first copy)", "set_order": ORDER,
                   "minhash_threshold": cfg.minhash_threshold, "minhash_ngram": cfg.minhash_ngram,
                   "minhash_num_perm": cfg.minhash_num_perm, "minhash_seed": cfg.minhash_seed,
                   "min_tokens": 1, "max_tokens": 2048, "clean": False,
                   "changed_from_default": {"min_tokens": "10 -> 1 (default drops short prompts; see would_fail_default_min_tokens_10)",
                                            "clean": "True -> False (audit the prompts as they are)",
                                            "write_hf_readme": "True -> False (nothing published)"}},
        "curatorkit_run_seconds": round(secs, 1), "curatorkit_default_min_tokens_run_seconds": round(secs10, 1),
        "curatorkit_stage_counts": res.stage_counts,
        "curatorkit_manifest": {k: ck_manifest.get(k) for k in ("pipeline_config_hash", "stage_counts", "rejected_breakdown", "dedup_stats",
                                                               "minhash_threshold", "wall_clock_seconds", "tool_versions")},
        "curatorkit_default_min_tokens_stage_counts": res10.stage_counts,
        "per_boundary": per_boundary,
        "per_set": per_set,
        "all_pairs": {"method": "brute force over every pair, CuratorKIT MinHash signature (128 perm, char 3-grams, seed 42) and CuratorKIT exact key",
                      "pairs_examined": n * (n - 1) // 2, "signature_seconds": round(sig_secs, 1), "pair_seconds": round(pair_secs, 1),
                      "threshold": THRESHOLD, "sensitivity_thresholds": SENSITIVITY,
                      "by_boundary_pair": by_boundary,
                      "by_set_pair_at_threshold": dict(sorted(by_set_pair.items())),
                      "cross_boundary_pairs_at_threshold": sum(p["a"]["boundary"] != p["b"]["boundary"] for p in pairs),
                      "frozen_ids_in_cross_boundary_pairs": frozen_ids_touched,
                      "frozen_ids_in_within_frozen_pairs": {s: len(v) for s, v in within_frozen_ids.items()},
                      "pairs": pairs,
                      "cross_boundary_or_frozen_pairs_by_set_pair_loose": loose_by_set_pair,
                      "cross_boundary_or_frozen_pairs_loose": sorted(loose, key=lambda p: -p["jaccard_estimate"])},
        "curatorkit_drops": drops,
        "curatorkit_drops_crossing_boundary": sum(d["crosses_boundary"] for d in drops),
        "gate_rejections_min_tokens_1": [x for x in rejected if x["step"] not in ("ExactDeduplicator", "MinHashDeduplicator")],
        "default_min_tokens_10_rejections": len_rejects,
        "ids_shared_by_several_sets": dict(shared_ids),
        "export_dir_local_only": "runs_private/round4/pipeline/curatorkit_export",
    }
    write_json(OUT4 / "curate_report.json", report)
    print(f"curatorkit {CK_VERSION}: {len(rows)} rows, kept {len(kept)}, stage counts {json.dumps(res.stage_counts)}")
    print(f"run {secs:.1f}s, default-min-tokens run {secs10:.1f}s, signatures {sig_secs:.1f}s, pairs {pair_secs:.1f}s")
    for bp, v in by_boundary.items():
        print(f"  {bp:26s} {v}")
    print(f"pairs at {THRESHOLD} or exact: {len(pairs)}; crossing a boundary: {report['all_pairs']['cross_boundary_pairs_at_threshold']}")
    print(f"curatorkit drops: {len(drops)}, crossing a boundary: {report['curatorkit_drops_crossing_boundary']}")
    print(f"default min_tokens=10 would reject {len(len_rejects)}")


if __name__ == "__main__":
    main()
