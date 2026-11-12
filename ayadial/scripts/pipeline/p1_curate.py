"""Step 1: run our five prompt sets through CuratorKIT for schema checks and duplicate removal.

CuratorKIT (https://github.com/Lexsi-Labs/CuratorKIT, PyPI `curatorkit`) is used as a curation
pipeline, not a generator: JSONLReader -> SchemaGate -> ExactDeduplicator -> MinHashDeduplicator.
Each row becomes a `prompt_only` DataSample whose metadata carries our id, set and fit/eval split,
so every accepted or dropped row can be traced back by id.

The sets go in this order: harmful, xstest_unsafe, benign, xstest_safe, orbench_hard. CuratorKIT
keeps the first copy of a duplicate and drops later ones, so the order decides which id survives.
For every dropped row the script finds the earlier row it duplicates, with CuratorKIT's own MinHash
primitives (same n-grams, permutations and seed), and records whether the pair crosses sets or
crosses the fit/eval split (a leak for phase-1 AUROC and phase-2 rates).

A second run with CuratorKIT's default min_tokens=10 reports which ids the default length gate
would reject. The main run uses min_tokens=1, because short XSTest prompts are real test items.

Outputs:
  results/round3/block8/curate_report.json  counts, dropped ids with partner ids, gate rejections (ids only)
  results/round3/block8/curated_ids.json    kept ids per set and split; steps 2 and 3 can filter on it
  build/runs_private/curatorkit_export/     CuratorKIT's own export folders (they contain prompt text,
                                            so they stay in the local-only folder)

Usage (from build/): .venv-pipeline/bin/python scripts/pipeline/p1_curate.py [--threshold 0.85]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import CURATED_EXPORT, DATA, OUT, load_set, load_splits, write_json  # noqa: E402

from curatorkit import Curator, CuratorConfig, DataSample, __version__ as CK_VERSION  # noqa: E402
from curatorkit.normalizers.dedup import (_jaccard_estimate, _make_hash_funcs, _minhash_signature,  # noqa: E402
                                          _ngrams, _sample_dedup_text)

ORDER = ["harmful", "xstest_unsafe", "benign", "xstest_safe", "orbench_hard"]


def split_of():
    out = {}
    for name, halves in load_splits().items():
        for half, ids in halves.items():
            for i in ids:
                out[(name, i)] = half
    return out


SPLIT = split_of()


def to_sample(row):
    """Our row -> a prompt_only DataSample. Returning a DataSample skips CuratorKIT's format detection."""
    return DataSample(
        source_uri=f"build/data/{row['set']}.jsonl",
        instruction=row["prompt"],
        task_type="prompt_only",
        metadata={"row_id": row["id"], "set": row["set"], "source": row["source"],
                  "split": SPLIT.get((row["set"], row["id"]), "none")},
    )


def run(threshold, min_tokens, tag):
    cfg = CuratorConfig(
        dataset=[str(DATA / f"{name}.jsonl") for name in ORDER],
        preprocessing_fn=to_sample,
        dedup="minhash", minhash_threshold=threshold,
        min_tokens=min_tokens, clean=False, write_hf_readme=False,
        output_dir=str(CURATED_EXPORT / tag), name=f"block8_{tag}",
    )
    return cfg, Curator(cfg).run()


def ident(sample):
    return (sample.metadata["set"], sample.metadata["row_id"])


def partners(all_rows, kept, cfg):
    """For each dropped row, the earlier kept-or-dropped row it is closest to, by CuratorKIT's MinHash estimate."""
    funcs = _make_hash_funcs(cfg.minhash_num_perm, cfg.minhash_seed)
    samples = [to_sample(r) for r in all_rows]
    sigs = [_minhash_signature(_ngrams(_sample_dedup_text(s).lower(), cfg.minhash_ngram), funcs) for s in samples]
    exact = [" ".join(_sample_dedup_text(s).lower().split()) for s in samples]
    out = []
    for i, s in enumerate(samples):
        if ident(s) in kept:
            continue
        best = max(range(i), key=lambda j: (exact[j] == exact[i], _jaccard_estimate(sigs[i], sigs[j])), default=None)
        if best is None:
            continue
        p = samples[best]
        out.append({
            "dropped": {"set": s.metadata["set"], "id": s.metadata["row_id"], "split": s.metadata["split"]},
            "kept_or_earlier": {"set": p.metadata["set"], "id": p.metadata["row_id"], "split": p.metadata["split"]},
            "exact_match": exact[best] == exact[i],
            "jaccard_estimate": round(_jaccard_estimate(sigs[i], sigs[best]), 4),
            "crosses_sets": s.metadata["set"] != p.metadata["set"],
            "crosses_fit_eval": s.metadata["split"] != p.metadata["split"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.85, help="MinHash Jaccard threshold (CuratorKIT default 0.85)")
    args = ap.parse_args()
    all_rows = [r for name in ORDER for r in load_set(name)]

    cfg, res = run(args.threshold, 1, "main")
    kept = {ident(s) for s in res.passed}
    rejected = [{"set": s.metadata.get("set"), "id": s.metadata.get("row_id"), "reason": s.rejection_reason, "step": s.rejecting_step} for s in res.rejected]
    pairs = partners(all_rows, kept, cfg)

    _, res_default = run(args.threshold, 10, "default_min_tokens")
    length_rejects = [{"set": s.metadata.get("set"), "id": s.metadata.get("row_id"), "reason": s.rejection_reason.split(":")[0]}
                      for s in res_default.rejected if s.rejection_reason.startswith("below_min_tokens")]

    per_set = {}
    for name in ORDER:
        rows = [r for r in all_rows if r["set"] == name]
        per_set[name] = {
            "rows_in": len(rows),
            "kept": sum((name, r["id"]) in kept for r in rows),
            "dropped_as_duplicate": sum(p["dropped"]["set"] == name for p in pairs),
            "dropped_eval_half": sum(p["dropped"]["set"] == name and p["dropped"]["split"] == "eval" for p in pairs),
            "rejected_by_gates": sum(r["set"] == name for r in rejected),
            "would_fail_default_min_tokens_10": sum(r["set"] == name for r in length_rejects),
        }
    report = {
        "curatorkit_version": CK_VERSION,
        "config": {"order": ORDER, "dedup": "exact then minhash", "minhash_threshold": cfg.minhash_threshold,
                   "minhash_ngram": cfg.minhash_ngram, "minhash_num_perm": cfg.minhash_num_perm,
                   "minhash_seed": cfg.minhash_seed, "min_tokens": 1, "clean": False},
        "stage_counts": res.stage_counts,
        "per_set": per_set,
        "duplicate_pairs": pairs,
        "pairs_crossing_sets": sum(p["crosses_sets"] for p in pairs),
        "pairs_crossing_fit_eval": sum(p["crosses_fit_eval"] for p in pairs),
        "gate_rejections": rejected,
        "default_min_tokens_rejections": length_rejects,
        "export_dir_local_only": str(CURATED_EXPORT.relative_to(DATA.parent)),
    }
    write_json(OUT / "curate_report.json", report)

    curated = {name: {"fit": [], "eval": []} for name in ORDER}
    for r in all_rows:
        half = SPLIT.get((r["set"], r["id"]))
        if (r["set"], r["id"]) in kept and half in ("fit", "eval"):
            curated[r["set"]][half].append(r["id"])
    write_json(OUT / "curated_ids.json", {"threshold": cfg.minhash_threshold, "curatorkit_version": CK_VERSION, "ids": curated})

    for name, row in per_set.items():
        print(name, row)
    print(f"duplicate pairs: {len(pairs)}, crossing sets: {report['pairs_crossing_sets']}, crossing fit/eval: {report['pairs_crossing_fit_eval']}")


if __name__ == "__main__":
    main()
