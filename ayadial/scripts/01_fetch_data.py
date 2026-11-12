"""Phase 0: build the five prompt sets and the seeded 50/50 fit/eval split."""
import json

import numpy as np
from datasets import load_dataset

from common import DATA, DAY1, SETS, load_config, write_jsonl


def rows(set_name, source, pairs):
    """pairs = (original index, prompt text)."""
    return [
        {"id": f"{source}-{i}", "prompt": text.strip(), "source": source, "set": set_name}
        for i, text in pairs
    ]


def sample(items, n, seed):
    idx = np.random.default_rng(seed).permutation(len(items))[:n]
    return [items[i] for i in sorted(idx)]


def main():
    cfg = load_config()
    seed = cfg["seed"]

    harmbench = load_dataset("walledai/HarmBench", "standard", split="train")
    jbb_harmful = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    harmful = rows("harmful", "harmbench", enumerate(harmbench["prompt"]))
    harmful += rows("harmful", "jbb", enumerate(jbb_harmful["Goal"]))
    seen, deduped = set(), []
    for row in harmful:
        key = row["prompt"].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    dropped = len(harmful) - len(deduped)

    jbb_benign = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="benign")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    # Only instructions that stand alone: a row with an "input" field needs that context.
    alpaca_pool = [(i, r["instruction"]) for i, r in enumerate(alpaca) if not r["input"].strip()]
    benign = rows("benign", "jbb", enumerate(jbb_benign["Goal"]))
    benign += rows("benign", "alpaca", sample(alpaca_pool, cfg["sizes"]["benign"] - 100, seed))

    xstest = load_dataset("walledai/XSTest", split="test")
    pool = {"safe": [], "unsafe": []}
    for i, r in enumerate(xstest):
        pool[r["label"]].append((i, r["prompt"]))

    orbench = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    or_rows = sample(list(enumerate(orbench["prompt"])), cfg["sizes"]["orbench_hard"], seed)

    built = {
        "harmful": deduped,
        "benign": benign,
        "xstest_safe": rows("xstest_safe", "xstest", pool["safe"]),
        "xstest_unsafe": rows("xstest_unsafe", "xstest", pool["unsafe"]),
        "orbench_hard": rows("orbench_hard", "orbench", or_rows),
    }

    splits = {}
    rng = np.random.default_rng(seed)
    for name in SETS:
        write_jsonl(DATA / f"{name}.jsonl", built[name])
        ids = [r["id"] for r in built[name]]
        assert len(ids) == len(set(ids)), f"duplicate ids in {name}"
        order = rng.permutation(len(ids))
        half = len(ids) // 2
        splits[name] = {
            "fit": [ids[i] for i in order[:half]],
            "eval": [ids[i] for i in order[half:]],
        }
    with open(DATA / "splits.json", "w") as f:
        json.dump(splits, f, indent=1)

    report = {
        "sizes": {k: len(v) for k, v in built.items()},
        "expected": cfg["sizes"],
        "harmful_duplicates_dropped": dropped,
        "fit_eval": {k: [len(v["fit"]), len(v["eval"])] for k, v in splits.items()},
    }
    DAY1.mkdir(parents=True, exist_ok=True)
    with open(DAY1 / "data_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
