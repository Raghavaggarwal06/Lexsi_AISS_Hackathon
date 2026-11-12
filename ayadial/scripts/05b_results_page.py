"""Phase 2 results page: writes results/knob_results.html, one self-contained file.

Each --source is LABEL=PATH, where PATH holds runs/ and results/phase2/. Pass several to get a run
switcher. Everything is recomputed with 05_score.py's own functions, so the numbers match it, and
nothing is written back to the source folders. Harmful prompt text stays out of the shareable
page; --include-harmful writes a private copy to runs/, which is never published.
"""
import argparse
import importlib.util
import json
from pathlib import Path

from common import DATA, RESULTS, ROOT, load_set

SPEC = importlib.util.spec_from_file_location("score", Path(__file__).parent / "05_score.py")
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)
WITHHELD = ("harmful", "xstest_unsafe")


def level(value):
    return f"{value:g}"


def build_run(label, path, include_harmful=False):
    score.ROOT = path
    runs = score.load_runs()
    cache = {r["key"]: r for r in score.read_jsonl(path / "runs" / "scores.jsonl")}
    table, vectors = score.build_table(runs, cache)
    rules = score.go_rules(table, vectors)

    cells, wall, scales = {}, {}, {}
    for r in table:
        method, name = r["method"], r["set"]
        cells[f"{method}|{level(r['scale'])}|{name}"] = {
            "r": r["refusal"], "ci": r["refusal_ci"], "u": r["unsafe"], "deg": r["degenerate_rate"], "n": r["n"],
        }
        scales.setdefault(method, set()).add(r["scale"])
    for (method, scale, name), vector in vectors.items():
        states = "".join("2" if v != v else str(int(v)) for v in vector)
        wall.setdefault(f"{method}|{level(scale)}", {})[name] = states

    base_rows = next(rows for method, _, rows in runs if method == "base")
    shown = score.SETS if include_harmful else score.HARMLESS
    text = {name: [r["prompt"] for r in base_rows if r["set"] == name] for name in shown}
    phase2 = path / "results" / "phase2"
    config = json.loads((phase2 / "sweep_config.json").read_text())
    firing_path = phase2 / "gate_firing.json"
    firing = None
    if firing_path.exists():
        log = json.loads(firing_path.read_text())
        firing = log[next(iter(log))]
    methods = [m for m in score.METHODS if m in scales]
    return score.clean_json({
        "label": label,
        "variant": config.get("variant"), "layer": config.get("layer"),
        "sizes": {r["set"]: r["n"] for r in table if r["method"] == "base"},
        "methods": methods,
        "scales": {m: [0.0] + sorted(s for s in scales[m] if s != 0) for m in methods},
        "cells": cells, "wall": wall, "text": text, "firing": firing,
        "base": rules["base"], "selectivity": rules["selectivity"],
        "meta": {m: {"breaks_at": rules["methods"][m]["breaks_at"], "degenerate": {level(k): v for k, v in rules["methods"][m]["degenerate_by_scale"].items()}, "monotone": rules["methods"][m]["monotone_within_3_points"]} for m in methods},
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="LABEL=PATH, relative to build/")
    ap.add_argument("--include-harmful", action="store_true", help="private copy with harmful prompt text, in runs/")
    args = ap.parse_args()
    runs = []
    for item in args.source:
        label, path = item.rsplit("=", 1)
        runs.append(build_run(label, (ROOT / path).resolve(), args.include_harmful))

    template = (Path(__file__).parent / "results_template.html").read_text(encoding="utf8")
    payload = json.dumps({"runs": runs}, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("/*__DATA__*/null", payload)
    if not args.include_harmful:
        for name in WITHHELD:
            for row in load_set(name):
                assert row["prompt"] not in html, f"harmful text leaked: {row['id']}"
    out = ROOT / "runs" / "knob_results_local.html" if args.include_harmful else RESULTS / "knob_results.html"
    out.write_text(html, encoding="utf8")
    print(f"wrote {out} ({len(html) // 1024} KB, {len(runs)} run(s)), data from {DATA.parent}")


if __name__ == "__main__":
    main()
