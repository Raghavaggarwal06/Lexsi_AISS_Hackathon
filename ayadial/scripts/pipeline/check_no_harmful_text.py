"""Leak check: no text from data/harmful.jsonl or data/xstest_unsafe.jsonl in the given files.

Round 4 extension: when runs_private/round4/sets.jsonl exists, every prompt in it (WildJailbreak adversarial
prompts, wrapped prompts with their jailbreak templates, frozen-test prompts) is private too. Those are long,
so their 40-character windows are indexed at every 8th offset: any copied span of 47 characters or more is caught.

For every file under the given paths it looks for (1) any whole private prompt, and (2) any 40-character
window of a private prompt (whitespace-normalised, case-folded), which catches partial copies and
jailbreak wrappers around a prompt. It also flags JSON keys named "prompt" or "response", since result
files should carry ids, labels and numbers only. Prints the offending file and line number, never the
matched text. Exit code 1 if anything is found.

Usage (from build/): python scripts/pipeline/check_no_harmful_text.py results/round3/block8 scripts/pipeline
"""
import json
import re
import sys
from pathlib import Path

BUILD = Path(__file__).resolve().parents[2]
WINDOW = 40


def norm(text):
    return " ".join(text.lower().split())


def private_prompts():
    out = []
    for name in ("harmful", "xstest_unsafe"):
        with open(BUILD / "data" / f"{name}.jsonl") as f:
            out += [norm(json.loads(line)["prompt"]) for line in f if line.strip()]
    return out


def extra_private():
    """Round 4 private prompts (WildJailbreak, wrapped prompts, frozen test), if the private file is present."""
    out = []
    for path in sorted((BUILD / "runs_private").glob("round*/sets.jsonl")):  # round 4 and round 5 (round 5 adds other benchmarks and other languages)
        with open(path) as f:
            out += [norm(json.loads(line)["prompt"]) for line in f if line.strip()]
    return out


def main(paths):
    prompts = private_prompts()
    extra = extra_private()
    whole = set(prompts)
    whole_extra = {p for p in extra if 20 <= len(p) < 400 or (6 <= len(p) < 20 and not p.isascii())}  # short prompts in other scripts are matched whole
    windows = {p[i:i + WINDOW] for p in prompts for i in range(0, max(1, len(p) - WINDOW + 1))}
    windows |= {p[i:i + WINDOW] for p in extra if len(p) >= WINDOW for i in range(0, len(p) - WINDOW + 1, 8)}
    print(f"private sources: {len(prompts)} prompts from data/, {len(extra)} from runs_private/round*/sets.jsonl")
    files = [f for p in paths for f in ([Path(p)] if Path(p).is_file() else Path(p).rglob("*")) if f.is_file()]
    hits, scanned = [], 0
    for path in files:
        try:
            lines = path.read_text().splitlines()
        except UnicodeDecodeError:
            continue
        scanned += 1
        for n, line in enumerate(lines, 1):
            text = norm(line)
            if any(p in text for p in whole if len(p) <= len(text)) or (len(text) < 20000 and any(p in text for p in whole_extra if len(p) <= len(text))):
                hits.append((path, n, "whole prompt"))
                continue
            if any(text[i:i + WINDOW] in windows for i in range(0, max(1, len(text) - WINDOW + 1))):
                hits.append((path, n, f"{WINDOW}-char window"))
                continue
            if re.search(r'"(prompt|response)"\s*:', line) and path.suffix in (".json", ".jsonl"):
                hits.append((path, n, "prompt/response field"))
    for path, n, why in hits:
        print(f"HIT {path.relative_to(BUILD) if path.is_relative_to(BUILD) else path}:{n} ({why})")
    by_reason = {why: sum(h[2] == why for h in hits) for why in ("whole prompt", f"{WINDOW}-char window", "prompt/response field")}
    print(f"scanned {scanned} files against {len(prompts)} private prompts: {len(hits)} hits {by_reason}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or [str(BUILD / "results" / "round3" / "block8")]))
