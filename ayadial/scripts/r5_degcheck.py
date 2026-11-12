"""Check of the degeneracy rule on normal text in the 10 MultiJail languages (CPU, Mac or runtime).
The rule of rounds 1-4 (scripts/05_score.py) counts combining marks as non-letters and treats an unsegmented script as one token per run,
so it calls normal Chinese, Thai and Bengali text degenerate. Round 5 uses is_degenerate5: letters include combining marks (Unicode L* and M*),
and runs in unsegmented scripts (Thai, Lao, Burmese, Khmer, kana, CJK) are split into characters. On Latin-script text the two rules are the same function.
  python scripts/r5_degcheck.py      -> results/round5/degenerate_rule_check.json"""
import collections, importlib.util, json, re, sys, unicodedata
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("score", ROOT / "scripts/05_score.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)
UNSEG = re.compile(r"[฀-๿຀-໿က-႟ក-៿぀-ヿ㐀-䶿一-鿿豈-﫿]")


def is_degenerate5(text):
    chars = [c for c in text if not c.isspace()]
    if chars and sum(unicodedata.category(c)[0] not in "LM" for c in chars) / len(chars) > 0.5:
        return True
    tokens = []
    for t in score.TOKEN.findall(text):
        tokens += list(t) if UNSEG.search(t) else [t]
    if len(set(tokens[:64])) < 5:
        return True
    grams = collections.Counter(zip(tokens, tokens[1:], tokens[2:]))
    return bool(grams) and max(grams.values()) > 10


if __name__ == "__main__":
    from datasets import load_dataset
    d = load_dataset("CohereLabs/aya_dataset")["train"].select_columns(["language_code", "language", "targets"])[:]
    want = {"en": ("eng", "English"), "zh": ("zho", "Simplified Chinese"), "it": ("ita", "Italian"), "vi": ("vie", "Vietnamese"), "ar": ("arb", "Standard Arabic"),
            "ko": ("kor", "Korean"), "th": ("tha", "Thai"), "bn": ("ben", "Bengali"), "sw": ("swh", "Swahili"), "jv": ("jav", "Javanese")}
    out = {"text": "CohereLabs/aya_dataset human-written answers, 20 to 700 characters, first 600 per language; 'short' = their first 40 characters", "languages": {}}
    for lg, (code, name) in want.items():
        T = [t for c, n, t in zip(d["language_code"], d["language"], d["targets"]) if c == code and n == name and 20 <= len(t) <= 700][:600]
        pc = lambda f, X: round(100 * sum(f(t) for t in X) / len(X), 1)
        out["languages"][lg] = {"n": len(T), "old_rule_flags_pct": pc(score.is_degenerate, T), "new_rule_flags_pct": pc(is_degenerate5, T),
                                "old_rule_flags_short_pct": pc(score.is_degenerate, [t[:40] for t in T]), "new_rule_flags_short_pct": pc(is_degenerate5, [t[:40] for t in T])}
    r4 = ROOT / "runs_private/round4/gen"
    rows = [json.loads(l) for f in ("base_wjx", "base_own", "foldcap@1", "ungated@0.1") for l in open(r4 / f"{f}.jsonl")]
    out["round4_english_replies"] = {"n": len(rows), "old_and_new_rule_differ_on": sum(score.is_degenerate(r["response"]) != is_degenerate5(r["response"]) for r in rows),
                                     "flagged_by_old_rule": sum(score.is_degenerate(r["response"]) for r in rows)}
    (ROOT / "results/round5/degenerate_rule_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
