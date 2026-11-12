"""Main judge (Qwen3Guard-Gen-0.6B) against Qwen3Guard-Gen-8B, by language, from the two label folders. Pure CPU; runs on the runtime or on the Mac.
  python scripts/r5_judge_lang.py
Pools every cell that has labels under both judges, separately for development and frozen rows, and writes results/round5/judge2/by_language.json.
Per language: agreement on the refusal label and on the unsafe label, and each judge's unsafe rate on the MultiJail replies (harmful requests) and
refusal rate on the Aya replies (harmless requests). English is the reference level of agreement (decision 9). Ids and numbers only."""
import collections, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r5_report as R

OUT = R.OUT
main_dir, q8_dir = OUT / "labels", OUT / "labels_q8"
out = {"main_judge": "Qwen/Qwen3Guard-Gen-0.6B", "second_judge": "Qwen/Qwen3Guard-Gen-8B", "note": "a second model of the same family, not a human check"}
if not q8_dir.exists():
    sys.exit("no labels_q8 folder yet")
pct = lambda v: round(float(np.mean(v)) * 100, 1) if len(v) else None
for side, key in (("dev", "languages"), ("frz", "languages_frozen")):
    rows, cells = collections.defaultdict(list), []
    for p in sorted(q8_dir.glob("*.json")):
        if not (main_dir / p.name).exists():
            continue
        q, m = json.loads(p.read_text()), json.loads((main_dir / p.name).read_text())
        used = False
        for k, v in q.items():
            set_ = k.split("|")[0]
            if not set_.endswith(f"_{side}") or k not in m:
                continue
            fam = set_.split("_")[0]
            if fam in ("mj", "aya"):
                lg = set_.split("_")[1]
            elif fam == "itw":
                lg = "en, in-the-wild templates"
            elif fam == "sorry":
                lg = "SORRY-Bench " + set_[len("sorry_"):-len(f"_{side}")]
            else:
                continue
            rows[lg].append((fam, m[k], v))
            used = True
        if used:
            cells.append(p.stem)
    order = [l for l in R.LG if l in rows] + sorted(l for l in rows if l not in R.LG)
    res = {}
    for lg in order:
        rs = rows[lg]
        e = {"n": len(rs), "agree_refusal": float(np.mean([a[0] == b[0] for _, a, b in rs])), "agree_unsafe": float(np.mean([a[1] == b[1] for _, a, b in rs])),
             "unparsed_or_degenerate_second": pct([b[2] for _, a, b in rs])}
        for fam, name in (("mj", "multijail"), ("aya", "aya"), ("itw", "in_the_wild"), ("sorry", "sorry")):
            sub = [(a, b) for f, a, b in rs if f == fam]
            if sub:
                e[name] = {"n": len(sub), "refusal_main": pct([a[0] for a, b in sub]), "refusal_second": pct([b[0] for a, b in sub]), "unsafe_main": pct([a[1] for a, b in sub]), "unsafe_second": pct([b[1] for a, b in sub])}
        res[lg] = e
    out[key] = res
    out[f"cells_{side}"] = cells
(OUT / "judge2").mkdir(exist_ok=True)
(OUT / "judge2" / "by_language.json").write_text(json.dumps(out, indent=1))
for key in ("languages", "languages_frozen"):
    print(key, out.get(f"cells_{'dev' if key == 'languages' else 'frz'}"))
    for lg, e in out[key].items():
        mj = e.get("multijail") or e.get("sorry") or e.get("in_the_wild") or {}
        print(f"  {lg[:34]:34s} n {e['n']:5d} agree refusal {e['agree_refusal']:.3f} unsafe {e['agree_unsafe']:.3f} | harmful side: unsafe main {mj.get('unsafe_main')} second {mj.get('unsafe_second')} refusal main {mj.get('refusal_main')} second {mj.get('refusal_second')}")
