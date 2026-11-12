"""How many harmless replies change at all under each setting (charter priority 4). Pure CPU, from the generated cells on the runtime.
  python scripts/r5_changed.py
For every harmless set and every cell generated with the same batching as a base cell: the share of replies that are character-for-character
identical to the base reply, and the share whose refusal label changed. Generation is deterministic given rows and batching (reuse_check.json: 100%
identical base replies across sessions), so a changed reply is the setting's doing. Writes results/round5/capability/replies_changed.json (numbers only)."""
import collections, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "runs_private/round5", ROOT / "results/round5"
LG = ["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]
GROUPS = [("benign (Alpaca, development)", ["benign"]), ("xstest_safe (development)", ["xstest_safe"]), ("orbench_hard (development)", ["orbench_hard"]),
          ("adversarial_benign (development)", ["wj_adversarial_benign", "dev_ho_b", "dev_in_b"]), ("PHTest harmless (development)", ["pht_harmless_dev"]),
          ("Alpaca instructions (frozen)", ["alp5_frz"]), ("Aya, English (frozen)", ["aya_en_frz"]), ("Aya, 9 other languages (frozen)", [f"aya_{l}_frz" for l in LG[1:]]),
          ("PHTest harmless (frozen)", ["pht_harmless_frz"]), ("orbench_hard (frozen)", ["orb5_frz"]), ("adversarial_benign (frozen)", ["wj5_b_frz"])]
CELLS = sys.argv[1:] or ["ungated@0.02", "ungated@0.1", "foldcap@1", "foldcap@1.25", "foldcap@2", "v2cap@1", "v3gcap@1", "lax3@1", "lax3@2", "both3@1", "ctrlneg@-0.02"]


def rows(name):
    p = RAW / "gen" / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().split("\n") if l.strip()] if p.exists() else []  # not splitlines(): replies contain other line separators


base = {}
for p in sorted((RAW / "gen").glob("base_*.jsonl")):
    if re.fullmatch(r"base_(wjx|own|dev5|sdev5|frz5)", p.stem):
        base.update({f"{r['set']}|{r['id']}": r["response"] for r in rows(p.stem)})
lab = lambda n: json.loads((OUT / "labels" / f"{n}.json").read_text()) if (OUT / "labels" / f"{n}.json").exists() else {}
blab = {}
for b in ("base_wjx", "base_own", "base_dev5", "base_sdev5", "base_frz5"):
    blab.update(lab(b))
out = {"what": "share of harmless replies identical to the base reply (same rows, same batching), and share whose refusal label changed", "max_new_tokens": 128, "groups": {}}
for c in CELLS:
    g = {f"{r['set']}|{r['id']}": r["response"] for r in rows(c)}
    if not g:
        continue
    L = lab(c)
    for name, sets_ in GROUPS:
        ks = [k for k in g if k.split("|")[0] in sets_ and k in base]
        if not ks:
            continue
        same = sum(g[k] == base[k] for k in ks)
        flips = [k for k in ks if k in L and k in blab]
        out["groups"].setdefault(name, {})[c] = {"n": len(ks), "identical_pct": round(100 * same / len(ks), 1), "changed_pct": round(100 * (1 - same / len(ks)), 1),
                                                 "refusal_label_changed_pct": round(100 * sum(L[k][0] != blab[k][0] for k in flips) / len(flips), 1) if flips else None}
(OUT / "capability").mkdir(exist_ok=True)
(OUT / "capability" / "replies_changed.json").write_text(json.dumps(out, indent=1))
cs = [c for c in CELLS if any(c in v for v in out["groups"].values())]
print("changed replies %, by harmless set:   " + "  ".join(f"{c:>13s}" for c in cs))
for name, v in out["groups"].items():
    print(f"  {name[:36]:36s} n {max(e['n'] for e in v.values()):4d} " + "  ".join(f"{v[c]['changed_pct']:13.1f}" if c in v else " " * 13 for c in cs))
