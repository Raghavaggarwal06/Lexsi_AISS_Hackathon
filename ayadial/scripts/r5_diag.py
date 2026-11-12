"""Read or push? python scripts/r5_diag.py <cell> [<cell> ...]         (CPU; reads the private gen files for the fire values, writes numbers only)
For each harmful set: of the prompts the base answers unsafely, on how many does the cell's unit fire (push above 0.5 r at the reply-start token),
and what happens to the ones it fires on and to the ones it misses. A failure with a low fire rate is the read; a failure where it fires and
the reply stays unsafe is the push (or the judge). Writes results/round5/diag/read_or_push_<cell>.json."""
import collections, sys
from r5lib import *

GROUPS = [("adversarial_harmful (dev)", WJ_H), ("harmful", ["harmful"]), ("harmful_jb", ["harmful_jb"]), ("in-the-wild + harmful question", ["itw_h_dev"]),
          ("MultiJail English", ["mj_en_dev"]), ("MultiJail high-resource", [f"mj_{l}_dev" for l in ("zh", "it", "vi")]), ("MultiJail medium-resource", [f"mj_{l}_dev" for l in ("ar", "ko", "th")]),
          ("MultiJail low-resource", [f"mj_{l}_dev" for l in ("bn", "sw", "jv")])] + [(f"JailbreakBench {m} ({p})", [f"jbb_{m}_{p}"]) for m in ("GCG", "DSN", "PAIR", "JBC", "RS") for p in ("pool", "dev")] + [("harmful + transplanted suffix", ["sfx_h_dev"])]
if "--frozen" in sys.argv:
    GROUPS = [("adversarial_harmful (fresh)", ["wj5_h_frz"]), ("in-the-wild + harmful question", ["itw_h_frz"]), ("MultiJail English", ["mj_en_frz"]), ("MultiJail high-resource", [f"mj_{l}_frz" for l in ("zh", "it", "vi")]),
              ("MultiJail medium-resource", [f"mj_{l}_frz" for l in ("ar", "ko", "th")]), ("MultiJail low-resource", [f"mj_{l}_frz" for l in ("bn", "sw", "jv")])] + [(f"JailbreakBench {m} (frozen)", [f"jbb_{m}_frz"]) for m in ("GCG", "DSN", "PAIR", "JBC", "RS")] + [("harmful + transplanted suffix", ["sfx_h_frz"])]
base = {}
for f in sorted((OUT / "labels").glob("base_*.json")):
    base.update(json.loads(f.read_text()))
(OUT / "diag").mkdir(exist_ok=True)
for cell in [a for a in sys.argv[1:] if not a.startswith("--")]:
    lab = labels(cell)
    fire = {rk(r): r.get("fire") for r in read_jsonl(RAW / "gen" / f"{cell}.jsonl")}
    out = {"cell": cell, "fires": "push above 0.5 r at the reply-start token", "groups": {}}
    for name, sets_ in GROUPS:
        keys = [k for k in lab if k.split("|")[0] in sets_ and k in base and fire.get(k) is not None]
        if not keys:
            continue
        f = np.array([fire[k] > 0.5 for k in keys])
        b, a = np.array([base[k] for k in keys]), np.array([lab[k] for k in keys])
        bu, ba = b[:, 1] == 1, (b[:, 0] == 0) & (b[:, 2] == 0)
        pct = lambda m: None if m.sum() == 0 else round(float(a[m, 1].mean() * 100), 1)
        out["groups"][name] = {"n": len(keys), "fires": round(float(f.mean() * 100), 1), "base_unsafe_n": int(bu.sum()), "fires_among_base_unsafe": None if bu.sum() == 0 else round(float(f[bu].mean() * 100), 1),
                               "base_answered_n": int(ba.sum()), "fires_among_base_answered": None if ba.sum() == 0 else round(float(f[ba].mean() * 100), 1),
                               "still_unsafe_among_base_unsafe_fired": pct(bu & f), "still_unsafe_among_base_unsafe_missed": pct(bu & ~f), "n_base_unsafe_fired": int((bu & f).sum()), "n_base_unsafe_missed": int((bu & ~f).sum()),
                               "unsafe_base": round(float(b[:, 1].mean() * 100), 1), "unsafe_cell": round(float(a[:, 1].mean() * 100), 1)}
        g = out["groups"][name]
        print(f"{cell} | {name:38s} n {g['n']:4d} unsafe {g['unsafe_base']:5.1f}->{g['unsafe_cell']:5.1f} | fires {g['fires']:5.1f}% | base-unsafe {g['base_unsafe_n']:3d}: fires on {g['fires_among_base_unsafe']}% ; still unsafe when fired {g['still_unsafe_among_base_unsafe_fired']} (n={g['n_base_unsafe_fired']}), when missed {g['still_unsafe_among_base_unsafe_missed']} (n={g['n_base_unsafe_missed']})")
    (OUT / "diag" / f"read_or_push_{cell}{'_frozen' if '--frozen' in sys.argv else ''}.json").write_text(json.dumps(out, indent=1))
sync()
