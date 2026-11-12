"""Round 7: the format blocks of the second test split by where the harmful request comes from (CPU, labels only).
  python scripts/r7_formats_by_source.py   ->  results/round7/formats_by_source.json
Unsafe rate (%) of Base, AyaDial (v3gcap@1, round 6) and AyaDial-WSP (round 7, st_wsp@1) on the same rows. Descriptive: no intervals, small n."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7lib as L7
Lr = L7.round_labels("round6")
aya, base = Lr["v3gcap@1"], Lr["base"]
wsp = L7.r7_labels(["xb6.st_wsp@1"] + [f"xbf_{f}.st_wsp@1" for f in L7.FORMATS])
rate = lambda lab, keys: round(float(np.mean([lab[k][1] for k in keys]) * 100), 1)
out = {}
groups = {"four system prompts": ["sysA", "sysB", "sysC", "sysD"], "three multi-turn chats": ["mt2", "mtref", "mtsys"], **{f: [f] for f in ["sysA", "sysB", "sysC", "sysD", "mt2", "mtref", "mtsys", "itwsys"]}}
for g, fs in groups.items():
    for src, name in (("xb_wj_h", "WildJailbreak (jailbreak-wrapped)"), ("xb_mj_en", "MultiJail English (plain questions)")):
        keys = [k for k in base if any(k.split("|")[0] == f"xbf_xa_{f}_{src}" for f in fs)]
        assert keys and all(k in aya and k in wsp for k in keys)
        out.setdefault(g, {})[name] = {"n": len(keys), "unsafe_base": rate(base, keys), "unsafe_ayadial": rate(aya, keys), "unsafe_ayadial_wsp": rate(wsp, keys)}
L7.write_json(L7.OUT / "formats_by_source.json", out)
for g in ("four system prompts", "three multi-turn chats"):
    print(g, out[g])
