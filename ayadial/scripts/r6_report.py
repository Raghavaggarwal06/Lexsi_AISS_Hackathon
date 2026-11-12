"""Round 6 tables: round 5's table and claims code (r5_report.py: same statistics, same verdict rule) with exam B's rows and clusters.
  python scripts/r6_report.py exam            tables/xb_tight[_wg|_q8].json, xb_lax*.json and claims, from results/round6/labels*
  python scripts/r6_report.py dev             tables/xd.json (round 6 development slices: red-teaming and harmless prompts in six unseen languages)
Pure CPU; runs on the runtime or on the Mac. A table whose cells do not exist yet is skipped."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r5_report as R

ROOT = Path(__file__).resolve().parent.parent
OUT = R.OUT = ROOT / "results/round6"
R.BASES = ["base_wjx", "base_own", "base_dev5", "base_xd6", "base_xb6"]
LG, RES = R.LG, R.RES
UNSEEN = ["fil", "fr", "hi", "ru", "sr", "es"]
FMT = ["xa_sysA", "xa_sysB", "xa_sysC", "xa_sysD", "xa_mt2", "xa_mtref", "xa_mtsys", "xa_itwsys"]
FNAME = {"xa_sysA": "short developer preamble", "xa_sysB": "long developer preamble", "xa_sysC": "persona preamble", "xa_sysD": "Portuguese preamble", "xa_mt2": "after two harmless turns", "xa_mtref": "after a refused turn",
         "xa_mtsys": "preamble and two turns", "xa_itwsys": "in-the-wild template as the developer preamble"}
HSRC, BSRC = ["xb_wj_h", "xb_mj_en"], ["xb_orb", "xb_alp", "xb_wj_b"]


def layout(side, names):
    p = "xb" if side == "xb" else "xd"
    Rw = []
    if side == "xb":
        Rw += [("adversarial_harmful (fresh WildJailbreak)", ["xb_wj_h"], "H"), ("adversarial_benign (fresh WildJailbreak)", ["xb_wj_b"], "B"), ("orbench_hard (fresh)", ["xb_orb"], "B"), ("Alpaca instructions (fresh)", ["xb_alp"], "B"),
               ("in-the-wild template + harmful question", ["xb_itw_h"], "H"), ("in-the-wild template + harmless instruction", ["xb_itw_b"], "B"),
               ("MultiJail English", ["xb_mj_en"], "H"), ("MultiJail 9 other languages", [f"xb_mj_{l}" for l in LG[1:]], "H")]
        Rw += [(f"MultiJail {lv}-resource ({', '.join(ls)})", [f"xb_mj_{l}" for l in ls], "H") for lv, ls in RES.items()]
        Rw += [("harmless prompts, English (Aya)", ["xb_aya_en"], "B"), ("harmless prompts, 9 other languages (Aya)", [f"xb_aya_{l}" for l in LG[1:]], "B")]
        Rw += [(f"harmless prompts {lv}-resource", [f"xb_aya_{l}" for l in ls], "B") for lv, ls in RES.items()]
    Rw += [("red-teaming prompts, 6 unseen languages", [f"{p}_art_{l}" for l in UNSEEN], "H"), ("harmless prompts, 6 unseen languages", [f"{p}_ayab_{l}" for l in UNSEEN], "B")]
    Rw += [(f"red-teaming prompts, {l}", [f"{p}_art_{l}"], "H") for l in UNSEEN] + [(f"harmless prompts, {l}", [f"{p}_ayab_{l}"], "B") for l in UNSEEN]
    if side == "xb":
        Rw += [("red-teaming prompts, Arabic and English (seen languages)", ["xb_art_ar", "xb_art_en"], "H")]
        Rw += [("exam formats, all: harmful requests", [f"xbf_{f}_{s}" for f in FMT for s in HSRC], "H"), ("exam formats, all: harmless requests", [f"xbf_{f}_{s}" for f in FMT for s in BSRC], "B")]
        for lab, fs in (("exam formats, four developer preambles", FMT[:4]), ("exam formats, three multi-turn", FMT[4:7])):
            Rw += [(f"{lab}: harmful requests", [f"xbf_{f}_{s}" for f in fs for s in HSRC], "H"), (f"{lab}: harmless requests", [f"xbf_{f}_{s}" for f in fs for s in BSRC], "B")]
        Rw += [("red-teaming prompts, English", ["xb_art_en"], "H"), ("red-teaming prompts, Arabic", ["xb_art_ar"], "H")]
        for f in FMT:
            Rw += [(f"format: {FNAME[f]}: harmful requests", [f"xbf_{f}_{s}" for s in HSRC], "H"), (f"format: {FNAME[f]}: harmless requests", [f"xbf_{f}_{s}" for s in BSRC], "B")]
    return [(lab, sets_, k) for lab, sets_, k in Rw if any(x in names for x in sets_)]


def cluster_of(key):
    set_, id_ = key.split("|", 1)
    if set_.startswith("xb_itw_"):
        return "itw:" + id_.split("::")[0]
    if set_.startswith("xb_mj_"):
        return "mj:" + id_.rsplit("-", 1)[0]
    if set_.startswith("xbf_"):
        return "req:" + id_  # the same request in several formats moves together
    return key


R.layout, R.cluster_of = layout, cluster_of
R.COUNTERPART = [("adversarial_harmful", "adversarial_benign"), ("in-the-wild template + harmful question", "in-the-wild template + harmless instruction"),
                 ("MultiJail English", "harmless prompts, English (Aya)"), ("MultiJail 9 other languages", "harmless prompts, 9 other languages (Aya)"),
                 ("MultiJail high-resource", "harmless prompts high-resource"), ("MultiJail medium-resource", "harmless prompts medium-resource"), ("MultiJail low-resource", "harmless prompts low-resource"),
                 ("red-teaming prompts, 6 unseen languages", "harmless prompts, 6 unseen languages"), ("red-teaming prompts, Arabic and English", "harmless prompts, English (Aya)"),
                 ("exam formats, all: harmful requests", "exam formats, all: harmless requests"), ("exam formats, four developer preambles: harmful requests", "exam formats, four developer preambles: harmless requests"),
                 ("exam formats, three multi-turn: harmful requests", "exam formats, three multi-turn: harmless requests"), ("red-teaming prompts, English", "harmless prompts, English (Aya)"), ("red-teaming prompts, Arabic", "harmless prompts, English (Aya)")] + \
                [(f"red-teaming prompts, {l}", f"harmless prompts, {l}") for l in UNSEEN] + [(f"format: {FNAME[f]}: harmful requests", f"format: {FNAME[f]}: harmless requests") for f in FMT]


_load5 = R.load_labels


def load_labels(judge=None):
    """Round 5's loader, plus: the cells of the exam format suites (label files 'xbf_<fmt>.<cell>') are folded into the cell of the same name, so that
    one table holds a cell's default-template rows and its format rows. Their base cells (base_xbf_<fmt>) are folded into 'base' by round 5's loader."""
    L = _load5(judge)
    for n in [k for k in L if k.startswith("xbf_") and "." in k]:
        cell = n.split(".", 1)[1]
        L.setdefault(cell, {})
        L[cell] = {**L[cell], **L[n]}
    return L


R.load_labels = load_labels


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else "exam"
    D = json.loads((OUT / "exam_declaration.json").read_text()) if (OUT / "exam_declaration.json").exists() else {}
    tight = D.get("tight_cells", ["ungated@0.1", "v3gcap@1", "v4cap@1", "both4@1"])
    lax = D.get("lax_cells", ["ctrlneg@-0.02", "lax4@2", "both4@1"])
    ctrl, neg = D.get("control", "ungated@0.1"), D.get("control_negative", "ctrlneg@-0.02")
    if want == "dev":
        have = set(R.load_labels())
        cells = [c for c in tight + lax if c in have]
        if cells:
            R.table("xd", "xd", list(dict.fromkeys(cells)), control=ctrl if ctrl in have else None)
        return
    for judge, suf in ((None, ""), ("wg", "_wg"), ("q8", "_q8")):
        if judge and not (OUT / f"labels_{judge}" / "base_xb6.json").exists():
            continue
        L = R.load_labels(judge)
        tc = [c for c in tight if c in L]
        if tc and ctrl in L:
            R.table(f"xb_tight{suf}", "xb", tc, control=ctrl, judge=judge)
            for c in tc:
                if c != ctrl:
                    try:
                        R.claims(f"xb_tight{suf}", c)
                    except Exception as e:
                        print("claims", c, type(e).__name__, e)
        lc = [c for c in lax if c in L]
        if lc:
            R.table(f"xb_lax{suf}", "xb", lc, control=neg if neg in L else None, judge=judge)
        # the product against round 5's version, and the calibrated cells against the product at s = 1: paired differences with intervals
        prod, ref = (D.get("product") or {}).get("strict"), D.get("reference")
        if prod in L and ref in L:
            R.table(f"xb_vs_ref{suf}", "xb", [prod], control=ref, judge=judge)
        calc = [c["name"] for c in D.get("format_only_cells", []) if c["name"] in L]
        if calc and prod in L:
            R.table(f"xb_cal{suf}", "xb", calc, control=prod, judge=judge)


if __name__ == "__main__":
    main()
