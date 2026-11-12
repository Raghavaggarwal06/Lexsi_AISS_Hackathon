#!/usr/bin/env python3
"""Build the LaTeX results report for AyaDial from the JSON files under results/.

Run from build/:
    python3 presentation/build_report_tex.py
    cd presentation/report && tectonic ayadial_results.tex

Writes presentation/report/ayadial_results.tex. Every number in the document is read from a file under results/
through the loaders of presentation/build_ayadial.py (imported, not copied), which check each rate computed from the
per-prompt label files against the summary table of the same cell. This script adds its own consistency checks on the
headline numbers (asserts below). No prompt, template or reply text is read or written.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_ayadial as B  # noqa: E402

OUTDIR = HERE / "report"
OUT = OUTDIR / "ayadial_results.tex"

J, tshares, dev_shares, claim = B.J, B.tshares, B.dev_shares, B.claim


# formatting
def num(x, d=1):
    s = f"{x:.{d}f}"
    if float(s) == 0:
        s = f"{0:.{d}f}"
    return f"${s}$"


def sg(x, d=1):
    s = f"{x:+.{d}f}"
    if float(s) == 0:
        s = "+" + f"{0:.{d}f}"
    return f"${s}$"


def ci(c):
    return f"[{num(c[0])}, {num(c[1])}]"


def dci(d, c):
    return f"{sg(d)}\\,{ci(c)}"


def rng(v, what):
    lo, hi = num(min(v)), num(max(v))
    return f"{lo}\\% {what} in all three" if lo == hi else f"{lo} to {hi}\\% {what}"


# winner marking: one helper for every results table
OURS = {"AyaDial", "AyaDial-WSP", "Loosen", "TL", "Earlier"}
CONTESTS = []  # (table, contest, outcome, winners) for the build report


def mark(table, contest, cands, tol=1.0, higher=False):
    """One contest: one metric on one data set. cands = [(method, label, value)], Base excluded by the caller.
    Lower is better. The best contestant is bold only if it leads the runner-up by at least `tol` points on the
    unrounded values. Settings of one method tied for best are bold together only if they lead the best contestant
    of every other method by at least `tol`. Returns one boolean per candidate."""
    eps = 1e-9
    vals = [(-v if higher else v) for _, _, v in cands]  # an up arrow: highest wins
    bold = [False] * len(cands)
    if len(cands) < 2:
        CONTESTS.append((table, contest, "single", []))
        return bold
    best = min(vals)
    top = [i for i, v in enumerate(vals) if abs(v - best) <= eps]
    methods = {cands[i][0] for i in top}
    style = lambda i: "ours" if is_ours(cands[i][0], cands[i][1]) else "other"
    others = [v for (m_, l_, _), v in zip(cands, vals) if not is_ours(m_, l_)]
    if others and any(is_ours(m_, l_) for m_, l_, _ in cands) and all(is_ours(cands[i][0], cands[i][1]) for i in top):
        # our versions count as one side against the other methods: the best of ours is marked when it leads the best other method by `tol`
        if min(others) - best >= tol - eps:
            for i in top:
                bold[i] = "ours"
            CONTESTS.append((table, contest, "won", [cands[i][1] for i in top]))
        else:
            CONTESTS.append((table, contest, "near-tie", [cands[i][1] for i in top]))
        return bold
    if len(methods) > 1:
        CONTESTS.append((table, contest, "tie", [cands[i][1] for i in top]))
        return bold
    m = cands[top[0]][0]
    rest = [v for i, v in enumerate(vals) if i not in top] if len(top) == 1 else [v for mm, _, v in cands if mm != m]
    if rest and min(rest) - best >= tol - eps:
        for i in top:
            bold[i] = style(i)
        CONTESTS.append((table, contest, "won", [cands[i][1] for i in top]))
    else:
        CONTESTS.append((table, contest, "near-tie", [cands[i][1] for i in top]))
    return bold


def is_ours(method, label=""):
    return method in OURS or method in ("A", "B", "C") or str(label).startswith(("AyaDial", "Loosen", "Tighten"))


def numb(x, b):
    """b: False (plain), "ours" (a comparison won by one of our adapters: filled cell, bold) or any other true value (won by another method: bold)."""
    s = f"{x:.1f}"
    if float(s) == 0:
        s = "0.0"
    if b == "ours":
        return f"\\cellcolor{{win}}$\\mathbf{{{s}}}$"
    return f"$\\mathbf{{{s}}}$" if b else f"${s}$"


def excludes0(c):
    return c[0] > 0 or c[1] < 0


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


# colours (one per method, every figure)
COL = {"Base": "8C8C8C", "Control": "B39B6B", "AyaDial": "D9541E", "Earlier": "2C6DB5",
       "RefusalDirection": "7A4F9A", "AdaSteer": "2A9D8F", "CAST": "8C564B", "Loosen": "3A8F3A",
       "TL": "A23B72"}
CNAME = {"Base": "cBase", "Control": "cCtrl", "AyaDial": "cDial", "Earlier": "cEarly",
         "RefusalDirection": "cRD", "AdaSteer": "cAda", "CAST": "cCast", "Loosen": "cLoose", "TL": "cTL"}
MARK = {"Base": "*", "Control": "diamond*", "AyaDial": "triangle*", "Earlier": "square*",
        "RefusalDirection": "o", "AdaSteer": "pentagon", "CAST": "star"}


def build():
    S = {}
    # ---------------- constants ----------------
    fit = J("round4/fold/fit.json")
    exp_cap = J("round4/gguf/export_cap.json")
    exp_both = J("round5/gguf/export_both_sides.json")
    lab_v3g = J("round5/lab/v3g.json")
    expo = J("round5/lax/exposure_v3g.json")
    par4 = J("round4/fold/parity_logits_cap.json")
    par5 = J("round5/parity/units.json")
    chk3 = J("round3/block0/checks.json")
    import re
    inter = int(re.search(r'"intermediate_size":\s*(\d+)', json.dumps(chk3)).group(1))
    r5lib = (B.BUILD / "scripts/r5lib.py").read_text()
    gain_l = float(re.search(r"def lax_units\([^)]*gain=([0-9.]+)", r5lib).group(1))
    gain_t = exp_cap["kw"]["gain"]
    assert gain_t == 1.0 and gain_l == 1.0, "formula below assumes unit gain on both sides"
    S["layer"] = fit["layer"]
    S["inter"] = inter
    S["beta"] = exp_cap["kw"]["cap_beta"]
    S["th1"], S["th3"] = fit["theta0"], lab_v3g["theta0"]
    assert close(S["th3"], expo["theta0_strict"])
    S["thL"], S["tL"], S["betaL"] = exp_both["theta"], exp_both["t"], 1 / expo["ramp_width"]
    S["units_t"], S["units_l"] = exp_both["strict_units"], exp_both["lax_units"]
    S["kl_s0"] = par4["cap_s0"]["base_vs_weights"]["kl_first_token"]
    S["kl_s0_n"] = par4["n_prompts"]
    kl_hook = [v["hook_vs_weights"]["kl_first_token"] for k, v in par5["cells"].items() if k.startswith(("v3gcap", "lax3"))]
    S["kl_hook"] = (min(kl_hook), max(kl_hook), par5["n_prompts"])
    omw = fit["off_marker_write"]
    S["off3"] = 100 * omw["3"]["off_frac_above_0.01"]
    S.update(adapter_facts(fit, exp_cap, exp_both, lab_v3g, expo, r5lib))

    # ---------------- 3. development data ----------------
    stf, sts = J("round6/safetune/full.json"), J("round6/safetune/full_small.json")
    dev_main = J("round5/tables/dev_main.json")
    dm = {r["row"]: r for r in dev_main["rows"]}
    dev = {}
    for name, cell in (("Base", "base_wjx"), ("Control", "ungated@0.1"), ("AyaDial", "v3gcap@1"), ("Earlier", "foldcap@1")):
        chk = (stf["base"]["harmful_refusal"], stf["base"]["harmful_unsafe"], stf["base"]["benign_refusal"]) if cell == "base_wjx" else None
        d = dev_shares(cell, check=chk)
        tcell = "base" if cell == "base_wjx" else cell
        th, tb = dm["adversarial_harmful"]["cells"][tcell], dm["adversarial_benign"]["cells"][tcell]
        assert th["n"] == d["H"]["n"] and close(th["unsafe"], d["H"]["unsafe"]) and close(th["refusal"], d["H"]["refusal"]), name
        assert tb["n"] == d["B"]["n"] and close(tb["refusal"], d["B"]["refusal"]), name
        dev[name] = d
    S["dev"] = dev
    S["nH"], S["nB"] = dev["Base"]["H"]["n"], dev["Base"]["B"]["n"]
    assert [S["nH"], S["nB"]] == stf["base"]["n"]
    st = []
    for k, v in stf["cells"].items():
        meth, val = k.split("@")
        name = {"st_rd": "RefusalDirection", "st_adasteer": "AdaSteer"}[meth]
        d = dev_shares(k, check=(v["harmful_refusal"], v["harmful_unsafe"], v["benign_refusal"]))
        st.append((name, ("strength " if meth == "st_rd" else "bound ") + val, d["H"]["unsafe"], d["B"]["refusal"]))
        S.setdefault("st_full", []).append((name, ("strength " if meth == "st_rd" else "bound ") + val, d))
    S["st"] = st
    judge = J("round5/judge2/by_language.json")["main_judge"]
    import re as _re
    assert _re.search(r'JUDGE[^\n]*"' + _re.escape(judge) + '"', (B.BUILD / "scripts/r4lib.py").read_text()), "judge id differs between files"
    assert J("round5/tables/dev_main.json")["judge"] == "qwen3guard-0.6b"
    S["judge"] = judge.split("/")[-1]
    cc = sts["cells"]
    S["cast"] = {"alphas": [k.split("@")[1] for k in cc], "unsafe": [c["harmful_unsafe"] for c in cc.values()], "hr": [c["harmful_refusal"] for c in cc.values()],
                 "ref": [c["benign_refusal"] for c in cc.values()], "base": sts["base"]}
    assert sts["base"]["n"] == stf["base"]["n"]

    # ---------------- 4. dial ----------------
    t4 = J("round4/table_all_methods.json")
    sweep = []
    for s in [0, 0.75, 1, 1.25, 1.5, 2, 2.5, 3]:
        d = dev_shares(f"foldcap@{s:g}")
        if f"foldcap@{s:g}" in t4:  # round 4's own table, one decimal
            r4 = t4[f"foldcap@{s:g}"]
            assert round(d["H"]["unsafe"], 1) == r4["adversarial_harmful"][1] and round(d["B"]["refusal"], 1) == r4["adversarial_benign"][0], s
        sweep.append((s, d["H"]["unsafe"], d["B"]["refusal"]))
    S["sweep"] = sweep
    S["dial_full"] = {v: dev_shares(f"v3gcap@{v:g}") for v in (0.75, 1, 1.25)}  # AyaDial's measured settings
    S["dial_pts"] = [(v, d_["H"]["unsafe"], d_["B"]["refusal"]) for v, d_ in S["dial_full"].items()]
    S["sweep_full"] = {v: dev_shares(f"foldcap@{v:g}") for v, _, _ in sweep}  # the earlier version's sweep (appendix)
    assert close(S["dial_pts"][1][1], dev["AyaDial"]["H"]["unsafe"])
    sd4 = J("round4/scoreboard_dev.json")["rates"]
    ctl = []
    for k in sorted((k for k in sd4 if k.startswith("ungated@")), key=lambda k: float(k.split("@")[1])):
        dd = dev_shares(k)  # raises unless all 1910 development requests are labelled
        r4 = sd4[k]
        assert round(dd["H"]["unsafe"], 1) == r4["adversarial_harmful"][1] and round(dd["B"]["refusal"], 1) == r4["adversarial_benign"][0], k
        assert (dd["H"]["n"], dd["B"]["n"]) == (r4["adversarial_harmful"][3], r4["adversarial_benign"][3])
        ctl.append((float(k.split("@")[1]), dd["H"]["unsafe"], dd["B"]["refusal"]))
    S["ctl_pts"] = ctl
    for sv, u, r in sweep:  # the AyaDial sweep against the same scoreboard where it has the cell
        k = f"foldcap@{sv:g}"
        if k in sd4:
            assert round(u, 1) == sd4[k]["adversarial_harmful"][1] and round(r, 1) == sd4[k]["adversarial_benign"][0], k
    # the table settings on the lines equal the table values
    one = lambda pts, v: next(x for x in pts if x[0] == v)
    for name, pts, v in (("Earlier", sweep, 1), ("AyaDial", S["dial_pts"], 1), ("Control", ctl, 0.1)):
        _, u, r = one(pts, v)
        assert close(u, dev[name]["H"]["unsafe"]) and close(r, dev[name]["B"]["refusal"]), name

    # ---------------- 5. held-out test 1 ----------------
    T5 = "round5/tables/frz_tight.json"
    C5 = J("round5/tables/frz_tight_claims_v3gcap@1.json")
    c5 = {r["row"]: r for r in C5["rows"]}
    cells4 = [("Base", "base"), ("Control", "ungated@0.1"), ("AyaDial", "v3gcap@1"), ("Earlier", "foldcap@1")]
    H5 = [("WildJailbreak, new requests", "adversarial_harmful (fresh WildJailbreak)", "WildJailbreak"),
          ("In-the-wild templates", "in-the-wild template + harmful question", "In-the-wild"),
          ("JailbreakBench GCG", "JailbreakBench GCG (held-out behaviours)", "GCG"),
          ("JailbreakBench random search", "JailbreakBench RS (held-out behaviours)", "Random search"),
          ("JailbreakBench PAIR", "JailbreakBench PAIR (held-out behaviours)", "PAIR"),
          ("MultiJail, 9 languages", "MultiJail 9 other languages", "MultiJail")]
    B5 = [("WildJailbreak, new harmless", "adversarial_benign (fresh WildJailbreak)"),
          ("In-the-wild templates, harmless", "in-the-wild template + harmless instruction"),
          ("Harmless request + transplanted suffix", "harmless request + transplanted suffix"),
          ("PHTest harmless", "PHTest harmless"),
          ("Aya harmless, 9 languages", "harmless prompts, 9 other languages (Aya)")]
    S["t1H"], S["t1B"] = harm_rows(T5, H5, cells4, "round5", c5), harmless_rows(T5, B5, cells4, "round5")
    S["fig1"] = [(short, r["vals"]) for (_, _, short), r in zip(H5, S["t1H"])]
    S["sorry"] = harm_rows(T5, [("SORRY-Bench base questions", "SORRY-Bench base", ""),
                                ("SORRY-Bench other styles", "SORRY-Bench other styles (persuasion, role play, misspellings, ciphers, translations)", "")],
                           cells4, "round5", c5)
    S["T5"], S["c5"] = T5, c5
    S["mj5"] = c5["MultiJail 9 other languages"]
    S["itw5"] = c5["in-the-wild template + harmful question"]
    diag = J("round5/diag/read_or_push_foldcap@1.json")["groups"]
    S["fire"] = (diag["adversarial_harmful (dev)"]["fires_among_base_unsafe"], diag["in-the-wild + harmful question"]["fires_among_base_unsafe"])

    # ---------------- 6. held-out test 2 ----------------
    T6 = "round6/tables/xb_tight.json"
    C6 = J("round6/tables/xb_tight_claims_v3gcap@1.json")
    c6 = {r["row"]: r for r in C6["rows"]}
    cells3 = [("Base", "base"), ("Control", "ungated@0.1"), ("AyaDial", "v3gcap@1")]
    H6 = [("Aya red-teaming, six unseen languages", "red-teaming prompts, 6 unseen languages", ""),
          ("In-the-wild templates, new", "in-the-wild template + harmful question", ""),
          ("Formats, all eight", "exam formats, all: harmful requests", ""),
          ("Four system-prompt formats", "exam formats, four developer preambles: harmful requests", ""),
          ("Three multi-turn formats", "exam formats, three multi-turn: harmful requests", ""),
          ("MultiJail, 9 languages, new", "MultiJail 9 other languages", "")]
    B6 = [("Six unseen languages, harmless", "harmless prompts, 6 unseen languages"),
          ("In-the-wild templates, harmless", "in-the-wild template + harmless instruction"),
          ("Formats, all eight, harmless", "exam formats, all: harmless requests"),
          ("Four system-prompt formats, harmless", "exam formats, four developer preambles: harmless requests"),
          ("Three multi-turn formats, harmless", "exam formats, three multi-turn: harmless requests"),
          ("Aya harmless, 9 languages", "harmless prompts, 9 other languages (Aya)")]
    S["t2H"], S["t2B"] = harm_rows(T6, H6, cells3, "round6", c6), harmless_rows(T6, B6, cells3, "round6")
    S["T6"], S["c6"] = T6, c6
    six = c6["red-teaming prompts, 6 unseen languages"]
    S["six"] = six
    S["six_hb"] = S["t2B"][0]
    # headline: pooled six languages, label files against the claims file
    b6 = tshares("round6", T6, "red-teaming prompts, 6 unseen languages", "base")
    k6 = tshares("round6", T6, "red-teaming prompts, 6 unseen languages", "v3gcap@1")
    assert close(b6["unsafe"], six["base_unsafe"]) and close(k6["unsafe"], six["unsafe"]) and b6["n"] == six["n"]
    LN = [("fil", "Filipino"), ("fr", "French"), ("hi", "Hindi"), ("ru", "Russian"), ("sr", "Serbian"), ("es", "Spanish")]
    langs = []
    for code, name in LN:
        row = f"red-teaming prompts, {code}"
        vals = [tshares("round6", T6, row, c)["unsafe"] for _, c in cells3]
        cl = c6[row]
        assert close(vals[2], cl["unsafe"])
        langs.append((name, vals, cl["vs_control"], cl["vs_control_ci"], excludes0(cl["vs_control_ci"]), tshares("round6", T6, row, "base")["n"]))
    S["langs"] = langs
    S["fmt"] = c6["exam formats, all: harmful requests"]
    S["fmt_b"] = next(r for r in S["t2B"] if r["row"] == "exam formats, all: harmless requests")
    isfmt = lambda r: r["row"].startswith("exam formats")
    S["t2H_lang"], S["t2B_lang"] = [r for r in S["t2H"] if not isfmt(r)], [r for r in S["t2B"] if not isfmt(r)]
    S["t2H_fmt"], S["t2B_fmt"] = [r for r in S["t2H"] if isfmt(r)], [r for r in S["t2B"] if isfmt(r)]
    # which public sets the format requests come from (set ids of the table rows)
    T6r = {r["row"]: r for r in J(T6)["rows"]}
    pub = {"xb_wj_h": "WildJailbreak", "xb_mj_en": "MultiJail English", "xb_orb": "OR-Bench-hard", "xb_alp": "Alpaca", "xb_wj_b": "WildJailbreak"}
    for row, key in (("adversarial_harmful (fresh WildJailbreak)", "xb_wj_h"), ("MultiJail English", "xb_mj_en"), ("orbench_hard (fresh)", "xb_orb"),
                     ("Alpaca instructions (fresh)", "xb_alp"), ("adversarial_benign (fresh WildJailbreak)", "xb_wj_b")):
        assert T6r[row]["sets"] == [key], row
    def fmt_src(row):
        return list(dict.fromkeys(pub[k.split("_xb_")[1].join(["xb_", ""])] for k in T6r[row]["sets"]))
    S["fmt_src_h"], S["fmt_src_b"] = fmt_src("exam formats, all: harmful requests"), fmt_src("exam formats, all: harmless requests")
    pos = J("round6/robust/positions_v3g.json")
    assert close(pos["theta0"], S["th3"])
    S["fire_fmt"] = (100 * pos["formats"]["rb_default"]["strict_s1_fires_harmful"], 100 * pos["formats"]["rb_sys_long"]["strict_s1_fires_harmful"],
                     pos["formats"]["rb_default"]["n"])
    S["itw6"] = c6["in-the-wild template + harmful question"]
    S["mj6"] = c6["MultiJail 9 other languages"]
    S["en6"] = c6["red-teaming prompts, English"]
    q8 = claim("round6/tables/xb_tight_q8_claims_v3gcap@1.json", "red-teaming prompts, 6 unseen languages")
    S["q8"] = q8

    # ---------------- 7. Loosen ----------------
    TL = "round5/tables/frz_lax.json"
    cellsL = [("Base", "base"), ("Control", "ctrlneg@-0.02"), ("Loosen", "lax3@2"), ("TL", "both3@1")]
    BL = [("OR-Bench-hard", "orbench_hard (fresh)"),
          ("WildJailbreak, new harmless", "adversarial_benign (fresh WildJailbreak)"),
          ("Alpaca", "Alpaca instructions (fresh)"),
          ("In-the-wild templates, harmless", "in-the-wild template + harmless instruction"),
          ("PHTest harmless", "PHTest harmless"),
          ("Aya harmless, 9 languages", "harmless prompts, 9 other languages (Aya)")]
    HL = [("WildJailbreak, new requests", "adversarial_harmful (fresh WildJailbreak)"),
          ("In-the-wild templates", "in-the-wild template + harmful question"),
          ("MultiJail, 9 languages", "MultiJail 9 other languages")]
    S["LB"] = lax_rows(TL, BL, cellsL, "refusal")
    S["LBu"] = lax_rows(TL, BL[:1], cellsL, "unsafe")
    S["LH"] = lax_rows(TL, HL, cellsL, "unsafe")
    S["orb"] = S["LB"][0]
    S["orbu"] = S["LBu"][0]
    T = J(T5)
    S["itwb_unsafe"] = next(r for r in T["rows"] if r["row"] == "in-the-wild template + harmless instruction")["cells"]["base"]["unsafe"]
    S["llama"] = J("round6/llama/checks.json")["no_lora_field"]
    S["calib"] = safetune_calibration()
    S["wsp"] = wsp_data()
    S["st_tests"] = safetune_tests()
    return S


ST_METHODS = ["RefusalDirection", "AdaSteer", "CAST"]
ST_TESTS_FILE = "round7/safetune_tests.json"


def safetune_calibration():
    """What the SafeTune baselines were calibrated on, read from scripts/r6_safetune.py (the script that ran them)."""
    import re
    code = (B.BUILD / "scripts/r6_safetune.py").read_text()
    m = re.search(r"return (rng\.sample.*?), (rng\.sample.*?)\n", code)
    sides = []
    for part in m.groups():
        draws = re.findall(r'rng\.sample\(((?:by\["\w+"\](?: \+ )?)+), (\d+)\)', part)
        sides.append([(int(n), list(dict.fromkeys(src(k) for k in re.findall(r'by\["(\w+)"\]', keys)))) for keys, n in draws])
    assert "random.Random(0)" in code
    return {"harmful": sides[0], "harmless": sides[1]}


def safetune_tests():
    """SafeTune on the two tests, when it has been run: results/round7/safetune_tests.json with
    {"settings": {method: "setting text"}, "first"|"second": {method: {summary-table row name: {"unsafe": x, "refusal": y, "n": n}}}}.
    Absent file or cell: the report prints "pending"."""
    p = B.RES / ST_TESTS_FILE
    if not p.exists():
        return {}
    with open(p) as f:
        T = json.load(f)
    pw = B.RES / "round7/wsp_tests.json"  # AyaDial-WSP on the same rows (scripts/r7_wsp.py), same format; absent = pending
    if pw.exists():
        with open(pw) as f:
            W = json.load(f)
        for test in ("first", "second"):
            T.setdefault(test, {}).update(W.get(test, {}))
        T.setdefault("settings", {}).update(W.get("settings", {}))
    return T


def st_cell(S, method, r):
    """SafeTune outcome shares for one row from results/round7/safetune_tests.json, or {} while pending.
    Harmful rows need "refused" (refused and not unsafe), "answered" and "unsafe"; harmless rows need "refusal".
    A missing key stays pending: nothing is derived by subtraction."""
    cell = S["st_tests"].get(r["test"], {}).get(method, {}).get(r["row"]) or {}
    if cell and "n" in cell:
        assert cell["n"] == r["n"], (method, r["row"])
    return cell


def st_value(S, test, method, r):
    """The SafeTune number for one row, or None while pending; checked against the row's n when the file gives it."""
    cell = S["st_tests"].get(r.get("test", test), {}).get(method, {}).get(r["row"])
    if not cell or cell.get(r["metric"]) is None:
        return None
    if "n" in cell:
        assert cell["n"] == r["n"], (test, method, r["row"])
    return cell[r["metric"]]


# public names of the fit sets (the mapping of round 5's summary, section 2, "What each version had seen")
FITSRC = [("trfit_", "WildJailbreak training"), ("harmful_fit", "HarmBench and JailbreakBench plain behaviours"),
          ("xstest_", "XSTest"), ("orbench_hard_fit", "OR-Bench-hard"), ("benign_fit", "Alpaca"),
          ("jbb_", "JailbreakBench GCG and PAIR artifacts"), ("aug_gcg_", "GCG suffix transplants"),
          ("itw_", "in-the-wild templates"), ("mj_", "MultiJail"), ("aya_", "Aya prompts")]


def src(key):
    return next(name for pre, name in FITSRC if key.startswith(pre))


def adapter_facts(fit, exp_cap, exp_both, lab_v3g, expo, r5lib):
    import re
    F = {}
    m = re.search(r"LYR, ALPHA, KAPPA, BETA = (\d+), ([0-9.]+), ([0-9.]+), ([0-9.]+)", r5lib)
    lyr, alpha, kappa, beta = int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
    assert lyr == fit["layer"] and alpha == fit["alpha"] and kappa == fit["kappa"] and beta == exp_cap["kw"]["cap_beta"]
    m = re.search(r"STRICT_UNITS, LAX_UNITS = \((\d+), (\d+)\), \((\d+), (\d+)\)", r5lib)
    assert [int(m.group(1)), int(m.group(2))] == exp_both["strict_units"] and [int(m.group(3)), int(m.group(4))] == exp_both["lax_units"]
    m = re.search(r"LAX_ALPHA, LAX_BETA = ([0-9.]+), ([0-9.]+)", r5lib)
    F["alphaL"] = float(m.group(1))
    assert float(m.group(2)) == 1 / expo["ramp_width"]
    F["alpha"], F["kappa"] = alpha, kappa
    F["pctL"] = 100 * expo["fit"]["trfit_h"]["below_h01"]  # share of harmful fit prompts below the Loosen threshold
    assert close(expo["thresholds"]["h01"], exp_both["theta"], 1e-4)
    # what each direction was computed from
    v1 = J("round5/fit/v1again.json")
    f1 = {**v1["fit_harmful_sets"], **v1["fit_harmless_sets"]}
    f3 = {**lab_v3g["fit_harmful"], **lab_v3g["fit_harmless"]}
    assert all(f3.get(k) == n for k, n in f1.items()), "AyaDial's fit must contain the earlier version's"
    F["wj"] = (v1["fit_harmful_sets"]["trfit_h"], v1["fit_harmless_sets"]["trfit_b"])
    F["v1_other"] = sum(n for k, n in f1.items() if not k.startswith("trfit_"))
    F["v1_other_src"] = list(dict.fromkeys(src(k) for k in f1 if not k.startswith("trfit_")))
    added = {k: n for k, n in f3.items() if k not in f1}
    groups = {}
    for k, n in added.items():
        groups[src(k)] = groups.get(src(k), 0) + n
    F["added"] = groups
    F["fit_langs"] = len({k.split("_")[1] for k in added if k.startswith("mj_")})
    assert F["fit_langs"] == len({k.split("_")[1] for k in added if k.startswith("aya_")})
    F["half"] = all(k.endswith(":a") for k in added if k.startswith(("itw_", "mj_", "aya_")))
    F["guard3"] = lab_v3g["guard"]
    # settings and the control, from the frozen declaration and the hook code
    decl = {c["name"]: c["spec"] for c in J("round5/frozen_declaration.json")["cells"]}
    F["set"] = {"Control": decl["ungated@0.1"]["scale"], "ControlL": decl["ctrlneg@-0.02"]["scale"], "Earlier": decl["foldcap@1"]["s"],
                "AyaDial": decl["v3gcap@1"]["s"], "Loosen": decl["lax3@2"]["t"], "TL": (decl["both3@1"]["s"], decl["both3@1"]["t"])}
    assert decl["both3@1"]["svec"] == decl["v3gcap@1"]["vec"] == decl["lax3@2"]["vec"]  # Tighten + Loosen uses AyaDial's vectors
    assert "pos" not in decl["ungated@0.1"] and "pos" not in decl["ctrlneg@-0.02"]
    r4lib = (B.BUILD / "scripts/r4lib.py").read_text()
    assert 'pos=c.get("pos", "all")' in r5lib and "'all' = every position incl. generated tokens" in r4lib
    assert re.search(r"class Steer:.*?write_layer=23", r4lib, re.S)  # hidden index 23 = output of layer 22
    # formats and the two-adapter split
    frz = J("round5/v1_freeze.json")["sha256"]
    F["v1_formats"] = [f for f in ("adapter_model.safetensors", ".gguf") if all(any(k.endswith(f) and part in k for k in frz) for part in ("install", "knob"))]
    assert F["v1_formats"] == ["adapter_model.safetensors", ".gguf"]
    F["r5_gguf"] = all(f"{a}_{b}_sha256" in exp_both for a in ("strict", "lax") for b in ("install", "knob"))
    assert F["r5_gguf"]
    shapes = exp_cap["adapter_shapes"]
    F["install_mods"] = sorted({k.split(".")[0] for k in shapes}, key=["gate_proj", "up_proj", "down_proj"].index)
    F["install_rank"] = {shapes[f"{mm}.lora_A.weight"][0] for mm in F["install_mods"]}
    assert len(F["install_rank"]) == 1
    F["install_rank"] = F["install_rank"].pop()
    summ4 = (B.RES / "round4/summary.md").read_text()
    m = re.search(r"`knob` adapter \(rank (\d+) on (\w+) only\)", summ4)
    F["knob_rank"], F["knob_mod"] = int(m.group(1)), m.group(2)
    assert int(re.search(r"knob\.gguf: n_tensors = (\d+)", exp_cap["convert_knob"]).group(1)) == 2  # one module: lora_A and lora_B
    return F


def harm_rows(table, spec, cells, rnd, claims):
    out = []
    for label, row, _ in spec:
        vals, n = {}, None
        for name, c in cells:
            s = tshares(rnd, table, row, c)
            vals[name] = s["unsafe"]
            n = s["n"]
        cl = claims[row]
        # the claims file and the summary table must agree with the label files
        assert close(cl["unsafe"], vals["AyaDial"]) and close(cl["base_unsafe"], vals["Base"]) and cl["n"] == n, row
        tcell = next(r for r in J(table)["rows"] if r["row"] == row)["cells"]["v3gcap@1"]["vs_control"]
        assert close(tcell["unsafe"], cl["vs_control"]) and close(tcell["unsafe_ci"][0], cl["vs_control_ci"][0]), row
        out.append({"label": label, "row": row, "n": n, "vals": vals, "d": cl["vs_control"], "ci": cl["vs_control_ci"], "metric": "unsafe",
                    "db": cl["vs_base"], "dbci": cl["vs_base_ci"], "test": "first" if rnd == "round5" else "second", "table": table, "rnd": rnd})
    return out


def harmless_rows(table, spec, cells, rnd):
    out = []
    for label, row in spec:
        vals, n = {}, None
        for name, c in cells:
            s = tshares(rnd, table, row, c)
            vals[name] = s["refusal"]
            n = s["n"]
        vc = next(r for r in J(table)["rows"] if r["row"] == row)["cells"]["v3gcap@1"]
        out.append({"label": label, "row": row, "n": n, "vals": vals, "d": vc["vs_control"]["refusal"], "ci": vc["vs_control"]["refusal_ci"], "metric": "refusal",
                    "test": "first" if rnd == "round5" else "second", "table": table, "rnd": rnd,
                    "db": vc["vs_base"]["refusal"], "dbci": vc["vs_base"]["refusal_ci"]})
    return out


def lax_rows(table, spec, cells, metric):
    out = []
    for label, row in spec:
        vals, n = {}, None
        for name, c in cells:
            s = tshares("round5", table, row, c)
            vals[name] = s[metric]
            n = s["n"]
        vc = next(r for r in J(table)["rows"] if r["row"] == row)["cells"]["lax3@2"]
        out.append({"label": label, "n": n, "vals": vals, "db": vc["vs_base"][metric], "dbci": vc["vs_base"][metric + "_ci"],
                    "d": vc["vs_control"][metric], "ci": vc["vs_control"][metric + "_ci"]})
    return out


# LaTeX
PRE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[a4paper,margin=2.1cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{makecell}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{needspace}
\usepackage{float}
\usepackage{xcolor}
\usepackage{colortbl}
\definecolor{win}{HTML}{CDEBD2}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc}
\pgfplotsset{compat=1.18}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\usepackage[hidelinks]{hyperref}
\usepackage{etoolbox}
\pretocmd{\section}{\ifnum\value{section}>0 \clearpage\fi}{}{}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\renewcommand{\arraystretch}{1.12}
"""


def colours():
    return "".join(f"\\definecolor{{{CNAME[k]}}}{{HTML}}{{{v}}}\n" for k, v in COL.items())


def tex(S):
    d = S["dev"]
    o = [PRE, colours(), r"""\pgfplotsset{every axis/.append style={font=\small, tick label style={font=\small}, label style={font=\small},
  legend style={font=\small, draw=none, fill=white, fill opacity=0.9, text opacity=1}, axis line style={black!60}, tick style={black!60}},
  every axis plot/.append style={line width=1pt}}
\begin{document}
"""]
    # ---------- 1. title
    o.append(r"""\begin{center}
{\LARGE\bfseries A Per-Request Strictness Knob in the Weights\\[3pt] of a Small Open Model\par}
\vspace{8pt}
{\large Model: \texttt{CohereLabs/tiny-aya-global}}\\[3pt]
Lexsi AI Safety and Alignment Hackathon, Track 2
\end{center}
\vspace{4pt}
""")
    o.append("AyaDial lowers unsafe replies to harmful requests while leaving harmless requests almost untouched. "
             f"It rewires two dormant MLP units of layer {S['layer']} in closed form, with no training, and its strength $s$, the knob of the title, is set per request. ")
    w = S["wsp"]["blocks"]["sys_long"]
    o.append(f"A weight edit written by a frontier LLM (Section~\\ref{{sec:wsp}}) lowers unsafe replies to WildJailbreak requests under a long system prompt from "
             f"{num(w['H']['rates']['A']['unsafe'])}\\% to {num(w['H']['rates']['B']['unsafe'])}\\% ({dci(w['H']['B_minus_A']['main']['diff'], w['H']['B_minus_A']['main']['ci95'])} points), "
             f"with harmless refusal at {num(w['B']['rates']['A']['refused'])}\\% and {num(w['B']['rates']['B']['refused'])}\\%; these are requests from the pool the editor worked with, not the set-aside tests.\n\n")
    o.append(r"\textbf{Labels.} Every rate in this report is the share of replies that a model judge (Qwen3Guard-Gen-0.6B) labels Unsafe (the reply contains harmful content) or Refused (the reply declines). "
             r"No reply has a human label yet. "
             r"A filled cell marks a comparison won by one of our adapters, and bold without fill one won by another method; a win is a lead of at least 1 point, and Base is the reference and is never marked." + "\n\n")

    # ---------- the adapters (one page)
    o.append(r"\clearpage" + "\n")
    o.append(adapters_page(S))

    # ---------- main result
    cal = S["calib"]
    o.append(r"\section{Main result on the benchmark it was built on}\label{sec:main}" + "\n")
    o.append(f"On this benchmark AyaDial lowers unsafe replies from {num(d['Base']['H']['unsafe'])}\\% to {num(d['AyaDial']['H']['unsafe'])}\\% "
             f"while harmless refusal moves from {num(d['Base']['B']['refusal'])}\\% to {num(d['AyaDial']['B']['refusal'])}\\% (at $s = 1$). ")
    o.append("Control is the same push along the same refusal direction with no harm gate (scale 0.1). "
             "The steering baselines are the three methods of SafeTune (Lexsi Labs), run with its own code. SafeTune RefusalDirection adds the refusal direction at every token; SafeTune AdaSteer sets the steering strength per input; "
             "SafeTune CAST steers only when the request matches a condition vector. "
             f"For this table they were calibrated on {sum(n for n, _ in cal['harmful'])} harmful requests ({calib_text(cal['harmful'])}) and "
             f"{sum(n for n, _ in cal['harmless'])} harmless requests ({calib_text(cal['harmless'])}), drawn with a fixed seed.\n\n")
    o.append(bench_line(S, "WildJailbreak",
                        f"{S['nH']} jailbreak-wrapped harmful and {S['nB']} jailbreak-wrapped harmless requests from the same benchmark the adapters were computed from, "
                        "also used to choose their settings, so these numbers are the easiest case."))
    o.append(main_tables(S))
    o.append(scatter(S))

    # ---------- the scale s, a subsection of the main result
    o.append(r"\subsection{The scale $s$}" + "\n")
    o.append(f"AyaDial was measured at $s$ = {', '.join(f'{v:g}' for v, _, _ in S['dial_pts'])} on the same {S['nH'] + S['nB']} WildJailbreak requests. "
             "The full sweep from $s = 0$ to 3 was measured on the earlier version of the adapter and is in Appendix~\\ref{app:earlier}.\n")
    o.append(dial(S))

    # ---------- new requests from known sources
    t1 = {r["row"]: r for r in S["t1H"] + S["t1B"]}
    t2 = {r["row"]: r for r in S["t2H"] + S["t2B"]}
    sor = {r["row"]: r for r in S["sorry"]}
    o.append(r"\section{New requests from known sources}\label{sec:t1}" + "\n")
    o.append("These are sources that contributed samples to AyaDial's direction, tested on requests and templates it has not seen. "
             "They show protection against requests in the style of sources the adapter has samples of.\n\n")
    o.append("The requests in this section and in Sections~\\ref{sec:t2}, \\ref{sec:loosen} and~\\ref{sec:open} were set aside before the adapters were computed and were never used to build or tune them; "
             "the list of requests and the settings were fixed in advance, and each of the two tests (the first and the second) was run once. They are called unseen requests below. "
             f"AyaDial's harm direction was computed from {sources_text(S)}; every request tested here is a different one. "
             "Every benchmark that was run, including those where AyaDial is not shown to differ from Control, is in Appendix~\\ref{app:all}.\n\n")
    o.append(section_legend(S))
    o.append(bars(S, ("WildJailbreak", "In-the-wild", "GCG", "MultiJail")))
    o.append(block(S, "WildJailbreak (new requests)", "jailbreak-wrapped requests from the WildJailbreak benchmark, first test",
                   t1["adversarial_harmful (fresh WildJailbreak)"], t1["adversarial_benign (fresh WildJailbreak)"]))
    o.append(block(S, "in-the-wild jailbreak templates", "jailbreak prompts collected from online communities, with a harmful question or a harmless instruction, first test",
                   t1["in-the-wild template + harmful question"], t1["in-the-wild template + harmless instruction"]))
    o.append(block(S, "JailbreakBench GCG", "an optimised attack suffix on behaviours AyaDial never saw; the harmless set carries a transplanted suffix, first test",
                   t1["JailbreakBench GCG (held-out behaviours)"], t1["harmless request + transplanted suffix"]))
    o.append(block(S, "MultiJail, 9 languages", "harmful questions in nine non-English languages, with Aya prompts in the same languages as the harmless set, first test",
                   t1["MultiJail 9 other languages"], t1["harmless prompts, 9 other languages (Aya)"]))

    # ---------- benchmarks never seen
    six, q8 = S["six"], S["q8"]
    o.append(r"\section{Benchmarks never seen}\label{sec:t2}" + "\n")
    o.append("No sample of these sources entered any computation of AyaDial. "
             "They test whether its harm score carries over to kinds of requests it has never met.\n\n")
    o.append(section_legend(S))
    o.append(block(S, "Aya red-teaming, six languages",
                   "harmful requests written by native speakers in Filipino, French, Hindi, Russian, Serbian and Spanish, second test; the harmless set is Aya prompts in the same languages, from a source AyaDial saw in ten other languages",
                   t2["red-teaming prompts, 6 unseen languages"], t2["harmless prompts, 6 unseen languages"],
                   extra=f"A second judge, Qwen3Guard-Gen-8B, gives {num(q8['base_unsafe'])}\\% to {num(q8['unsafe'])}\\% unsafe ({dci(q8['vs_control'], q8['vs_control_ci'])} against Control)."))
    exc = [n for n, _, _, _, e, _ in S["langs"] if e]
    o.append(langs_fig(S, exc))
    o.append(block(S, "JailbreakBench random search", "an attack method AyaDial never saw, run on behaviours it never saw, first test",
                   t1["JailbreakBench RS (held-out behaviours)"], None))
    o.append(block(S, "SORRY-Bench, base questions", "harmful questions asked plainly, first test", sor["SORRY-Bench base"], None))
    o.append(block(S, "SORRY-Bench, other styles", "the same kind of questions rewritten in 20 styles (persuasion, role play, misspellings, ciphers, translations), first test",
                   sor["SORRY-Bench other styles (persuasion, role play, misspellings, ciphers, translations)"], None))
    o.append(block(S, "PHTest", "harmless requests that look harmful, first test", None, t1["PHTest harmless"]))

    # ---------- Loosen
    o.append(r"\section{Loosen: fewer false refusals}\label{sec:loosen}" + "\n")
    o.append("Loosen is the other half of selectivity: it cuts refusals of harmless requests that only sound harmful. ")
    o.append(f"Loosen uses two other dormant units and pushes away from refusal when the harm score is below $\\theta_L$. "
             "Its Control is the same push with no gate (scale 0.02). All sets are unseen requests from the first test (see Section~\\ref{sec:t1}).\n\n")
    o.append(bench_line(S, "OR-Bench-hard, PHTest, Alpaca, Aya, WildJailbreak, in-the-wild templates, MultiJail",
                        "Harmless sets measure false refusals; harmful sets (WildJailbreak, in-the-wild templates, MultiJail) check that nothing is released."))
    o.append(lax_table(S))
    orb, orbu = S["orb"], S["orbu"]
    o.append(f"On OR-Bench-hard, false refusals fall from {num(orb['vals']['Base'])}\\% to {num(orb['vals']['Loosen'])}\\% "
             f"({dci(orb['db'], orb['dbci'])} against Base). The judge's unsafe label on those harmless replies rises from "
             f"{num(orbu['vals']['Base'])}\\% to {num(orbu['vals']['Loosen'])}\\% ({dci(orbu['db'], orbu['dbci'])} against Base). "
             "On the harmful sets, the intervals against Base bound any release at about "
             f"{num(max(r['dbci'][1] for r in S['LH']))} points.\n")

    # ---------- limits
    mj5, mj6, en6, itw5, itw6 = S["mj5"], S["mj6"], S["en6"], S["itw5"], S["itw6"]
    ll = S["llama"]
    o.append(r"\section{Limits}" + "\n" + r"\begin{itemize}\setlength{\itemsep}{1pt}" + "\n")
    lim = [
        f"Judge labels only: a second judge (Qwen3Guard-Gen-8B) gives {num(q8['base_unsafe'])}\\% to {num(q8['unsafe'])}\\% on the six languages, and the main judge marks {num(S['itwb_unsafe'])}\\% of Base replies to harmless templates unsafe.",
        "System prompts and multi-turn chats: the open problem of Section~\\ref{sec:open}.",
        "SORRY-Bench: lower than Base but not shown to differ from Control; " + "; ".join(f"{lab} {dci(r['d'], r['ci'])}" for lab, r in
            (("base questions", {x["label"]: x for x in S["sorry"]}["SORRY-Bench base questions"]), ("other styles", {x["label"]: x for x in S["sorry"]}["SORRY-Bench other styles"]))) + " against Control.",
        f"MultiJail, 9 languages: {dci(mj5['vs_control'], mj5['vs_control_ci'])} against Control in the first test, {dci(mj6['vs_control'], mj6['vs_control_ci'])} on new requests in the second test.",
        f"Plain harmful requests in English: {dci(en6['vs_base'], en6['vs_base_ci'])} against Base; Control is lower by {num(en6['vs_control'])} {ci(en6['vs_control_ci'])}.",
        f"Harmless requests in in-the-wild templates: refusal rises {dci(itw5['counterpart_refusal_rise'], itw5['counterpart_refusal_rise_ci'])} (first test) and {dci(itw6['counterpart_refusal_rise'], itw6['counterpart_refusal_rise_ci'])} (second test) against Base.",
        f"llama.cpp: {ll['identical_to_all_scales_1']} of {ll['of']} requests sent without per-request scales got every adapter at full scale, Loosen included.",
    ]
    o.append("".join(f"\\item {x}\n" for x in lim))
    o.append(r"\end{itemize}" + "\n")

    # ---------- open problem: deployment formats
    o.append(open_problem(S))

    # ---------- weight-space programming, then the appendix
    o.append(wsp_section(S))
    o.append(summary_section(S))
    o.append(appendix_earlier(S))
    o.append(appendix_all(S))
    o.append(r"\end{document}" + "\n")
    return "".join(o)


PENDING_SENTENCE = ("SafeTune RefusalDirection, AdaSteer and CAST were calibrated on the same 7555 requests AyaDial's direction was computed from, and their setting was chosen on development data with the same rule as AyaDial's (RefusalDirection: strength 0.1 added at every layer, SafeTune's default placement; AdaSteer: bound 0.2; CAST: alpha 0.03).")


def open_problem(S):
    fmt, fb = S["fmt"], S["fmt_b"]
    d0, dl, n = S["fire_fmt"]
    o = [r"\section{Open problem: system prompts and multi-turn chats}\label{sec:open}" + "\n"]
    o.append("In real use a model sits behind a developer's system prompt or inside a longer conversation. "
             f"In those formats AyaDial lowers unsafe replies against Base ({dci(fmt['vs_base'], fmt['vs_base_ci'])} points) but is not shown to differ from Control "
             f"({dci(fmt['vs_control'], fmt['vs_control_ci'])}), while it avoids Control's extra refusals of harmless requests ({dci(fb['d'], fb['ci'])} against Control, "
             f"{dci(fb['db'], fb['dbci'])} against Base). "
             f"The measured reason: on a separate set of requests, AyaDial's harm score passes its threshold on {num(d0)}\\% of harmful requests in the default format "
             f"and on {num(dl)}\\% behind a long system prompt. "
             "Section~\\ref{sec:wsp} shows what a weight edit written by a frontier LLM did on this problem.\n\n")
    o.append(f"Harmful requests come from {' and '.join(S['fmt_src_h'])}, harmless requests from {', '.join(S['fmt_src_b'][:-1])} and {S['fmt_src_b'][-1]}; "
             "they are unseen requests from the second test.\n\n")
    o.append(section_legend(S))
    o.append(fmt_fig(S))
    t2 = {r["row"]: r for r in S["t2H"] + S["t2B"]}
    for name, key, what in (("all eight formats", "all", "four system prompts, three multi-turn chats and one in-the-wild template as the system prompt"),
                            ("four system prompts", "four developer preambles", "the request placed behind one of four system prompts"),
                            ("three multi-turn chats", "three multi-turn", "the request placed after earlier turns of a conversation")):
        o.append(block(S, name, what, t2[f"exam formats, {key}: harmful requests"], t2[f"exam formats, {key}: harmless requests"]))
    return "".join(o)


def fmt_fig(S):
    """Three format rows, harmful Unsafe and harmless Refused, Base / Control / AyaDial."""
    names = ["Base", "Control", "AyaDial"]
    short = {"exam formats, all: ": "All eight", "exam formats, four developer preambles: ": "System prompts", "exam formats, three multi-turn: ": "Multi-turn"}
    fmt_rows = lambda rows: [r for r in rows if r["row"].startswith("exam formats")]

    def lab(r):
        return next(v for k, v in short.items() if r["row"].startswith(k))

    def panel(rows, ylabel, legend):
        coords = [f"f{i}" for i in range(len(rows))]
        ymax = max(r["vals"][n] for r in rows for n in names) * 1.2
        o = [r"\nextgroupplot[ybar=0.6pt, bar width=7pt, symbolic x coords={" + ",".join(coords) + r"}, xtick=data, xticklabels={" + ",".join("{" + lab(r) + "}" for r in rows) + "}, "
             + f"ymin=0, ymax={ymax:.1f}, ylabel={{{ylabel}}}" + (r", legend to name=fmtleg, legend columns=3, legend style={/tikz/every even column/.append style={column sep=8pt}}" if legend else "") + "]\n"]
        for n in names:
            o.append(f"\\addplot[fill={CNAME[n]}, draw=none, area legend] coordinates {{" + " ".join(f"({c},{r['vals'][n]:.3f})" for c, r in zip(coords, rows)) + "};"
                     + (f"\\addlegendentry{{{n}}}" if legend else "") + "\n")
        return "".join(o)

    return (r"""\begin{figure}[H]\centering
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.8cm}, width=0.47\linewidth, height=4.6cm,
  enlarge x limits=0.25, ymajorgrids, grid style={black!8}, tick label style={font=\footnotesize}]
""" + panel(fmt_rows(S["t2H_fmt"]), r"Harmful unsafe (\%) $\downarrow$", True) + panel(fmt_rows(S["t2B_fmt"]), r"Harmless refused (\%) $\downarrow$", False) + r"""\end{groupplot}
\node[anchor=south] at ($(group c1r1.north)!0.5!(group c2r1.north)+(0,4pt)$) {\pgfplotslegendfromname{fmtleg}};
\end{tikzpicture}
\caption{The three format blocks below at a glance: harmful requests on the left, harmless on the right.}\label{fig:fmt}
\end{figure}
""")


ST_ROWS = [("RefusalDirection", "SafeTune RefusalDirection"), ("AdaSteer", "SafeTune AdaSteer"), ("CAST", "SafeTune CAST")]
WSP_ROW = [("AyaDial-WSP", "AyaDial-WSP")]
SUMMARY = []  # (benchmark, kind, n, {method: value}) collected by block() for the closing table


def section_legend(S):
    return (f"Labels: {S['judge']} (model judge), for both the refusal and the unsafe label. $\\downarrow$ lower is better, $\\uparrow$ higher is better. "
            "Each benchmark below has its own table. " + PENDING_SENTENCE + " " + wsp_sentence(S) + "\n\n")


def wsp_sentence(S):
    if any("AyaDial-WSP" in S["st_tests"].get(t, {}) for t in ("first", "second")):
        return ("AyaDial-WSP is the weight edit of Section~\\ref{sec:wsp}, written into the same two units at $s = 1$; it was made after these tests had been opened, on other requests, "
                "so its rows are a later run and not a setting fixed in advance.")
    return "AyaDial-WSP (the weight edit of Section~\\ref{sec:wsp}) is being run on the same requests; its cells read pending."


def summary_section(S):
    """Closing table: every benchmark of Sections 3, 4 and 6, our two adapters against every other method."""
    cols = [("AyaDial", "AyaDial"), ("AyaDial-WSP", r"\makecell{AyaDial-\\WSP}"), ("Base", "Base"), ("Control", "Control"),
            ("RefusalDirection", r"\makecell{SafeTune\\Refusal-\\Direction}"), ("AdaSteer", r"\makecell{SafeTune\\AdaSteer}"), ("CAST", r"\makecell{SafeTune\\CAST}")]
    o = [r"\section{Summary of all results}\label{sec:summary}" + "\n"]
    o.append("One row per benchmark of Sections~\\ref{sec:t1}, \\ref{sec:t2} and~\\ref{sec:open}; our two adapters are on the left, every other method on the right. "
             "A filled cell is a comparison won by one of our adapters, bold without fill one won by another method (a lead of at least 1 point over the other side; Base is not a contestant). "
             + wsp_sentence(S) + "\n\n")
    for kind, title in (("H", r"Harmful requests: Unsafe replies (\%) $\downarrow$"), ("B", r"Harmless requests: Refused (\%) $\downarrow$")):
        rows = [r for r in SUMMARY if r[1] == kind]
        o.append(r"\begin{table}[H]\centering\footnotesize\renewcommand{\arraystretch}{1.3}\setlength{\tabcolsep}{4pt}\begin{tabular}{|l|r|r|r||r|r|r|r|r|}\hline" + "\n")
        o.append(f"\\multicolumn{{9}}{{|l|}}{{\\textbf{{{title}}}}}\\\\\\hline\n")
        o.append(r"\multicolumn{2}{|c|}{} & \multicolumn{2}{c||}{\textbf{Ours}} & \multicolumn{5}{c|}{\textbf{Other methods}}\\\hline" + "\n")
        o.append("Benchmark & $n$ & " + " & ".join(h for _, h in cols) + "\\\\\\hline\n")
        wins = 0
        for name, _, n, v, *_ in rows:
            c = [(m, m, v[m]) for m, _ in cols if m != "Base" and m in v]
            b = dict(zip([x[0] for x in c], mark("Summary, " + ("harmful" if kind == "H" else "harmless"), name, c)))
            wins += any(x == "ours" for x in b.values())
            cells = [(numb(v[m], b.get(m, False)) if m in v else r"\textit{pending}") for m, _ in cols]
            o.append(f"{name[0].upper() + name[1:]} & {n} & " + " & ".join(cells) + "\\\\\\hline\n")
        o.append(r"\end{tabular}\end{table}" + "\n")
        o.append(f"One of our adapters wins {wins} of these {len(rows)} comparisons by at least 1 point.\n\n")
    return "".join(o)



def block(S, name, what, H, Bh, extra=""):
    """One benchmark: the bold benchmark line, the harmful table (three outcomes) and the harmless table (two outcomes), rows are methods."""
    n_txt = ", ".join(f"{x['n']} {k}" for x, k in ((H, "harmful"), (Bh, "harmless")) if x)
    o = [f"\\needspace{{18\\baselineskip}}\\medskip{{\\raggedright\\noindent\\large\\bfseries Benchmark: {name}\\par}}\\vspace{{-3pt}}\\noindent {n_txt} requests: {what}.\n\n"]
    base = {"Base": "base", "Control": "ungated@0.1", "AyaDial": "v3gcap@1"}
    if H:
        sh = {}
        for m, c in base.items():
            x = tshares(H["rnd"], H["table"], H["row"], c)  # per-request labels, checked against the summary table
            assert abs(x["refused_only"] + x["answered"] + x["unsafe"] - 100) < 1e-9 and x["n"] == H["n"], (H["row"], c)
            sh[m] = {"refused": x["refused_only"], "answered": x["answered"], "unsafe": x["unsafe"]}
        assert close(sh["AyaDial"]["unsafe"], H["vals"]["AyaDial"]) and close(sh["Control"]["unsafe"], H["vals"]["Control"])
        for m, _ in WSP_ROW + ST_ROWS:
            sh[m] = {k: v for k, v in st_cell(S, m, H).items() if k in ("refused", "answered", "unsafe")}
        rows = [("Base", "Base")] + [("Control", "Control"), ("AyaDial", "AyaDial")] + WSP_ROW + ST_ROWS
        SUMMARY.append((name, "H", H["n"], {m: sh[m]["unsafe"] for m, _ in rows if "unsafe" in sh[m]}, H["test"], H["row"]))
        live = [(m, lab) for m, lab in rows if m != "Base"]
        bold = {}
        for key, higher in (("refused", True), ("unsafe", False)):
            c = [(m, lab, sh[m][key]) for m, lab in live if key in sh[m]]
            for (m, _, _), b in zip(c, mark(name + ", harmful", key, c, higher=higher)):
                bold[(m, key)] = b
        o.append(r"\begin{table}[H]\centering\small\begin{tabular}{@{}lrrrl@{}}\toprule" + "\n")
        o.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{Harmful requests ($n = {H['n']}$)}}}}\\\\\n")
        o.append(r"Method & Refused (\%) $\uparrow$ & \makecell[r]{Answered without\\harm (\%)} & Unsafe (\%) $\downarrow$ & \makecell[l]{Unsafe, difference from\\Control (95\% interval)}\\\midrule" + "\n")
        for m, lab in rows:
            cells = [numb(sh[m][k], bold.get((m, k), False) if k != "answered" else False) if k in sh[m] else r"\textit{pending}" for k in ("refused", "answered", "unsafe")]
            o.append(f"{lab} & " + " & ".join(cells) + f" & {dci(H['d'], H['ci']) if m == 'AyaDial' else ''}\\\\\n")
        o.append(r"\bottomrule\end{tabular}\end{table}" + "\n")
    if Bh:
        sb = {}
        for m, c in base.items():
            x = tshares(Bh["rnd"], Bh["table"], Bh["row"], c)
            assert x["n"] == Bh["n"] and close(x["refusal"], Bh["vals"][m]), (Bh["row"], c)
            sb[m] = {"refusal": x["refusal"]}
        for m, _ in WSP_ROW + ST_ROWS:
            sb[m] = {k: v for k, v in st_cell(S, m, Bh).items() if k == "refusal"}
        rows = [("Base", "Base"), ("Control", "Control"), ("AyaDial", "AyaDial")] + WSP_ROW + ST_ROWS
        SUMMARY.append((name, "B", Bh["n"], {m: sb[m]["refusal"] for m, _ in rows if "refusal" in sb[m]}, Bh["test"], Bh["row"]))
        c = [(m, lab, sb[m]["refusal"]) for m, lab in rows if m != "Base" and "refusal" in sb[m]]
        bref = dict(zip([x[0] for x in c], mark(name + ", harmless", "refused", c)))
        bans = dict(zip([x[0] for x in c], mark(name + ", harmless", "answered", [(m, lab, 100 - v) for m, lab, v in c], higher=True)))
        o.append(r"\begin{table}[H]\centering\small\begin{tabular}{@{}lrrl@{}}\toprule" + "\n")
        o.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textbf{{Harmless requests ($n = {Bh['n']}$)}}}}\\\\\n")
        o.append(r"Method & Refused (\%) $\downarrow$ & Answered (\%) $\uparrow$ & \makecell[l]{Refused, difference from\\Control (95\% interval)}\\\midrule" + "\n")
        for m, lab in rows:
            if "refusal" in sb[m]:
                v = sb[m]["refusal"]
                cells = [numb(v, bref.get(m, False)), numb(100 - v, bans.get(m, False))]  # Answered is the complement of the refusal label
            else:
                cells = [r"\textit{pending}"] * 2
            o.append(f"{lab} & " + " & ".join(cells) + f" & {dci(Bh['d'], Bh['ci']) if m == 'AyaDial' else ''}\\\\\n")
        o.append(r"\bottomrule\end{tabular}\end{table}" + "\n")
    if H:
        v = verdict(H)
        o.append(f"Unsafe, AyaDial against Control: {dci(H['d'], H['ci'])}, " + {"established against Control": "established against Control (the interval excludes 0).",
                 "lower than Base, not shown to differ from Control": "lower than Base, not shown to differ from Control.",
                 "not shown to differ from Base or Control": "not shown to differ from Base or Control."}[v] + (" " + extra if extra else "") + "\n\n")
    elif Bh:
        word = lambda c, who: (f"higher than {who}" if c[0] > 0 else f"lower than {who}" if c[1] < 0 else f"not shown to differ from {who}")
        o.append(f"Refused, AyaDial against Control: {dci(Bh['d'], Bh['ci'])}, {word(Bh['ci'], 'Control')}; "
                 f"against Base: {dci(Bh['db'], Bh['dbci'])}, {word(Bh['dbci'], 'Base')}.\n\n")
    return "".join(o)


def verdict(r):
    """What the files say against Control and against Base (95% intervals); an interval that touches 0 does not exclude it."""
    if r["ci"][1] < 0:
        return "established against Control"
    if r["dbci"][1] < 0:
        return "lower than Base, not shown to differ from Control"
    return "not shown to differ from Base or Control"


def verdicts(H):
    groups = {}
    for r in H:
        groups.setdefault(verdict(r), []).append(r["label"].replace("$^{b}$", ""))
    order = [("established against Control", "Established against Control (the interval excludes 0)"),
             ("lower than Base, not shown to differ from Control", "Lower than Base, not shown to differ from Control"),
             ("not shown to differ from Base or Control", "Not shown to differ from Base or Control")]
    return " ".join(f"{text}: {', '.join(groups[v])}." for v, text in order if v in groups) + "\n\n"


def public_name(row):
    """Plain names for the summary-table rows (no internal words)."""
    import re
    fixed = {"adversarial_harmful (fresh WildJailbreak)": "WildJailbreak, new requests", "adversarial_benign (fresh WildJailbreak)": "WildJailbreak, new harmless",
             "orbench_hard (fresh)": "OR-Bench-hard", "Alpaca instructions (fresh)": "Alpaca",
             "in-the-wild template + harmful question": "In-the-wild templates", "in-the-wild template + harmless instruction": "In-the-wild templates, harmless",
             "MultiJail 9 other languages": "MultiJail, 9 languages", "harmless prompts, English (Aya)": "Aya prompts, English",
             "harmless prompts, 9 other languages (Aya)": "Aya prompts, 9 languages", "harmful request + transplanted suffix": "Harmful request + transplanted suffix",
             "harmless request + transplanted suffix": "Harmless request + transplanted suffix", "SORRY-Bench base": "SORRY-Bench, base questions",
             "red-teaming prompts, 6 unseen languages": "Aya red-teaming, six languages", "harmless prompts, 6 unseen languages": "Aya prompts, six languages",
             "red-teaming prompts, Arabic and English (seen languages)": "Aya red-teaming, Arabic and English",
             "exam formats, all: harmful requests": "Formats, all eight, harmful", "exam formats, all: harmless requests": "Formats, all eight, harmless",
             "exam formats, four developer preambles: harmful requests": "Four system-prompt formats, harmful",
             "exam formats, four developer preambles: harmless requests": "Four system-prompt formats, harmless",
             "exam formats, three multi-turn: harmful requests": "Three multi-turn formats, harmful", "exam formats, three multi-turn: harmless requests": "Three multi-turn formats, harmless"}
    LANG = {"fil": "Filipino", "fr": "French", "hi": "Hindi", "ru": "Russian", "sr": "Serbian", "es": "Spanish"}
    if row in fixed:
        return fixed[row]
    m = re.match(r"JailbreakBench (\w+) \(held-out behaviours\)", row)
    if m:
        return "JailbreakBench " + {"RS": "random search"}.get(m.group(1), m.group(1))
    m = re.match(r"MultiJail (high|medium|low)-resource \((.*)\)", row)
    if m:
        return f"MultiJail, {m.group(1)}-resource ({m.group(2)})"
    m = re.match(r"MultiJail ([a-z]{2})$", row)
    if m:
        return f"MultiJail, {m.group(1)}"
    m = re.match(r"harmless prompts (high|medium|low)-resource$", row)
    if m:
        return f"Aya prompts, {m.group(1)}-resource"
    m = re.match(r"harmless prompts, ([a-z]+)$", row)
    if m:
        return f"Aya prompts, {LANG[m.group(1)]}"
    m = re.match(r"red-teaming prompts, (\w+)$", row)
    if m:
        return f"Aya red-teaming, {LANG.get(m.group(1), m.group(1))}"
    m = re.match(r"format: (.*): (harmful|harmless) requests$", row)
    if m:
        return f"Format, {m.group(1).replace('developer preamble', 'system prompt')}: {m.group(2)}"
    m = re.match(r"SORRY-Bench other styles", row)
    if m:
        return "SORRY-Bench, other styles"
    m = re.match(r"SORRY-Bench (\w+)$", row)
    if m:
        return f"SORRY-Bench, {m.group(1)}"
    return row


def appendix_all(S):
    """Appendix B: every set of the two tests, Base / Control / AyaDial, AyaDial minus Control with its 95% interval. No bold."""
    o = [r"\section{All benchmarks run}\label{app:all}" + "\n"]
    o.append("Every harmful and harmless set of the two tests, including those where AyaDial is not shown to differ from Control. "
             "Harmful rows show Unsafe, harmless rows show Refused; PHTest controversial shows Refused.\n\n")
    import re
    bad = ("exam", "held-out", "fresh", "developer", "dev ", "unseen languages")
    parts = []
    for test, table, claims, rnd in (("first", S["T5"], S["c5"], "round5"), ("second", S["T6"], S["c6"], "round6")):
        rows = []
        for r in J(table)["rows"]:
            metric = "unsafe" if r["kind"] == "H" else "refusal"
            vals = {n: tshares(rnd, table, r["row"], c)[metric] for n, c in (("Base", "base"), ("Control", "ungated@0.1"), ("AyaDial", "v3gcap@1"))}
            vc = r["cells"]["v3gcap@1"]["vs_control"]
            dd, cc = vc[metric], vc[metric + "_ci"]
            if r["kind"] == "H" and r["row"] in claims:
                assert close(claims[r["row"]]["vs_control"], dd) and close(claims[r["row"]]["vs_control_ci"][1], cc[1]), r["row"]
            name = re.sub(r"(,|:) (harmful|harmless)$", "", public_name(r["row"])) if r["row"].startswith(("format", "exam")) else public_name(r["row"])
            assert not any(b in name.lower() for b in bad), name
            rows.append((r["kind"], name, tshares(rnd, table, r["row"], "base")["n"], vals, dd, cc))
        for kind, title in (("H", "harmful, Unsafe \\% $\\downarrow$"), ("B", "harmless, Refused \\% $\\downarrow$"), ("C", "controversial, Refused \\%")):
            rr = [x for x in rows if x[0] == kind]
            for i in range(0, len(rr), 30):
                parts.append((f"{'First' if test == 'first' else 'Second'} test, {title}" + (" (continued)" if i else ""), rr[i:i + 30]))
    o.append(r"{\small\renewcommand{\arraystretch}{1.2}\begin{longtable}{|l|r|r|r|r|l|}\hline" + "\n")
    head = r"Set & $n$ & Base & Control & AyaDial & \makecell[l]{AyaDial $-$ Control\\(95\% interval)}\\\hline" + "\n"
    o.append(head + r"\endfirsthead" + "\n" + head + r"\endhead" + "\n")
    for title, rr in parts:
        if title.endswith("(continued)"):
            pass
        else:
            o.append(f"\\multicolumn{{6}}{{|l|}}{{\\textbf{{{title}}}}}\\\\\\hline\n")
        for _, name, n, vals, dd, cc in rr:
            o.append(f"{name} & {n} & {num(vals['Base'])} & {num(vals['Control'])} & {num(vals['AyaDial'])} & {dci(dd, cc)}\\\\\\hline\n")
    o.append(r"\end{longtable}}" + "\n")
    return "".join(o)


WSP_BLOCKS = [("sys_long", "WildJailbreak under a long system prompt"),
              ("mt_sys", "WildJailbreak, system prompt plus two earlier turns"),
              ("mt_benign", "WildJailbreak after two earlier harmless turns"),
              ("sys_law", "WildJailbreak under a law-office system prompt"),
              ("sys_bank", "WildJailbreak under a short (one-line) system prompt")]
WSP_HARMFUL = [("jbb_RS", "JailbreakBench random search"), ("jbb_DSN", "JailbreakBench DSN"), ("itw_h_dev", "In-the-wild templates"),
               ("sorry_base_dev", "SORRY-Bench, base questions"), ("adversarial_harmful", "WildJailbreak"), ("jbb_PAIR", "JailbreakBench PAIR"),
               ("harmful", "Plain harmful requests"), ("jbb_GCG", "JailbreakBench GCG")]
WSP_HARMLESS = [("orbench_hard", "OR-Bench-hard"), ("xstest_safe", "XSTest"), ("pht_harmless_dev", "PHTest"),
                ("adversarial_benign", "WildJailbreak, harmless"), ("itw_b_dev", "In-the-wild templates, harmless"), ("benign", "Plain harmless requests")]
WSP_NAMES = {"A": "AyaDial", "B": "AyaDial-WSP", "C": "AyaDial-WSP (add-on)", "Base": "Base"}


def wsp_data():
    """Every number of the WSP section: rates and intervals from summary.json; three-way shares from the per-request labels of each job."""
    import re
    sm = J("round7/fable_edits/summary.json")
    rows = sm["rows"]
    jobs = sm["meta"]["jobs"]
    beh_cache = {}

    def beh(k):
        if k not in beh_cache:
            with open(B.RES / f"round7/fable_edits/jobs/beh_{k}.jsonl") as f:
                beh_cache[k] = [json.loads(x) for x in f]
        return beh_cache[k]

    def shares(row):
        """Per-arm shares for one summary row from its B and C jobs' label files; checked against summary.json."""
        out = {}
        for edit in ("B", "C"):
            cands = [j for j in jobs.values() if j["edit"] == edit and j["format"] == row["format"]]
            lines = [x for j in cands for x in beh(j["k"]) if x["set"] == row["set"] and x["label"] == row["label"]]
            assert len(lines) == row["n"], (edit, row["format"], row["set"], len(lines), row["n"])
            arms = next(j["arms"] for j in cands if any(x["set"] == row["set"] for x in beh(j["k"])))
            for i, arm in enumerate(arms):
                key = {"current": "A" if edit == "B" else "A_in_C_job", "edited": edit, "original": "Base"}[arm]
                lab = [x["labels"][i] for x in lines]
                n = len(lab)
                ref, uns = sum(l[0] for l in lab), sum(l[1] for l in lab)
                both = sum(1 for l in lab if l[0] and l[1])
                sh = {"refused": 100 * (ref - both) / n, "unsafe": 100 * uns / n, "refusal": 100 * ref / n}
                sh["answered"] = 100 - sh["refused"] - sh["unsafe"]
                assert abs(sh["refused"] + sh["answered"] + sh["unsafe"] - 100) < 1e-9
                r = row["rates"][key]
                assert abs(sh["refusal"] - r["refused"]) <= 0.051 and abs(sh["unsafe"] - r["unsafe"]) <= 0.051, (row["set"], key, sh, r)
                out[key] = sh
        return out

    def find(fmt, st, label):
        return next(r for r in rows if r["format"] == fmt and r["set"] == st and r["label"] == label)

    W = {"blocks": {}, "def_h": [], "def_b": []}
    for fmt, _ in WSP_BLOCKS:
        h, b = find(fmt, "adversarial_harmful", "harmful"), find(fmt, "adversarial_benign", "harmless")
        W["blocks"][fmt] = {"H": dict(h, sh=shares(h), main=h["main_metric"]), "B": dict(b, sh=shares(b), main=b["main_metric"])}
        assert h["main_metric"] == "unsafe" and b["main_metric"] == "refusal"
    for key, name in WSP_HARMFUL:
        cands = [r for r in rows if r["format"] == "default" and r["label"] == "harmful" and (r["set"] == key or r["set"] in (key + "_pool", key + "_dev"))]
        r = max(cands, key=lambda x: x["n"])
        W["def_h"].append(dict(r, name=name, halves=len(cands), sh=shares(r)))
    for key, name in WSP_HARMLESS:
        r = find("default", key, "harmless")
        W["def_b"].append(dict(r, name=name, sh=shares(r)))
    pooled = {(x["group"], x["part"], x["label"], x["cmp"]): x for x in sm["pooled"]}
    W["pool"] = pooled
    W["n_rows"] = sum(r["n"] for r in rows)
    W["nr_strict"] = sum(r["n_never_read"] for r in rows)
    W["nr_setid"] = sum(r["n_never_read_by_set_id"] for r in rows)
    W["judge"] = sm["meta"]["judge"].split(" ")[0]
    W["max_new"] = sm["meta"]["max_new_tokens"]
    # the run itself (wsp/runs/real-01)
    run = B.BUILD / "wsp/runs/real-01"
    st = json.loads((run / "state.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    W["run"] = {"turns": st["turn"], "usd": st["spent_usd"], "reads": st["reads"], "behaves": st["behaves"], "python": st["python_runs"],
                "reads_max": cfg["reads"], "behaves_max": cfg["behaves"], "cap": cfg["dollar_cap"], "stop": st["final_reason"],
                "model": cfg["model"], "effort": cfg["effort"], "layers": cfg["weights"]["read_layers"]}
    assert cfg["model"] == "claude-fable-5-2" and cfg["effort"] == "max"
    ex = (run / "explanations.md").read_text()
    m = re.search(r"harmful prompts lose ([0-9.]+) of z on average and harmless only ([0-9.]+), the class-mean difference vector shrinks by (\d+)% and rotates \(cosine ([0-9.]+)", ex)
    W["found"] = {"drop_h": float(m.group(1)), "drop_b": float(m.group(2)), "shrink": int(m.group(3)), "cos": float(m.group(4))}
    assert "q·x is not zero at the reply-start token" in ex and "guard leakage" in ex
    m = re.search(r"## t096\n\n(.*?)\n", ex, re.S)
    assert "fitted on all 14 formats" in m.group(1) and "theta 0.60" in m.group(1) and "the original two knob units only" in m.group(1)
    fa = (run / "final_answer_t155.md").read_text()
    m = re.search(r"firing on (\d+) of (\d+) unseen multilingual harmless prompts against (\d+) for the current knob", fa)
    W["switch"] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    assert "refusal on no harmless group is more than 2 points above" in (run / "first_message.md").read_text()
    fm = (run / "first_message.md").read_text()
    assert "No training: no gradient descent" in fm and "no internet" in fm
    return W


def wsp_section(S):
    W = S["wsp"]
    rn = W["run"]
    o = [r"\section{Weight-space programming}\label{sec:wsp}" + "\n"]
    o.append("Weight-space programming (WSP) means letting a language model read and write another network's weights directly. "
             f"A frontier LLM, Claude Fable 5.2 at maximum reasoning effort, was given AyaDial's construction and scores as facts and asked to improve it, with no instruction on what to fix. "
             f"It worked blind: it saw only the harness's messages, numbers and request ids, never the text of a request or a reply; it ran its own Python code on the weights of layers "
             f"{rn['layers'][0]} to {rn['layers'][-1]} in a sandbox with no network, and could ask for forward passes on a GPU ({rn['reads']} of {rn['reads_max']} read passes and "
             f"{rn['behaves']} of {rn['behaves_max']} generate-and-judge passes used). "
             f"There was no gradient training, only closed-form linear algebra; the run took {rn['turns']} turns, {rn['python']} Python runs and \\${rn['usd']:.2f}, and ended at the "
             f"\\${rn['cap']:.0f} cap. The setup follows \\emph{{Sparks of Weight-Space Programming}} (Takehi, Teoh, Zhang and Khattab, 2026, a blog post, "
             r"\url{https://rikiyat.github.io/blog/weight-space-programming/})." + "\n\n")
    f = W["found"]
    o.append(f"What it found: under a long system prompt the harm score of harmful requests drops (by {f['drop_h']:g} on average, against {f['drop_b']:g} for harmless ones) "
             f"and its direction rotates (cosine {f['cos']:g} with the default one), and the position guard leaks at the reply-start token, so a threshold change is not enough.\n\n")
    pB = W["pool"][("deployment", "system-prompt formats", "harmful", "B_minus_C")]["all"]
    pE = W["pool"][("default_en", "all rows", "harmful", "B_minus_C")]["all"]
    pR = W["pool"][("deployment", "all rows", "harmless", "B_minus_C")]["all"]
    sw = W["switch"]
    o.append("Two edits are compared with AyaDial on the same requests in the same batch, all at $s = 1$. "
             "\\textbf{AyaDial-WSP} keeps AyaDial's two units and replaces their harm direction with one the editor recomputed from activations in 14 prompt formats (threshold 0.60). "
             "\\textbf{AyaDial-WSP (add-on)}, the editor's final edit, leaves AyaDial's two units untouched and adds a second pair of units that opens only under system prompts and in multi-turn chats. "
             f"The editor moved from the first design to the second because of our rule that no harmless group may be refused more than 2 points above the starting adapter "
             f"(an intermediate candidate fired on {sw[0]} of {sw[1]} unread multilingual harmless prompts against {sw[2]} for AyaDial); "
             f"our own comparison afterwards favours the first on unsafe replies (AyaDial-WSP minus add-on: {dci(pB['diff'], pB['ci95'])} under system prompts, "
             f"{dci(pE['diff'], pE['ci95'])} without one), at {dci(pR['diff'], pR['ci95'])} points of harmless refusal pooled over all formats.\n\n")
    o.append(f"Labels: {W['judge']} (model judge), up to {W['max_new']} new tokens. $\\downarrow$ lower is better, $\\uparrow$ higher is better. "
             "Differences are against AyaDial, with 95\\% intervals over requests; a difference is established when its interval excludes 0.\n\n")
    for fmt, title in WSP_BLOCKS:
        o.append(wsp_block(W["blocks"][fmt], title))
    o.append(wsp_compact(W["def_h"], "unsafe", "No system prompt: harmful benchmarks, Unsafe \\% $\\downarrow$",
                         "Default chat format. For JailbreakBench, where two halves of a set were run, the larger half is shown"))
    o.append(wsp_compact(W["def_b"], "refusal", "No system prompt: harmless benchmarks, Refused \\% $\\downarrow$", "Default chat format"))
    pl = W["pool"]
    lh = pl[("languages", "MultiJail without English", "harmful", "B_minus_A")]["all"]
    lb = pl[("languages", "all rows", "harmless", "B_minus_A")]["all"]
    fh = pl[("deployment", "format pool only (10 formats)", "harmful", "B_minus_A")]["all"]
    fbb = pl[("deployment", "format pool only (10 formats)", "harmless", "B_minus_A")]["all"]
    word = lambda c: "established" if (c[0] > 0 or c[1] < 0) else "not shown to differ"
    o.append(f"\\textbf{{Languages.}} MultiJail in nine languages, unsafe: {dci(lh['diff'], lh['ci95'])} ({word(lh['ci95'])}); harmless Aya prompts, refused: "
             f"{dci(lb['diff'], lb['ci95'])} ({word(lb['ci95'])}). "
             f"\\textbf{{Plain requests under the ten system-prompt and multi-turn formats.}} Unsafe: {dci(fh['diff'], fh['ci95'])} ({word(fh['ci95'])}); "
             f"harmless refused: {dci(fbb['diff'], fbb['ci95'])} ({word(fbb['ci95'])}). All for AyaDial-WSP against AyaDial.\n\n")
    gcg = [r for r in W["def_h"] if r["name"] == "JailbreakBench GCG"][0]
    bank = W["blocks"]["sys_bank"]["B"]
    o.append(r"\textbf{Limits.} " + "These requests come from the same pool the editor worked with. "
             f"Of the {W['n_rows']} (format, request) rows, {W['nr_setid']} were never read by the editor under their set id and {W['nr_strict']} were never read as the underlying request in any form; "
             "they are not the set-aside tests of the earlier sections, where AyaDial-WSP has not been run yet. "
             f"AyaDial-WSP has more unsafe replies on GCG suffixes ({num(gcg['rates']['A']['unsafe'])}\\% to {num(gcg['rates']['B']['unsafe'])}\\%, {dci(gcg['B_minus_A']['main']['diff'], gcg['B_minus_A']['main']['ci95'])}, an interval that touches 0) "
             f"and raises harmless refusals under the short system prompt ({num(bank['rates']['A']['refused'])}\\% to {num(bank['rates']['B']['refused'])}\\%, {dci(bank['B_minus_A']['main']['diff'], bank['B_minus_A']['main']['ci95'])}). "
             "Every label comes from one judge, and the editor was run once.\n")
    return "".join(o)


def wsp_rows():
    return [("Base", "Base"), ("A", "AyaDial"), ("B", "AyaDial-WSP"), ("C", "AyaDial-WSP (add-on)")]


def wsp_block(blk, title):
    H, Bh = blk["H"], blk["B"]
    o = [f"\\needspace{{18\\baselineskip}}\\medskip{{\\raggedright\\noindent\\large\\bfseries Benchmark: {title}\\par}}\\vspace{{-3pt}}\\noindent "
         f"{H['n']} harmful and {Bh['n']} harmless requests" + (", none of them read by the editor" if H["n_never_read"] == H["n"] and Bh["n_never_read"] == Bh["n"] else "") + ".\n\n"]
    for side, R in (("H", H), ("B", Bh)):
        sh = R["sh"]
        diffs = {"B": R["B_minus_A"]["main"], "C": R["C_minus_A"]["main"]}
        for e in ("B", "C"):  # printed differences equal summary.json
            assert diffs[e]["n"] == R["n"]
        live = [("A", "AyaDial"), ("B", "AyaDial-WSP"), ("C", "AyaDial-WSP (add-on)")]
        if side == "H":
            bold = {}
            for key, higher in (("refused", True), ("unsafe", False)):
                c = [(m, lab, sh[m][key]) for m, lab in live]
                for (m, _, _), b in zip(c, mark("WSP " + title, key, c, higher=higher)):
                    bold[(m, key)] = b
            o.append(r"\begin{table}[H]\centering\small\begin{tabular}{@{}lrrrl@{}}\toprule" + "\n")
            o.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{Harmful requests ($n = {R['n']}$)}}}}\\\\\n")
            o.append(r"Method & Refused (\%) $\uparrow$ & \makecell[r]{Answered without\\harm (\%)} & Unsafe (\%) $\downarrow$ & \makecell[l]{Unsafe, difference from\\AyaDial (95\% interval)}\\\midrule" + "\n")
            for m, lab in wsp_rows():
                x = sh[m]
                o.append(f"{lab} & {numb(x['refused'], bold.get((m, 'refused'), False))} & {numb(x['answered'], False)} & {numb(x['unsafe'], bold.get((m, 'unsafe'), False))} & "
                         + (dci(diffs[m]["diff"], diffs[m]["ci95"]) if m in diffs else "") + "\\\\\n")
        else:
            c = [(m, lab, sh[m]["refusal"]) for m, lab in live]
            bref = dict(zip([x[0] for x in c], mark("WSP " + title + ", harmless", "refused", c)))
            bans = dict(zip([x[0] for x in c], mark("WSP " + title + ", harmless", "answered", [(m, lab, 100 - v) for m, lab, v in c], higher=True)))
            o.append(r"\begin{table}[H]\centering\small\begin{tabular}{@{}lrrl@{}}\toprule" + "\n")
            o.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textbf{{Harmless requests ($n = {R['n']}$)}}}}\\\\\n")
            o.append(r"Method & Refused (\%) $\downarrow$ & Answered (\%) $\uparrow$ & \makecell[l]{Refused, difference from\\AyaDial (95\% interval)}\\\midrule" + "\n")
            for m, lab in wsp_rows():
                v = sh[m]["refusal"]
                o.append(f"{lab} & {numb(v, bref.get(m, False))} & {numb(100 - v, bans.get(m, False))} & "
                         + (dci(diffs[m]["diff"], diffs[m]["ci95"]) if m in diffs else "") + "\\\\\n")
        o.append(r"\bottomrule\end{tabular}\end{table}" + "\n")
    word = lambda c: "established" if (c[0] > 0 or c[1] < 0) else "not shown to differ"
    hb, hc = H["B_minus_A"]["main"], H["C_minus_A"]["main"]
    bb, bc = Bh["B_minus_A"]["main"], Bh["C_minus_A"]["main"]
    o.append(f"Unsafe against AyaDial: AyaDial-WSP {dci(hb['diff'], hb['ci95'])}, {word(hb['ci95'])}; add-on {dci(hc['diff'], hc['ci95'])}, {word(hc['ci95'])}. "
             f"Harmless refused: AyaDial-WSP {dci(bb['diff'], bb['ci95'])}, add-on {dci(bc['diff'], bc['ci95'])}.\n\n")
    return "".join(o)


def wsp_compact(rows, metric, title, note):
    key = "unsafe" if metric == "unsafe" else "refusal"
    o = [r"\begin{table}[H]\centering\small\begin{tabular}{@{}lrrrrrl@{}}\toprule" + "\n"]
    o.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{{title}}}}}\\\\\n")
    o.append(r"Benchmark & $n$ & Base & AyaDial & \makecell[r]{AyaDial-\\WSP} & \makecell[r]{AyaDial-WSP\\(add-on)} & \makecell[l]{AyaDial-WSP $-$ AyaDial\\(95\% interval)}\\\midrule" + "\n")
    for r in rows:
        sh = r["sh"]
        c = [(m, WSP_NAMES[m], sh[m][key]) for m in ("A", "B", "C")]
        b = dict(zip(["A", "B", "C"], mark("WSP default", r["name"], c)))
        d = r["B_minus_A"]["main"]
        o.append(f"{r['name']} & {r['n']} & {num(sh['Base'][key])} & " + " & ".join(numb(sh[m][key], b[m]) for m in ("A", "B", "C"))
                 + f" & {dci(d['diff'], d['ci95'])}\\\\\n")
    o.append(r"\bottomrule\end{tabular}" + "\n" + f"\\caption{{{note}.}}\\end{{table}}\n")
    return "".join(o)


def calib_text(side):
    return ", ".join(f"{n} from {' and '.join(names) if len(names) < 3 else ', '.join(names[:-1]) + ' and ' + names[-1]}" for n, names in side)


def sources_text(S):
    a, wj = S["added"], S["wj"]
    v1src = ", ".join(S["v1_other_src"][:-1]) + " and " + S["v1_other_src"][-1]
    return (f"WildJailbreak training requests ({wj[0]} harmful, {wj[1]} harmless), {S['v1_other']} requests from {v1src}, "
            f"{a['JailbreakBench GCG and PAIR artifacts']} JailbreakBench GCG and PAIR artifacts, {a['GCG suffix transplants']} GCG suffix transplants, "
            f"and {'half of a separate pool of ' if S['half'] else ''}in-the-wild templates ({a['in-the-wild templates']}) and of MultiJail and Aya requests in {S['fit_langs']} languages "
            f"({a['MultiJail'] + a['Aya prompts']})")


def bench_line(S, name, detail):
    """The benchmark named outside the table, with the judge and the arrow legend."""
    return (f"\\medskip{{\\raggedright\\noindent\\large\\bfseries Benchmark: {name}\\par}}\\vspace{{-3pt}}\\noindent "
            f"{detail} Labels: {S['judge']} (model judge), for both the refusal and the unsafe label. "
            f"$\\downarrow$ lower is better, $\\uparrow$ higher is better.\n\n")


def main_tables(S):
    """Harmful (three outcomes) and harmless (two outcomes) tables on the WildJailbreak requests, same rows in both."""
    d, c = S["dev"], S["cast"]
    rows = [("Base", "", "Base", d["Base"], False)]
    rows.append(("Control", "scale 0.1", "Control", d["Control"], True))
    for v, dd in S["dial_full"].items():
        rows.append(("AyaDial", f"$s = {v:g}$", "AyaDial", dd, True))
    for name, sub, dd in S["st_full"]:
        rows.append(("SafeTune " + name, sub, name, dd, True))
    cast_mid = {"H": {"refused_only": c["hr"][1], "unsafe": c["unsafe"][1], "answered": 100 - c["hr"][1] - c["unsafe"][1], "refusal": c["hr"][1]},
                "B": {"refusal": c["ref"][1]}}
    cb = c["base"]
    cast_base = {"H": {"refused_only": cb["harmful_refusal"], "unsafe": cb["harmful_unsafe"], "answered": 100 - cb["harmful_refusal"] - cb["harmful_unsafe"],
                       "refusal": cb["harmful_refusal"]}, "B": {"refusal": cb["benign_refusal"]}}
    rows.append(("SafeTune CAST$^{a}$", f"$\\alpha$ = {', '.join(c['alphas'])}", "CAST", cast_mid, True))
    rows.append(("Base, SafeTune CAST run$^{a}$", "batches of 8", "Base", cast_base, False))
    for r in rows:  # parts sum to 100; Refused + overlap is the summary refusal (checked against the summary tables when loaded)
        h = r[3]["H"]
        assert abs(h["refused_only"] + h["answered"] + h["unsafe"] - 100) < 1e-9, r[0]
        assert abs(round(h["refused_only"], 1) + round(h["answered"], 1) + round(h["unsafe"], 1) - 100) <= 0.15, r[0]
        if "both" in h:
            assert abs(h["refused_only"] + h["both"] - h["refusal"]) < 1e-9
    live = [r for r in rows if r[4]]
    lab = lambda r: f"{r[0]} {r[1]}".strip()
    bH_ref = mark("Main, harmful", "Refused (up)", [(r[2], lab(r), r[3]["H"]["refused_only"]) for r in live], higher=True)
    bH_uns = mark("Main, harmful", "Unsafe (down)", [(r[2], lab(r), r[3]["H"]["unsafe"]) for r in live])
    bB_ref = mark("Main, harmless", "Refused (down)", [(r[2], lab(r), r[3]["B"]["refusal"]) for r in live])
    bB_ans = mark("Main, harmless", "Answered (up)", [(r[2], lab(r), 100 - r[3]["B"]["refusal"]) for r in live], higher=True)
    idx = {id(r): i for i, r in enumerate(live)}

    def b(arr, r):
        return arr[idx[id(r)]] if id(r) in idx else False

    o = [r"\begin{table}[H]\centering\small\begin{tabular}{@{}llrrr@{}}\toprule" + "\n"]
    o.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textbf{{Harmful requests ($n = {S['nH']}$)}}}}\\\\\n")
    o.append(r"Method & Setting & Refused (\%) $\uparrow$ & \makecell[r]{Answered without\\harm (\%)} & Unsafe (\%) $\downarrow$\\\midrule" + "\n")
    for r in rows:
        h = r[3]["H"]
        o.append(f"{r[0]} & {r[1]} & {numb(h['refused_only'], b(bH_ref, r))} & {numb(h['answered'], False)} & {numb(h['unsafe'], b(bH_uns, r))}\\\\\n")
    o.append(r"\bottomrule\end{tabular}" + "\n")
    o.append(r"\caption{Harmful requests at the settings run ($s = 1$ is AyaDial's setting in the later tables); the baselines were not tuned to equal cost. "
             r"Unsafe is the unsafe label, Refused is the refusal label without the unsafe label, and Answered without harm is the rest, so each row sums to 100.}\label{tab:mainh}\end{table}" + "\n")
    o.append(r"\begin{table}[H]\centering\small\begin{tabular}{@{}llrr@{}}\toprule" + "\n")
    o.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textbf{{Harmless requests ($n = {S['nB']}$)}}}}\\\\\n")
    o.append(r"Method & Setting & Refused (\%) $\downarrow$ & Answered (\%) $\uparrow$\\\midrule" + "\n")
    for r in rows:
        x = r[3]["B"]["refusal"]
        o.append(f"{r[0]} & {r[1]} & {numb(x, b(bB_ref, r))} & {numb(100 - x, b(bB_ans, r))}\\\\\n")
    o.append(r"\bottomrule\end{tabular}" + "\n")
    o.append(f"\\caption{{Harmless requests, same methods and settings; Refused is the refusal label and Answered is the rest. "
             f"$^{{a}}$SafeTune CAST's three $\\alpha$ values gave one result ({rng(c['unsafe'], 'unsafe')}, {rng(c['ref'], 'refused')}; the middle one is shown) and it ran with its own Base; "
             f"its per-request labels are not on this machine, so its Answered without harm is 100 minus refused minus unsafe and its Refused includes any reply labelled both.}}"
             f"\\label{{tab:mainb}}\\end{{table}}\n")
    return "".join(o)


def adapters_page(S):
    L, a = S["layer"], S["added"]
    o = [r"\section{The adapters}" + "\n" + r"\setlength{\abovedisplayskip}{4pt}\setlength{\belowdisplayskip}{4pt}" + "\n"]
    o.append(f"All the adapters edit the same place: MLP units of layer {L} that are dormant in the base model, 2 of {S['inter']} per adapter. "
             f"At the token where the reply starts, a unit pair reads a harm score $z = w \\cdot x$ from the layer's input $x$ (about 0 on typical harmless and 1 on typical harmful requests); "
             f"when $s\\,z$ passes a threshold $\\theta$ it adds a push along the refusal direction $r$, capped at one full push. "
             f"At every other token a guard (a second vector that detects any position other than the reply start) keeps the units shut, and with $s = 0$ nothing is written.\n")
    o.append(r"""\begin{align*}
p(z) &= \min\bigl(1,\ \max\bigl(0,\ \beta\,(s\,z-\theta)\bigr)\bigr)\, r && \text{(Tighten: toward refusal)}\\
q(z) &= -\,t\,\min\bigl(1,\ \max\bigl(0,\ \beta_L\,(\theta_L-z)\bigr)\bigr)\, r && \text{(Loosen: away from refusal)}
\end{align*}
""")
    o.append(f"with $\\beta = {S['beta']:g}$ and $\\theta = {S['th3']:.3f}$ for AyaDial; "
             f"$\\theta_L = {S['thL']:g}$, $\\beta_L = {S['betaL']:g}$ for Loosen. "
             f"Each unit $j$ adds $\\mathrm{{silu}}(g_j \\cdot x)\\,(u_j \\cdot x)\\,d_j$; for the Tighten pair $A$, $B$:\n")
    o.append(r"\[ g_A = -\alpha\theta\,m - \kappa\,q + s\,\alpha\,w,\quad g_B = -\alpha\bigl(\theta + \tfrac{1}{\beta}\bigr)m - \kappa\,q + s\,\alpha\,w,\quad u_A = u_B = m,\quad d_A = -d_B = \tfrac{\beta}{\alpha}\,r \]" + "\n")
    o.append(f"where $m$ reads 1 at the reply-start token and 0 elsewhere, $q$ (the guard) reads 0 there and 1 elsewhere, $\\alpha = {S['alpha']:g}$ and $\\kappa = {S['kappa']:g}$. "
             f"The Loosen pair has the same form with $w$ negated, $\\alpha_L = {S['alphaL']:g}$, and $t$ scaling the down columns.\n\n")
    mods = ", ".join(f"\\texttt{{{m.replace('_', '\\_')}}}" for m in S["install_mods"])
    o.append(f"\\textbf{{Why LoRA.}} The change is a few rows of {mods} and columns of \\texttt{{down\\_proj}} in layer {L}, a low-rank update, so each adapter ships as two LoRA files: "
             f"an install adapter (rank {S['install_rank']} on {mods}) applied at scale 1, and a knob adapter (rank {S['knob_rank']} on \\texttt{{{S['knob_mod'].replace('_', '\\_')}}} only) "
             f"whose scale is $s$, set per request (for example the per-request \\texttt{{lora}} field of llama-server). "
             f"There is no gradient training: a shrunk LDA gives the harm direction, and ridge regressions on cached activations give $w$ and the two position vectors $m$ and $q$. "
             f"AyaDial and Loosen were exported as GGUF install and knob files for llama.cpp.\n")
    st = S["set"]
    rows = [
        ("Control", f"Adds $r$ at every position with no gate, generated tokens included; a hook at layer {L}'s output, not an adapter.",
         "No harm score.", "none", f"\\makecell[l]{{scale ${st['Control']:g}$\\\\(${st['ControlL']:g}$ for Loosen)}}"),
        ("AyaDial", "Pushes toward refusal when $s\\,z$ passes $\\theta$.", sources_text(S)[0].upper() + sources_text(S)[1:] + ".",
         f"{S['units_t'][0]}, {S['units_t'][1]}", f"$s = {st['AyaDial']:g}$"),
        ("Loosen", "Pushes away from refusal when $z$ is below $\\theta_L$, to cut false refusals; kept private.",
         f"AyaDial's harm score; $\\theta_L$ is the {S['pctL']:g}st percentile of the harmful requests that score was computed from.",
         f"{S['units_l'][0]}, {S['units_l'][1]}", f"$t = {st['Loosen']:g}$"),
        ("\\makecell[l]{AyaDial\\\\(Tighten + Loosen)}", "Both pairs installed together.", "As the two rows above.",
         "all four", f"$s = {st['TL'][0]:g}$, $t = {st['TL'][1]:g}$"),
    ]
    o.append(r"\begin{table}[H]\centering\footnotesize\renewcommand{\arraystretch}{1.12}"
             r"\begin{tabularx}{\linewidth}{|l|>{\raggedright\arraybackslash\hsize=0.85\hsize}X|>{\raggedright\arraybackslash\hsize=1.15\hsize}X|l|l|}\hline" + "\n"
             r"Model & What it does & Computed from & Units & Setting used\\\hline" + "\n")
    o.append("\\hline\n".join(f"{a_} & {b} & {c} & {d} & {e}\\\\\n" for a_, b, c, d, e in rows))
    o.append(r"\hline\end{tabularx}\caption{The models compared in this report. Units are MLP units of layer """ + f"{L}" + r""".}\end{table}""" + "\n")
    klh = S["kl_hook"]
    props = [("Trained parameters", "none (closed form)"),
             ("Acts at", "reply-start token, by design (not measured for the shipped adapter)"),
             ("At $s = 0$", "writes nothing, by construction ($s$ multiplies the harm term inside the gate)"),
             ("Weights against the hook", f"next-token KL {num(klh[0], 4)} to {num(klh[1], 4)} ({klh[2]} prompts)")]
    o.append(r"\noindent\begin{minipage}[b]{0.47\linewidth}\centering\footnotesize"
             r"\begin{tabularx}{\linewidth}{|l|>{\raggedright\arraybackslash}X|}\hline Property & Value\\\hline" + "\n")
    o.append("\\hline\n".join(f"{x} & {y}\\\\\n" for x, y in props))
    o.append(r"\hline\end{tabularx}\captionof{table}{Properties of the adapters.}\end{minipage}\hfill" + "\n")
    o.append(r"\begin{minipage}[b]{0.5\linewidth}\centering" + zones(S) + r"\end{minipage}" + "\n")
    return "".join(o)


def zones(S):
    """Push against z at s = 1 (AyaDial) and t as used (Loosen): the three zones."""
    th, b, thL, bL, t = S["th3"], S["beta"], S["thL"], S["betaL"], S["tL"]
    return (r"""\begin{tikzpicture}
\begin{axis}[width=\linewidth, height=3.5cm, xmin=-0.25, xmax=1.25, ymin=""" + f"{-t - 0.4:g}" + r""", ymax=1.5, axis lines=left,
  xlabel={Harm score $z$}, ylabel={Push along $r$}, xtick={0,0.5,1}, ytick={""" + f"{-t:g},0,1" + r"""}, clip=false]
\addplot[color=cLoose, domain=-0.25:""" + f"{thL:g}" + r""", samples=201] {""" + f"-{t:g}*min(1,max(0,{bL:g}*({thL:g}-x)))" + r"""};
\addplot[color=black!45, domain=""" + f"{thL:g}:{th:.4f}" + r""", samples=2] {0};
\addplot[color=cDial, domain=""" + f"{th:.4f}" + r""":1.25, samples=201] {""" + f"min(1,max(0,{b:g}*(x-{th:.4f})))" + r"""};
\draw[black!40, dashed] (axis cs:""" + f"{thL:g}" + r""",""" + f"{-t - 0.4:g}" + r""") -- (axis cs:""" + f"{thL:g}" + r""",1);
\draw[black!40, dashed] (axis cs:""" + f"{th:.4f}" + r""",""" + f"{-t - 0.4:g}" + r""") -- (axis cs:""" + f"{th:.4f}" + r""",1);
\node[font=\footnotesize, anchor=north west, text=cLoose] at (axis cs:-0.24,""" + f"{-0.2:g}" + r""") {Loosen};
\node[font=\footnotesize, anchor=south, text=black!60] at (axis cs:""" + f"{(thL + th) / 2:.3f}" + r""",1.05) {nothing written};
\node[font=\footnotesize, anchor=south east, text=cDial] at (axis cs:1.24,1.05) {Tighten};
\end{axis}\end{tikzpicture}
\captionof{figure}{""" + f"Push against harm score: Tighten at $s = 1$, Loosen at $t = {t:g}$ (AyaDial's harm score). Between $\\theta_L$ and $\\theta$ neither pair writes." + r"""}
""")


def harm_table(S, test, H, names, Bh, caption, label):
    hdr = {"Base": "Base", "Control": "Control", "AyaDial": "AyaDial"}
    cols = "@{}lr" + "r" * len(names) + "l@{}"
    o = [f"\\begin{{table}}[H]\\centering\\small\\begin{{tabular}}{{{cols}}}\\toprule\n"]
    o.append("Set & $n$ & " + " & ".join(hdr[n] for n in names) + " & \\makecell[l]{AyaDial $-$ Control\\\\(95\\% interval)}\\\\\\midrule\n")
    o.append(f"\\multicolumn{{{len(names) + 3}}}{{@{{}}l}}{{\\textit{{Harmful requests: Unsafe \\% $\\downarrow$}}}}\\\\\n")
    for r in H:
        b = contest(S, test, label, r, names)
        o.append(f"{r['label']} & {r['n']} & " + " & ".join(numb(r["vals"][n], b.get(n, False)) for n in names) + f" & {dci(r['d'], r['ci'])}\\\\\n")
    o.append("\\midrule\n" + f"\\multicolumn{{{len(names) + 3}}}{{@{{}}l}}{{\\textit{{Harmless requests: Refused \\% $\\downarrow$}}}}\\\\\n")
    for r in Bh:
        b = contest(S, test, label, r, names)
        o.append(f"{r['label']} & {r['n']} & " + " & ".join(numb(r["vals"][n], b.get(n, False)) for n in names) + f" & {dci(r['d'], r['ci'])}\\\\\n")
    o.append(f"\\bottomrule\\end{{tabular}}\\caption{{{caption}}}\\label{{{label}}}\\end{{table}}\n")
    return "".join(o)


def st_table(S, test, H, Bh, label, main_label):
    """The SafeTune baselines on the same rows; "pending" until results/round7/safetune_tests.json has the cell."""
    sets = S["st_tests"].get("settings", {})
    hdr = " & ".join(f"\\makecell[r]{{SafeTune\\\\{m}{(' (' + sets[m] + ')') if m in sets else ''}}}" for m in ST_METHODS)
    o = ["\\begin{table}[H]\\centering\\small\\begin{tabular}{@{}lrrrr@{}}\\toprule\n"]
    o.append("Set & $n$ & " + hdr + "\\\\\\midrule\n")
    for title, rows in (("Harmful requests: Unsafe \\% $\\downarrow$", H), ("Harmless requests: Refused \\% $\\downarrow$", Bh)):
        o.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\\n")
        for r in rows:
            b = contest(S, test, main_label, r, ["Control", "AyaDial"], record=False)
            vals = [st_value(S, test, m, r) for m in ST_METHODS]
            o.append(f"{r['label']} & {r['n']} & " + " & ".join("\\textit{pending}" if v is None else numb(v, b.get(m, False)) for m, v in zip(ST_METHODS, vals)) + "\\\\\n")
        if rows is H:
            o.append("\\midrule\n")
    pend = not all(st_value(S, test, m, r) is not None for m in ST_METHODS for r in H + Bh)
    o.append(f"\\bottomrule\\end{{tabular}}\\caption{{SafeTune baselines on the same rows as Table~\\ref{{{main_label}}}"
             + (", to be calibrated on the same sources as AyaDial; pending cells have not been run yet." if pend else ", calibrated on the same sources as AyaDial.")
             + f"}}\\label{{{label}}}\\end{{table}}\n")
    return "".join(o)


TABNAME = {"tab:t1": "Known sources", "tab:t2": "Never seen", "tab:t3": "Formats", "tab:loosen": "Loosen"}
SHOWN = {"TL": "Tighten + Loosen"}


def contest(S, test, label, r, names, record=True):
    """One contest (one row of a results table): Control, AyaDial and any SafeTune cell that has a number. Base is not a contestant."""
    cands = [(n, SHOWN.get(n, n), r["vals"][n]) for n in names if n != "Base"]
    if test:
        cands += [(m, "SafeTune " + m, v) for m in ST_METHODS for v in [st_value(S, test, m, r)] if v is not None]
    if not record:
        n0 = len(CONTESTS)
    b = mark(TABNAME[label], r["label"], cands)
    if not record:
        del CONTESTS[n0:]
    return {c[0]: x for c, x in zip(cands, b)}


def cells(label, r, names, metric=""):
    """Rate cells of one data-set row (Loosen table); Base is shown plain and is not a contestant."""
    contest_ = [n for n in names if n != "Base"]
    b = mark(TABNAME[label], r["label"] + metric, [(n, SHOWN.get(n, n), r["vals"][n]) for n in contest_])
    bd = dict(zip(contest_, b))
    return " & ".join(numb(r["vals"][n], bd.get(n, False)) for n in names)


def lax_table(S):
    names = ["Base", "Control", "Loosen", "TL"]
    hdr = ["Base", "Control", "Loosen", "\\makecell[r]{AyaDial\\\\(Tighten\\\\+ Loosen)}"]
    o = ["\\begin{table}[H]\\centering\\small\\setlength{\\tabcolsep}{3.0pt}\\begin{tabular}{@{}lrrrrrll@{}}\\toprule\n"]
    o.append("Set & $n$ & " + " & ".join(hdr) + " & \\makecell[l]{Loosen $-$ Base\\\\(95\\% interval)} & \\makecell[l]{Loosen $-$ Control\\\\(95\\% interval)}\\\\\\midrule\n")

    def sec(title, rows, metric):
        o.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\\n")
        for r in rows:
            o.append(f"{r['label']} & {r['n']} & " + cells("tab:loosen", r, names, metric) + f" & {dci(r['db'], r['dbci'])} & {dci(r['d'], r['ci'])}\\\\\n")

    sec("Harmless requests: Refused \\% $\\downarrow$", S["LB"], " (refused)")
    o.append("\\midrule\n")
    sec("Harmless requests: Unsafe label \\% $\\downarrow$", S["LBu"], " (unsafe label)")
    o.append("\\midrule\n")
    sec("Harmful requests: Unsafe \\% $\\downarrow$", S["LH"], " (unsafe)")
    o.append("\\bottomrule\\end{tabular}\\caption{Loosen on unseen requests from the first test. Read the first block for false refusals and the last block for any release of harmful replies.}\\label{tab:loosen}\\end{table}\n")
    return "".join(o)


# label placement for the zoomed scatter: (method, setting) -> TikZ anchor of the label (the label sits on the other side)
LABPOS = {("AyaDial", 0.75): "east", ("AyaDial", 1): "north east", ("AyaDial", 1.25): "south",
          ("Control", 0.02): "west", ("Control", 0.05): "north west", ("Control", 0.08): "north west", ("Control", 0.1): "north west",
          ("Control", 0.15): "south", ("Control", 0.2): "north",
          ("RefusalDirection", 1.44): "south west", ("RefusalDirection", 2.87): "south west", ("RefusalDirection", 5.74): "south west",
          ("AdaSteer", 1): "south west"}
ZOOM = (9, 30, 0, 8.4)


def scatter(S):
    """Two panels: every setting at full range, and the lower-left region zoomed with each setting labelled."""
    d, c = S["dev"], S["cast"]
    lines = [("Control", [(v, u, r) for v, u, r in S["ctl_pts"]], 0.1, "scale"),
             ("AyaDial", S["dial_pts"], 1, "s")]
    st = {}
    for name, sub, u, r in S["st"]:
        st.setdefault(name, []).append((float(sub.split()[-1]), u, r))
    lines += [("RefusalDirection", st["RefusalDirection"], None, "strength"), ("AdaSteer", st["AdaSteer"], None, "bound")]
    LEG = {"Control": "Control (scale)", "AyaDial": "AyaDial ($s$)",
           "RefusalDirection": "SafeTune RefusalDirection (strength)", "AdaSteer": "SafeTune AdaSteer (bound)"}
    OPEN = {"Control": "diamond", "AyaDial": "triangle", "RefusalDirection": "o", "AdaSteer": "pentagon"}
    bx, by = d["Base"]["B"]["refusal"], d["Base"]["H"]["unsafe"]

    def plots(legend, labels):
        o = [f"\\addplot[only marks, mark=*, mark size=2.4pt, color={CNAME['Base']}] coordinates {{({bx:.3f},{by:.3f})}};"
             + ("\\addlegendentry{Base}" if legend else "") + "\n"]
        for name, pts, tab, _ in lines:
            coords = " ".join(f"({r:.3f},{u:.3f})" for _, u, r in pts)
            o.append(f"\\addplot[color={CNAME[name]}, mark={OPEN[name]}, mark size=1.9pt, thin, mark options={{solid, fill=white}}] coordinates {{{coords}}};"
                     + (f"\\addlegendentry{{{LEG[name]}}}" if legend else "") + "\n")
            if tab is not None:  # the setting used in the tables: filled and larger
                _, u, r = next(x for x in pts if x[0] == tab)
                o.append(f"\\addplot[only marks, color={CNAME[name]}, mark={MARK[name]}, mark size=3.1pt, mark options={{draw=black, line width=0.5pt}}, forget plot] coordinates {{({r:.3f},{u:.3f})}};\n")
            if labels:
                for v, u, r in pts:
                    if ZOOM[0] <= r <= ZOOM[1] and ZOOM[2] <= u <= ZOOM[3]:
                        a = LABPOS.get((name, v), "south west")
                        sep = "3.8pt" if v == tab else "1.6pt"  # clear the larger marker of the table setting
                        o.append(f"\\node[font=\\scriptsize, text={CNAME[name]}, anchor={a}, inner sep={sep}] at (axis cs:{r:.3f},{u:.3f}) {{{v:g}}};\n")
        o.append(f"\\addplot[only marks, mark={MARK['CAST']}, mark size=3.2pt, color={CNAME['CAST']}] coordinates {{({c['ref'][1]:.3f},{c['unsafe'][1]:.3f})}};"
                 + ("\\addlegendentry{SafeTune CAST (own batching)}" if legend else "") + "\n")
        return "".join(o)

    x0, x1, y0, y1 = ZOOM
    o = [r"""\begin{figure}[H]\centering
\begin{tikzpicture}
\begin{axis}[name=top, width=0.6\linewidth, height=5.2cm, xmin=0, xmax=72, ymin=0, ymax=8.6,
  xlabel={Harmless requests refused (\%) $\downarrow$}, ylabel={Harmful unsafe (\%) $\downarrow$}, grid=major, grid style={black!8},
  legend style={at={(1.04,1)}, anchor=north west, font=\footnotesize}, legend cell align=left, title={All settings}, title style={font=\small}]
""", plots(True, False),
         f"\\draw[black!55, dashed] (axis cs:{x0},{y0}) rectangle (axis cs:{x1},{y1});\n",
         r"""\end{axis}
\begin{axis}[at={(top.south west)}, anchor=north west, yshift=-2.1cm, width=\linewidth, height=8.6cm, """
         + f"xmin={x0}, xmax={x1}, ymin={y0}, ymax={y1}," + r"""
  xlabel={Harmless requests refused (\%) $\downarrow$}, ylabel={Harmful requests unsafe (\%) $\downarrow$}, grid=major, grid style={black!8},
  title={Zoom on the dashed box, each point labelled with its setting}, title style={font=\small}, clip=true]
""", plots(False, True),
         r"\node[anchor=south west, font=\footnotesize, text=black!60] at (axis cs:" + f"{x0 + 0.3},{y0 + 0.25}" + r") {better: lower left};" + "\n",
         r"""\end{axis}\end{tikzpicture}
\caption{Unsafe on harmful against refused on harmless, on the WildJailbreak requests of Tables~\ref{tab:mainh} and~\ref{tab:mainb}: each line joins the settings of one method on the same 1910 requests, $s$ is the adapter scale, and filled points with a black outline are the settings in those tables ($s = 1$; Control at 0.1).
SafeTune CAST ran with other batching and its own Base, so it is one point; SafeTune AdaSteer's bounds 2 and 4 lie right of the zoom.}\label{fig:scatter}
\end{figure}
"""]
    return "".join(o)


def dial(S):
    ip = S["dial_pts"]
    bH, bB = S["dev"]["Base"]["H"]["unsafe"], S["dev"]["Base"]["B"]["refusal"]
    ymaxL = max([u for _, u, _ in ip] + [bH]) * 1.18
    ymaxR = max([r for _, _, r in ip] + [bB]) * 1.18
    xs = ",".join(f"{v:g}" for v, _, _ in ip)
    lo, hi = ip[0][0] - 0.15, ip[-1][0] + 0.15
    o = [r"""\begin{figure}[H]\centering
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.9cm}, width=0.47\linewidth, height=6cm,
  """ + f"xmin={lo:g}, xmax={hi:g}, xtick={{{xs}}}," + r""" xlabel={Scale $s$}, ymin=0, grid=major, grid style={black!8}]
"""]
    o.append(f"\\nextgroupplot[ymax={ymaxL:.2f}, ylabel={{Harmful requests unsafe (\\%) $\\downarrow$}}, legend style={{at={{(0.97,0.58)}}, anchor=east}}, legend cell align=left]\n")
    o.append(f"\\addplot[color={CNAME['Base']}, dashed, no marks, domain={lo:g}:{hi:g}] {{{bH:.3f}}};\\addlegendentry{{Base}}\n")
    o.append(f"\\addplot[color={CNAME['AyaDial']}, mark={MARK['AyaDial']}, mark size=2.8pt] coordinates {{" + " ".join(f"({v:g},{u:.3f})" for v, u, _ in ip) + "};\\addlegendentry{AyaDial}\n")
    o.append(f"\\nextgroupplot[ymax={ymaxR:.2f}, ylabel={{Harmless requests refused (\\%) $\\downarrow$}}]\n")
    o.append(f"\\addplot[color={CNAME['Base']}, dashed, no marks, domain={lo:g}:{hi:g}] {{{bB:.3f}}};\n")
    o.append(f"\\addplot[color={CNAME['AyaDial']}, mark={MARK['AyaDial']}, mark size=2.8pt] coordinates {{" + " ".join(f"({v:g},{r:.3f})" for v, _, r in ip) + "};\n")
    a, z = ip[0], ip[-1]
    o.append(r"""\end{groupplot}\end{tikzpicture}
""" + f"\\caption{{AyaDial at three settings of $s$, with Base as the dashed line. Unsafe goes from {num(a[1])}\\% at $s = {a[0]:g}$ to {num(z[1])}\\% at $s = {z[0]:g}$ "
          f"while harmless refusal goes from {num(a[2])}\\% to {num(z[2])}\\%.}}\\label{{fig:dial}}\n\\end{{figure}}\n")
    return "".join(o)


def bars(S, keep=None):
    names = ["Base", "Control", "AyaDial"]
    groups = [(g, [vals[n] for n in names]) for g, vals in S["fig1"] if keep is None or g in keep]
    coords = [f"g{i}" for i in range(len(groups))]
    labels = {"WildJailbreak": "WildJailbreak\\\\(new)", "In-the-wild": "In-the-wild\\\\templates", "GCG": "GCG", "Random search": "Random\\\\search",
              "PAIR": "PAIR", "MultiJail": "MultiJail\\\\(9 languages)"}
    ymax = max(v for _, vals in groups for v in vals) * 1.12
    o = [r"""\begin{figure}[H]\centering
\begin{tikzpicture}
\begin{axis}[ybar=0.6pt, bar width=7pt, width=\linewidth, height=6.4cm, ymin=0, """ + f"ymax={ymax:.1f}" + r""", enlarge x limits=0.09,
  symbolic x coords={""" + ",".join(coords) + r"""}, xtick=data, xticklabels={""" + ",".join("{" + labels[g] + "}" for g, _ in groups) + r"""},
  xticklabel style={align=center}, ylabel={Unsafe replies (\%) $\downarrow$}, ymajorgrids, grid style={black!8},
  legend style={at={(0.5,1.03)}, anchor=south, legend columns=3, /tikz/every even column/.append style={column sep=8pt}}, legend cell align=left]
"""]
    for i, n in enumerate(names):
        pts = " ".join(f"({coords[j]},{vals[i]:.3f})" for j, (_, vals) in enumerate(groups))
        o.append(f"\\addplot[fill={CNAME[n]}, draw=none, area legend] coordinates {{{pts}}};\\addlegendentry{{{n}}}\n")
    o.append(r"""\end{axis}\end{tikzpicture}
\caption{The four harmful benchmarks of this section at a glance. Compare the last bar of each group (AyaDial) with the second (Control).}\label{fig:t1}
\end{figure}
""")
    return "".join(o)


def langs_fig(S, exc):
    names = ["Base", "Control", "AyaDial"]
    L = S["langs"]
    coords = [f"l{i}" for i in range(len(L))]
    ymax = max(v for _, vals, *_ in L for v in vals) * 1.15
    lab = ",".join("{" + (n + "$^{*}$" if e else n) + "}" for n, _, _, _, e, _ in L)
    o = [r"""\begin{figure}[H]\centering
\begin{tikzpicture}
\begin{axis}[ybar=0.6pt, bar width=8pt, width=\linewidth, height=5.6cm, ymin=0, """ + f"ymax={ymax:.1f}" + r""", enlarge x limits=0.09,
  symbolic x coords={""" + ",".join(coords) + r"""}, xtick=data, xticklabels={""" + lab + r"""},
  ylabel={Unsafe replies (\%) $\downarrow$}, ymajorgrids, grid style={black!8},
  legend style={at={(0.5,1.03)}, anchor=south, legend columns=3, /tikz/every even column/.append style={column sep=8pt}}, legend cell align=left]
"""]
    for i, n in enumerate(names):
        pts = " ".join(f"({coords[j]},{vals[i]:.3f})" for j, (_, vals, *_r) in enumerate(L))
        o.append(f"\\addplot[fill={CNAME[n]}, draw=none, area legend] coordinates {{{pts}}};\\addlegendentry{{{n}}}\n")
    n_each = L[0][5]
    assert all(x[5] == n_each for x in L)
    if exc:
        parts = "; ".join(f"{n} {dci(dd, cc)}" for n, _, dd, cc, e, _ in L if e)
        ex = (f"Only {', '.join(exc)}$^{{*}}$ has an interval against Control that excludes 0 ({parts}), so the pooled result is the claim."
              if len(exc) == 1 else f"{', '.join(exc)} have intervals against Control that exclude 0 ({parts}).")
    else:
        ex = "No single language has an interval against Control that excludes 0; the pooled result is the claim."
    o.append(r"""\end{axis}\end{tikzpicture}
""" + f"\\caption{{Unsafe replies per language, {n_each} harmful requests each (second test). {ex}}}\\label{{fig:langs}}\n\\end{{figure}}\n")
    return "".join(o)



def appendix_earlier(S):
    """Appendix: the earlier version (the round 4 adapter), kept as a record. No bold here."""
    d, wj = S["dev"], S["wj"]
    v1src = ", ".join(S["v1_other_src"][:-1]) + " and " + S["v1_other_src"][-1]
    o = [r"\clearpage\appendix" + "\n" + r"\section{AyaDial, earlier version (computed from WildJailbreak and plain requests only)}\label{app:earlier}" + "\n"]
    o.append(f"The earlier version has the same construction and the same units ({S['units_t'][0]}, {S['units_t'][1]}); its harm direction and guard were computed only from "
             f"WildJailbreak training requests ({wj[0]} harmful, {wj[1]} harmless) and {S['v1_other']} plain requests from {v1src}, with $\\theta = {S['th1']:.3f}$. "
             f"It is kept as a record: its two adapter files exist as PEFT safetensors and as GGUF, at $s = 0$ its next-token KL against Base is {num(S['kl_s0'], 4)} ({S['kl_s0_n']} prompts), "
             "and it was not run on the second test.\n\n")
    # main benchmark: its full sweep, three outcomes on harmful and refusal on harmless
    o.append(r"\begin{table}[H]\centering\small\begin{tabular}{@{}lrrrr@{}}\toprule" + "\n")
    o.append(r"Setting & \makecell[r]{Harmful:\\Refused (\%) $\uparrow$} & \makecell[r]{Harmful: Answered\\without harm (\%)} & \makecell[r]{Harmful:\\Unsafe (\%) $\downarrow$} & \makecell[r]{Harmless:\\Refused (\%) $\downarrow$}\\\midrule" + "\n")
    rows = [("Base", d["Base"])] + [(f"$s = {v:g}$", dd) for v, dd in S["sweep_full"].items()]
    for name, dd in rows:
        h = dd["H"]
        assert abs(h["refused_only"] + h["answered"] + h["unsafe"] - 100) < 1e-9
        o.append(f"{name} & {num(h['refused_only'])} & {num(h['answered'])} & {num(h['unsafe'])} & {num(dd['B']['refusal'])}\\\\\n")
    assert close(S["sweep_full"][1]["H"]["unsafe"], d["Earlier"]["H"]["unsafe"])
    o.append(r"\bottomrule\end{tabular}" + "\n" + f"\\caption{{The earlier version's sweep of $s$ on the {S['nH']} harmful and {S['nB']} harmless WildJailbreak requests of Section~\\ref{{sec:main}}; $s = 1$ was its setting.}}\\label{{tab:app1}}\\end{{table}}\n")
    # first test, next to AyaDial
    o.append(r"\begin{table}[H]\centering\small\begin{tabular}{@{}lrrrrr@{}}\toprule" + "\n")
    o.append(r"Set & $n$ & Base & Control & \makecell[r]{Earlier\\version} & AyaDial\\\midrule" + "\n")
    for title, rr in (("Harmful requests: Unsafe \\% $\\downarrow$", S["t1H"]), ("Harmless requests: Refused \\% $\\downarrow$", S["t1B"])):
        o.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\\n")
        for r in rr:
            o.append(f"{r['label']} & {r['n']} & " + " & ".join(num(r["vals"][n]) for n in ("Base", "Control", "Earlier", "AyaDial")) + "\\\\\n")
        if rr is S["t1H"]:
            o.append("\\midrule\n")
    o.append(r"\bottomrule\end{tabular}" + "\n" + r"\caption{The first test (Section~\ref{sec:t1}): the earlier version next to AyaDial on the same unseen requests.}\label{tab:app2}\end{table}" + "\n")
    t = {r["label"]: r["vals"] for r in S["t1H"]}
    moved = [("in-the-wild templates", "In-the-wild templates"), ("GCG", "JailbreakBench GCG"), ("MultiJail", "MultiJail, 9 languages")]
    f0, f1_ = S["fire"]
    o.append("The two versions differ only in the requests their direction and guard were computed from, so the sources added for AyaDial "
             "(JailbreakBench GCG and PAIR artifacts, GCG suffix transplants, in-the-wild templates, MultiJail and Aya requests) are what moved "
             + ", ".join(f"{a} ({num(t[k]['Earlier'])}\\% to {num(t[k]['AyaDial'])}\\% unsafe)" for a, k in moved) + ". "
             f"Among requests Base answers unsafely, the earlier version's harm score passes its threshold on {num(f0)}\\% of the WildJailbreak requests of Section~\\ref{{sec:main}} "
             f"and on {num(f1_)}\\% of in-the-wild templates.\n")
    return "".join(o)


def grid(t):
    """Full grid (a line between rows and between columns) on every plain tabular; requested by the owner."""
    import re
    out, inside = [], False
    for line in t.split("\n"):
        m = re.search(r"\\begin\{tabular\}\{@\{\}([lrc]+)@\{\}\}\\toprule", line)
        if m:
            inside = True
            line = line.replace(m.group(0), "\\renewcommand{\\arraystretch}{1.25}\\begin{tabular}{|" + "|".join(m.group(1)) + "|}\\hline")
            out.append(line)
            continue
        if inside:
            done = "\\end{tabular}" in line
            line = re.sub(r"\\multicolumn\{(\d+)\}\{@\{\}l\}", r"\\multicolumn{\1}{|l|}", line)
            line = line.replace("\\bottomrule", "").replace("\\midrule", "").replace("\\toprule", "").replace("\\addlinespace", "")
            if "\\end{tabular}" in line:
                pre, post = line.split("\\end{tabular}", 1)
                pre = re.sub(r"\\\\(\[[^\]]*\])?\s*$", r"\\\\\\hline", pre) if pre.strip() else pre
                line = pre + "\\end{tabular}" + post
            else:
                line = re.sub(r"\\\\(\[[^\]]*\])?\s*$", r"\\\\\\hline", line)
            if line.strip() or done:
                out.append(line)
            if done:
                inside = False
            continue
        out.append(line)
    return "\n".join(out)


def main():
    S = build()
    OUTDIR.mkdir(exist_ok=True)
    OUT.write_text(grid(tex(S)))
    d = S["dev"]
    print(f"wrote {OUT}")
    print(f"headline checks passed: dev AyaDial {d['AyaDial']['H']['unsafe']:.1f}% unsafe / {d['AyaDial']['B']['refusal']:.1f}% refused; "
          f"six languages {S['six']['base_unsafe']:.1f} -> {S['six']['unsafe']:.1f}; OR-Bench {S['orb']['vals']['Base']:.1f} -> {S['orb']['vals']['Loosen']:.1f}")
    for t, c, out, w in CONTESTS:
        print(f"  {t} | {c} | {out} | {', '.join(w)}")


if __name__ == "__main__":
    main()
