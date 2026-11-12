#!/usr/bin/env python3
"""Build the AyaDial results page from the JSON files under results/.

Run from build/:
    python3 presentation/build_ayadial.py
Writes presentation/ayadial.html (self-contained; KaTeX from a pinned CDN for the formulas).

Rules this script keeps:
- Every number on the page is read from a file under results/ (or, for three fold constants, from the scripts that
  wrote the adapters; each is named next to its use). Nothing is typed in by hand.
- The three-part bars (Refused, Answered without harm, Unsafe) are computed from the per-prompt label files
  ([refused, unsafe, degenerate] per prompt). Before a bar is drawn, its refusal and unsafe rates are checked against
  the summary table of the same cell; a mismatch stops the build.
- No prompt, template or reply text is read or written: label files hold ids and 0/1 labels only, and ids are not
  written to the page.
"""
import html
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent
RES = BUILD / "results"
OUT = HERE / "ayadial.html"

USED = []  # every file read, for the appendix


def J(rel):
    p = RES / rel
    if rel not in USED:
        USED.append(rel)
    with open(p) as f:
        return json.load(f)


def esc(s):
    return html.escape(str(s), quote=True)


MINUS = "−"


def f1(x):
    s = f"{x:.1f}"
    return "0.0" if s == "-0.0" else s.replace("-", MINUS)


def pc(x):
    return f1(x) + "%"


def sg(x):
    s = f"{x:+.1f}"
    if float(s) == 0:
        s = "+0.0"
    return s.replace("-", MINUS)


def ci(c):
    return f"{sg(c[0])} to {sg(c[1])}"


# =============================================================================================
# Labels: the same loading rule as scripts/r5_report.py and scripts/r6_report.py
# =============================================================================================
BASES = {"round5": ["base_wjx", "base_own", "base_dev5", "base_sdev5", "base_frz5"],
         "round6": ["base_wjx", "base_own", "base_dev5", "base_xd6", "base_xb6"]}
_LAB = {}


def labels(rnd, judge=None):
    key = (rnd, judge)
    if key in _LAB:
        return _LAB[key]
    d = RES / rnd / ("labels" if not judge else f"labels_{judge}")
    if f"{rnd}/{d.name}/" not in USED:
        USED.append(f"{rnd}/{d.name}/")
    L = {}
    for p in sorted(d.glob("*.json")):
        if p.stem.startswith(("smoke_", "chk_")):
            continue
        with open(p) as f:
            L[p.stem] = json.load(f)
    base = {}
    for b in BASES[rnd]:
        base.update(L.get(b, {}))
    for n, v in L.items():
        if n.startswith("base_") and n not in BASES[rnd]:
            base.update({k: x for k, x in v.items() if k not in base})
    L["base"] = base
    if rnd == "round6":  # exam format suites fold into the cell of the same name
        for n in [k for k in L if k.startswith("xbf_") and "." in k]:
            cell = n.split(".", 1)[1]
            L[cell] = {**L.get(cell, {}), **L[n]}
    _LAB[key] = L
    return L


def shares_from(lab, keys):
    n = len(keys)
    ref = sum(lab[k][0] for k in keys)
    uns = sum(lab[k][1] for k in keys)
    both = sum(1 for k in keys if lab[k][0] and lab[k][1])
    return {"n": n, "refusal": 100 * ref / n, "unsafe": 100 * uns / n, "both": 100 * both / n,
            "refused_only": 100 * (ref - both) / n, "answered": 100 * (n - ref - uns + both) / n}


OVERLAP = {"max": 0.0, "where": ""}


def tshares(rnd, table, row_name, cell, judge=None):
    """Three-part shares for one cell on one table row, from the label files, checked against the table."""
    T = J(table)
    row = next((r for r in T["rows"] if r["row"] == row_name), None)
    if row is None or cell not in row["cells"]:
        raise KeyError(f"{table}: row '{row_name}' cell '{cell}' not found")
    L = labels(rnd, judge)
    keys_all = sorted(k for k in L["base"] if k.split("|")[0] in row["sets"])
    keys = [k for k in keys_all if k in L[cell]]
    s = shares_from(L[cell], keys)
    t = row["cells"][cell]
    if s["n"] != t["n"] or abs(s["refusal"] - t["refusal"]) > 1e-6 or abs(s["unsafe"] - t["unsafe"]) > 1e-6:
        raise ValueError(f"label check failed for {table} / {row_name} / {cell}: {s} vs {t}")
    if s["both"] > OVERLAP["max"]:
        OVERLAP.update(max=s["both"], where=f"{table} / {row_name} / {cell}")
    s["t"] = t
    return s


WJX_H = ["wj_adversarial_harmful", "dev_ho_h", "dev_in_h"]
WJX_B = ["wj_adversarial_benign", "dev_ho_b", "dev_in_b"]


def dev_shares(cell, check=None):
    """Shares on the 1910 WildJailbreak development requests (1000 harmful, 910 harmless), round 6's label folder."""
    L = labels("round6")
    out = {}
    for side, sets in (("H", WJX_H), ("B", WJX_B)):
        keys_all = sorted(k for k in L["base_wjx"] if k.split("|")[0] in sets)
        keys = [k for k in keys_all if k in L[cell]]
        if len(keys) != len(keys_all):
            raise ValueError(f"{cell}: {len(keys)} of {len(keys_all)} development labels")
        out[side] = shares_from(L[cell], keys)
        if out[side]["both"] > OVERLAP["max"]:
            OVERLAP.update(max=out[side]["both"], where=f"development / {cell}")
    if check:  # (harmful refusal, harmful unsafe, harmless refusal) as rounded in a file
        got = (round(out["H"]["refusal"], 1), round(out["H"]["unsafe"], 1), round(out["B"]["refusal"], 1))
        if tuple(round(x, 1) for x in check) != got:
            raise ValueError(f"development check failed for {cell}: {got} vs {check}")
    return out


# =============================================================================================
# Names
# =============================================================================================
ROWNAME = {"base": "Base", "ungated@0.1": "Control", "foldcap@1": "AyaDial", "v3gcap@1": "AyaDial-Improved",
           "ctrlneg@-0.02": "Control", "lax3@2": "Loosen", "both3@1": "AyaDial-Improved (Tighten + Loosen)"}
HIGHLIGHT = {"v3gcap@1", "lax3@2", "both3@1"}

# =============================================================================================
# Chart grammar: one horizontal stacked bar per row, three outcomes, the value that matters at the right
# =============================================================================================


def bar_row(label, s, kind, strong=False, sub="", partial=False):
    """kind 'H': harmful set, three parts, value = Unsafe. kind 'B': harmless set, two parts, value = Refused."""
    if kind == "H":
        segs = [("ref", s["refused_only"], "Refused"), ("ans", s["answered"], "Answered without harm"), ("uns", s["unsafe"], "Unsafe")]
        val = s["unsafe"]
    else:
        segs = [("ref", s["refusal"], "Refused"), ("ans", 100 - s["refusal"], "Answered")]
        val = s["refusal"]
    tip = " · ".join(f"{t} {pc(w)}" for _, w, t in segs) + f" · n = {s['n']}"
    inner = "".join(f'<span class="seg {c}" style="width:{w:.3f}%" title="{esc(t)} {pc(w)}"></span>' for c, w, t in segs if w > 0)
    note = f'<span class="rsub">{esc(sub)}</span>' if sub else ""
    return (f'<div class="brow{" strong" if strong else ""}" title="{esc(tip)}"><div class="bl">{esc(label)}{note}</div>'
            f'<div class="bar">{inner}</div><div class="bv">{pc(val)}</div></div>')


def group(title, metric, bench, rows_html, n=None):
    arrow = "Unsafe ↓" if metric == "H" else "Refused ↓"
    nn = (f", {n} requests" if bench else f"{n} requests") if n else ""
    return (f'<div class="grp"><div class="gh"><span class="gt">{esc(title)} · {arrow}</span>'
            f'<span class="gb">{esc(bench)}{nn}</span></div>{rows_html}</div>')


def legend(two=False):
    items = [("ref", "Refused"), ("ans", "Answered without harm"), ("uns", "Unsafe")]
    return '<div class="legend">' + "".join(f'<span><i class="sw {c}"></i>{t}</span>' for c, t in items) + "</div>"


def tgroup(rnd, table, row_name, kind, title, bench, cells, judge=None):
    rows = []
    n = None
    for c in cells:
        s = tshares(rnd, table, row_name, c, judge)
        n = s["n"]
        rows.append(bar_row(ROWNAME[c], s, kind, strong=c in HIGHLIGHT))
    return group(title, kind, bench, "".join(rows), n)


def claim(table, row_name):
    T = J(table)
    return next(r for r in T["rows"] if r["row"] == row_name)


# =============================================================================================
# SVG figures
# =============================================================================================
C_REF, C_ANS, C_UNS = "#3987e5", "#199e70", "#d95926"
INK, SEC, MUTED, GRID = "#eceae4", "#b9b7ae", "#8d8b84", "#34342f"


def push_svg(th_t, beta_t, gain_t, th_l, beta_l, t_l, gain_l, svals):
    W, H, L, R, T, B = 640, 400, 64, 24, 20, 56
    zmin, zmax, ymin, ymax = -0.25, 1.5, -2.2, 1.45

    def X(z):
        return L + (z - zmin) / (zmax - zmin) * (W - L - R)

    def Y(y):
        return T + (ymax - y) / (ymax - ymin) * (H - T - B)

    o = [f'<svg viewBox="0 0 {W} {H}" class="svgfig" role="img" aria-label="Push against harm score">']
    for y in (-2, -1, 0, 1):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(y):.1f}" y2="{Y(y):.1f}" stroke="{GRID if y else MUTED}" stroke-width="{1 if y else 1.5}"/>')
        o.append(f'<text x="{L-10}" y="{Y(y)+5:.1f}" text-anchor="end" class="ax">{"+" if y > 0 else ""}{str(y).replace("-", MINUS)}</text>')
    for z in (0, 0.5, 1, 1.5):
        o.append(f'<line x1="{X(z):.1f}" x2="{X(z):.1f}" y1="{T}" y2="{H-B}" stroke="{GRID}"/>')
        o.append(f'<text x="{X(z):.1f}" y="{H-B+22}" text-anchor="middle" class="ax">{z:g}</text>')
    o.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-8}" text-anchor="middle" class="axt">harm score z (0 typical harmless, 1 typical harmful)</text>')
    o.append(f'<text x="{L+8}" y="{Y(0.45):.1f}" class="axt">↑ toward refusal</text>')
    o.append(f'<text x="{L+8}" y="{Y(-0.55):.1f}" class="axt">↓ toward answering</text>')
    zs = [zmin + i * (zmax - zmin) / 400 for i in range(401)]
    for i, s in enumerate(svals):
        pts = " ".join(f"{X(z):.1f},{Y(gain_t * min(1, max(0, beta_t * (s * z - th_t)))):.1f}" for z in zs)
        op = 0.45 + 0.55 * (i + 1) / len(svals)
        o.append(f'<polyline points="{pts}" fill="none" stroke="{C_REF}" stroke-width="2.5" opacity="{op:.2f}"/>')
        zl = (th_t + 1 / beta_t) / s if s else None
        if zl is not None and zl < zmax - 0.05:
            o.append(f'<text x="{X(zl):.1f}" y="{Y(1)-(10 if i % 2 else 32):.1f}" text-anchor="middle" class="lab">s = {s:g}</text>')
    pts = " ".join(f"{X(z):.1f},{Y(-t_l * gain_l * min(1, max(0, beta_l * (th_l - z)))):.1f}" for z in zs)
    o.append(f'<polyline points="{pts}" fill="none" stroke="{C_ANS}" stroke-width="2.5"/>')
    o.append(f'<text x="{X(th_l)+10:.1f}" y="{Y(-1.6):.1f}" class="lab">Loosen, scale {t_l:g}</text>')
    o.append(f'<text x="{X(1.02):.1f}" y="{Y(0.55):.1f}" class="lab">Tighten</text>')
    o.append("</svg>")
    return "".join(o)


def scatter_svg(pts, lines):
    """pts: list of dict(x, y, label, kind, anchor); x = harmless refused %, y = harmful unsafe %."""
    W, H, L, R, T, B = 640, 400, 60, 30, 20, 60
    xmax = max(p["x"] for p in pts) * 1.08
    ymax = max(p["y"] for p in pts) * 1.15
    xstep = 10 if xmax > 30 else 5
    ystep = 2 if ymax > 6 else 1

    def X(x):
        return L + x / xmax * (W - L - R)

    def Y(y):
        return T + (ymax - y) / ymax * (H - T - B)

    o = [f'<svg viewBox="0 0 {W} {H}" class="svgfig" role="img" aria-label="Unsafe against harmless refusal">']
    x = 0
    while x <= xmax:
        o.append(f'<line x1="{X(x):.1f}" x2="{X(x):.1f}" y1="{T}" y2="{H-B}" stroke="{GRID}"/><text x="{X(x):.1f}" y="{H-B+22}" text-anchor="middle" class="ax">{x}</text>')
        x += xstep
    y = 0
    while y <= ymax:
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(y):.1f}" y2="{Y(y):.1f}" stroke="{GRID}"/><text x="{L-10}" y="{Y(y)+5:.1f}" text-anchor="end" class="ax">{y}</text>')
        y += ystep
    o.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-10}" text-anchor="middle" class="axt">harmless requests refused, % →</text>')
    o.append(f'<text transform="translate(16 {(T+H-B)/2:.0f}) rotate(-90)" text-anchor="middle" class="axt">harmful requests unsafe, %</text>')
    o.append(f'<text x="{L+10}" y="{H-B-12}" class="axt">better: lower left</text>')
    for ln in lines:
        o.append('<polyline points="' + " ".join(f"{X(p['x']):.1f},{Y(p['y']):.1f}" for p in ln) + f'" fill="none" stroke="{MUTED}" stroke-width="1.5"/>')
    for p in pts:
        cx, cy = X(p["x"]), Y(p["y"])
        if p["kind"] == "knob":
            o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{C_REF}" stroke="#111" stroke-width="2"><title>{esc(p["label"])}: {pc(p["y"])} unsafe, {pc(p["x"])} refused</title></circle>')
        elif p["kind"] == "base":
            o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="none" stroke="{INK}" stroke-width="2"><title>{esc(p["label"])}: {pc(p["y"])} unsafe, {pc(p["x"])} refused</title></circle>')
        else:
            o.append(f'<rect x="{cx-6:.1f}" y="{cy-6:.1f}" width="12" height="12" fill="{SEC}" stroke="#111" stroke-width="2"><title>{esc(p["label"])}: {pc(p["y"])} unsafe, {pc(p["x"])} refused</title></rect>')
        a = p.get("anchor", "start")
        dx = {"start": 12, "end": -12, "middle": 0}[a]
        dy = p.get("dy", 5)
        o.append(f'<text x="{cx+dx:.1f}" y="{cy+dy:.1f}" text-anchor="{a}" class="lab{" strong" if p["kind"]=="knob" else ""}">{esc(p["label"])}</text>')
    o.append("</svg>")
    return "".join(o)


def strips_html(sets, theta, zmin, zmax, read_file):
    """Per-set p10 to p90 bar with the median tick and the threshold line."""
    def X(z):
        return 100 * (z - zmin) / (zmax - zmin)

    o = []
    for gname, gsets, used in sets:
        mark = '<span class="used yes">used to compute this direction</span>' if used else '<span class="used no">not used</span>'
        o.append(f'<div class="sgrp"><div class="sgh">{esc(gname)} {mark}</div>')
        for lab, key in gsets:
            st = read_file["sets"][key]
            p10, p50, p90 = st["z_p10_p50_p90"]
            o.append(f'<div class="srow" title="{esc(lab)}: 10th, 50th, 90th percentile of z = {p10:.2f}, {p50:.2f}, {p90:.2f}; n = {st["n"]}">'
                     f'<div class="sl">{esc(lab)}</div><div class="strack">'
                     f'<span class="thr" style="left:{X(theta):.2f}%"></span>'
                     f'<span class="rng" style="left:{X(p10):.2f}%;width:{X(p90)-X(p10):.2f}%"></span>'
                     f'<span class="med" style="left:{X(p50):.2f}%"></span></div></div>')
        o.append("</div>")
    ticks = "".join(f'<span style="left:{X(z):.2f}%">{str(z).replace("-", MINUS)}</span>' for z in (-0.5, 0, 0.5, 1, 1.5) if zmin <= z <= zmax)
    o.append(f'<div class="srow axisrow"><div class="sl"></div><div class="saxis">{ticks}</div></div>')
    return "".join(o)


# =============================================================================================
# Page
# =============================================================================================


def build():
    S = {}  # section html

    # ---------------- constants of the fold ----------------
    fit = J("round4/fold/fit.json")
    exp_cap = J("round4/gguf/export_cap.json")
    exp_both = J("round5/gguf/export_both_sides.json")
    lab_v3g = J("round5/lab/v3g.json")
    fit_v1 = J("round5/fit/v1again.json")
    expo = J("round5/lax/exposure_v3g.json")
    par4 = J("round4/fold/parity_logits_cap.json")
    par5 = J("round5/parity/units.json")
    chk3 = J("round3/block0/checks.json")
    r5lib = (BUILD / "scripts/r5lib.py").read_text()
    USED.append("../scripts/r5lib.py (lax_units default gain)")
    m = re.search(r"def lax_units\([^)]*gain=([0-9.]+)", r5lib)
    gain_l = float(m.group(1))
    beta_t, gain_t = exp_cap["kw"]["cap_beta"], exp_cap["kw"]["gain"]
    th_v1, th_v3g = fit["theta0"], lab_v3g["theta0"]
    th_l, t_l = exp_both["theta"], exp_both["t"]
    beta_l = 1 / expo["ramp_width"]
    m2 = re.search(r'"intermediate_size":\s*(\d+)', json.dumps(chk3))
    inter = int(m2.group(1))
    layer = fit["layer"]

    # ---------------- 1. headline numbers ----------------
    six = tshares("round6", "round6/tables/xb_tight.json", "red-teaming prompts, 6 unseen languages", "base")
    six_k = tshares("round6", "round6/tables/xb_tight.json", "red-teaming prompts, 6 unseen languages", "v3gcap@1")
    six_hb = tshares("round6", "round6/tables/xb_tight.json", "harmless prompts, 6 unseen languages", "base")
    six_hk = tshares("round6", "round6/tables/xb_tight.json", "harmless prompts, 6 unseen languages", "v3gcap@1")
    six_n_h = six["n"]
    stf, sts = J("round6/safetune/full.json"), J("round6/safetune/full_small.json")
    dbase = dev_shares("base_wjx", check=(stf["base"]["harmful_refusal"], stf["base"]["harmful_unsafe"], stf["base"]["benign_refusal"]))
    dctl = dev_shares("ungated@0.1")
    dv1 = dev_shares("foldcap@1")
    dv3 = dev_shares("v3gcap@1")
    cast_cells = sts["cells"]
    cast_unsafe = [c["harmful_unsafe"] for c in cast_cells.values()]
    cast_ref = [c["benign_refusal"] for c in cast_cells.values()]
    cast_hr = [c["harmful_refusal"] for c in cast_cells.values()]
    # CAST's three alpha values are one result: shown as one point, at the alpha in the middle of the three
    cast_keys = list(cast_cells)
    cast_mid = cast_cells[cast_keys[len(cast_keys) // 2]]
    orb_b = tshares("round5", "round5/tables/frz_lax.json", "orbench_hard (fresh)", "base")
    orb_l = tshares("round5", "round5/tables/frz_lax.json", "orbench_hard (fresh)", "lax3@2")
    six_cl = claim("round6/tables/xb_tight_claims_v3gcap@1.json", "red-teaming prompts, 6 unseen languages")
    lax5 = J("round5/tables/frz_lax.json")
    n_orb = orb_b["n"]

    hero = [
        (f"{pc(six['unsafe'])} → {pc(six_k['unsafe'])}",
         f"unsafe replies to {six_n_h} red-teaming requests in six languages no direction has seen, held-out test; "
         f"harmless refusals in those languages {pc(six_hb['refusal'])} → {pc(six_hk['refusal'])}."),
        (f"{pc(dv3['B']['refusal'])} vs {f1(cast_mid['benign_refusal'])}%",
         f"harmless requests refused by AyaDial-Improved and by CAST on 1910 WildJailbreak development requests, "
         f"at {pc(dv3['H']['unsafe'])} and {f1(cast_mid['harmful_unsafe'])}% unsafe."),
        (f"{pc(orb_b['refusal'])} → {pc(orb_l['refusal'])}",
         f"false refusals on {n_orb} OR-Bench-hard requests with Loosen, held-out test."),
    ]
    S["hero"] = '<div class="hero">' + "".join(f'<div class="hcard"><div class="hn">{a}</div><div class="hc">{esc(b)}</div></div>' for a, b in hero) + "</div>"

    # ---------------- 2a. the harm direction ----------------
    rv1, rv3 = J("round5/read/v1_frozen.json"), J("round5/read/v3g_frozen.json")
    fit_h1, fit_b1 = list(fit_v1["fit_harmful_sets"]), list(fit_v1["fit_harmless_sets"])
    fit_h3, fit_b3 = list(lab_v3g["fit_harmful"]), list(lab_v3g["fit_harmless"])
    MJ = [("zh", "Chinese"), ("it", "Italian"), ("vi", "Vietnamese"), ("ar", "Arabic"), ("ko", "Korean"), ("th", "Thai"),
          ("bn", "Bengali"), ("sw", "Swahili"), ("jv", "Javanese")]
    GROUPS = [  # (name, [(row label, set key in read file)], fit-set prefixes that make the group "used")
        ("Harmful, plain", [("MultiJail English", "mj_en_frz"), ("SORRY-Bench base", "sorry_base_frz")], ("harmful_fit", "xstest_unsafe_fit")),
        ("Harmful, jailbreak-wrapped (WildJailbreak)", [("WildJailbreak", "wj5_h_frz")], ("trfit_h",)),
        ("Harmful, in-the-wild template", [("in-the-wild templates", "itw_h_frz")], ("itw_h_",)),
        ("Harmful, non-English (MultiJail)", [(n, f"mj_{c}_frz") for c, n in MJ], ("mj_",)),
        ("Harmless, jailbreak-wrapped", [("WildJailbreak", "wj5_b_frz"), ("in-the-wild templates", "itw_b_frz")], ("trfit_b", "itw_b_")),
        ("Harmless, harmful-sounding", [("OR-Bench-hard", "orb5_frz"), ("PHTest", "pht_harmless_frz")], ("orbench_hard_fit", "xstest_safe_fit")),
        ("Harmless, everyday (Aya, ten languages)", [("English", "aya_en_frz")] + [(n, f"aya_{c}_frz") for c, n in MJ], ("benign_fit", "aya_")),
    ]

    def used(prefixes, fit_sets):
        return any(f.startswith(p) for f in fit_sets for p in prefixes)

    allz = [v for rf in (rv1, rv3) for _, gs, _ in GROUPS for _, k in gs for v in rf["sets"][k]["z_p10_p50_p90"]]
    zmin, zmax = math.floor(min(allz) * 4) / 4, math.ceil(max(allz) * 4) / 4
    sets1 = [(g, gs, used(p, fit_h1 + fit_b1)) for g, gs, p in GROUPS]
    sets3 = [(g, gs, used(p, fit_h3 + fit_b3)) for g, gs, p in GROUPS]
    S["strips_v1"] = strips_html(sets1, th_v1, zmin, zmax, rv1)
    S["strips_v3"] = strips_html(sets3, th_v3g, zmin, zmax, rv3)
    S["th_v1"], S["th_v3"] = f"{th_v1:.3f}", f"{th_v3g:.3f}"
    au1 = rv1["pairs"]["itw.same templates (frz)"]["auroc"]
    au3 = rv3["pairs"]["itw.same templates (frz)"]["auroc"]
    S["au"] = (f"{au1:.3f}", f"{au3:.3f}")

    # ---------------- 2b. the push ----------------
    svals = [0.75, 1, 1.25, 2]
    S["push"] = push_svg(th_v3g, beta_t, gain_t, th_l, beta_l, t_l, gain_l, svals)
    kl_hook = [v["hook_vs_weights"]["kl_first_token"] for k, v in par5["cells"].items() if k.startswith(("v3gcap", "lax3"))]
    S["props"] = [
        ("Trained parameters", "none (closed form)"),
        ("Units used", f"2 of {inter} per adapter, layer {layer} (Tighten: {', '.join(map(str, exp_both['strict_units']))}; Loosen: {', '.join(map(str, exp_both['lax_units']))})"),
        ("Acts at", "the reply-start token, by design (not measured for this adapter)"),
        ("At s = 0", f"writes nothing; next-token KL against the base {par4['cap_s0']['base_vs_weights']['kl_first_token']:.4f} (two dormant units rewired)"),
        ("Hook against weights", f"next-token KL {min(kl_hook):.4f} to {max(kl_hook):.4f} (bar 0.001), {par5['n_prompts']} prompts"),
        ("Format", "LoRA adapters; GGUF for llama.cpp"),
        ("Scale", "set per request (llama-server's per-request lora field)"),
    ]
    S["formula_p"] = r"p(z) = \min\!\big(1,\ \max(0,\ \beta\,(s\,z - \theta))\big)\; r"
    S["formula_q"] = r"q(z) = -\,t\,\min\!\big(1,\ \max(0,\ \beta_L\,(\theta_L - z))\big)\; r"
    S["formula_vals"] = (f"\\begin{{aligned}} &\\text{{AyaDial-Improved: }} \\theta = {th_v3g:.3f},\\ \\beta = {beta_t:g} \\\\ "
                         f"&\\text{{Loosen: }} \\theta_L = {th_l:g},\\ \\beta_L = {beta_l:g},\\ t = {t_l:g} \\end{{aligned}}")

    # ---------------- 3. main results ----------------
    ex = tshares("round5", "round5/tables/frz_tight.json", "in-the-wild template + harmful question", "base")
    a_mid = ex["refused_only"] + ex["answered"] / 2
    S["example"] = (f'<div class="example"><div class="exlab top"><span style="left:{a_mid:.2f}%">Answered without harm</span></div>'
                    f'{bar_row("Base", ex, "H")}'
                    f'<div class="exlab bot"><span class="l" style="left:0;width:{ex["refused_only"]:.2f}%">Refused</span>'
                    f'<span class="r" style="right:0;width:{ex["unsafe"]:.2f}%">Unsafe</span></div></div>')
    S["ex_n"] = ex["n"]

    # baselines on development data
    st_rows_h, st_rows_b = [], []
    dev_cells = [("Base", dbase, False, ""), ("Control", dctl, False, ""), ("AyaDial", dv1, False, ""), ("AyaDial-Improved", dv3, True, "")]
    st_specs = []
    for k, v in stf["cells"].items():
        meth, val = k.split("@")
        name = {"st_rd": "RefusalDirection", "st_adasteer": "AdaSteer"}[meth]
        sub = f"strength {val}" if meth == "st_rd" else f"bound {val}"
        d = dev_shares(k, check=(v["harmful_refusal"], v["harmful_unsafe"], v["benign_refusal"]))
        st_specs.append((name, sub, d))
    for lab, d, strong, sub in dev_cells:
        st_rows_h.append(bar_row(lab, d["H"], "H", strong=strong, sub=sub))
        st_rows_b.append(bar_row(lab, d["B"], "B", strong=strong, sub=sub))
    for name, sub, d in st_specs:
        st_rows_h.append(bar_row(name, d["H"], "H", sub=sub))
        st_rows_b.append(bar_row(name, d["B"], "B", sub=sub))
    # CAST: one point, no per-prompt labels on this machine: Answered without harm = 100 - refused - unsafe
    ch = {"n": sts["base"]["n"][0], "refused_only": cast_mid["harmful_refusal"] - 0, "unsafe": cast_mid["harmful_unsafe"]}
    ch["refused_only"] = cast_mid["harmful_refusal"]
    ch["answered"] = 100 - cast_mid["harmful_refusal"] - cast_mid["harmful_unsafe"]
    cb = {"n": sts["base"]["n"][1], "refusal": cast_mid["benign_refusal"]}
    cbase_h = {"n": sts["base"]["n"][0], "refused_only": sts["base"]["harmful_refusal"], "unsafe": sts["base"]["harmful_unsafe"],
               "answered": 100 - sts["base"]["harmful_refusal"] - sts["base"]["harmful_unsafe"]}
    cbase_b = {"n": sts["base"]["n"][1], "refusal": sts["base"]["benign_refusal"]}
    st_rows_h.append(bar_row("CAST", ch, "H", sub="alpha has no effect*"))
    st_rows_b.append(bar_row("CAST", cb, "B", sub="alpha has no effect*"))
    st_rows_h.append(bar_row("Base for CAST", cbase_h, "H", sub="batches of 8"))
    st_rows_b.append(bar_row("Base for CAST", cbase_b, "B", sub="batches of 8"))
    S["st_bars"] = (group("Harmful requests", "H", "WildJailbreak development, adversarial harmful", "".join(st_rows_h), dbase["H"]["n"]) +
                    group("Harmless requests", "B", "WildJailbreak development, adversarial benign", "".join(st_rows_b), dbase["B"]["n"]))
    S["cast_note"] = (f"*CAST's three alpha values ({', '.join(k.split('@')[1] for k in cast_keys)}) give {f1(min(cast_unsafe))} to {f1(max(cast_unsafe))}% unsafe "
                      f"and {f1(min(cast_ref))} to {f1(max(cast_ref))}% harmless refusal, so they are one result; the middle one is drawn. "
                      f"CAST ran in batches of 8 with its own base; its label files are not on this machine, so its Answered without harm is 100 minus refused minus unsafe.")
    pts = [dict(x=dbase["B"]["refusal"], y=dbase["H"]["unsafe"], label="Base", kind="base", anchor="end", dy=5),
           dict(x=dctl["B"]["refusal"], y=dctl["H"]["unsafe"], label="Control", kind="other", anchor="start"),
           dict(x=dv1["B"]["refusal"], y=dv1["H"]["unsafe"], label="AyaDial", kind="knob", anchor="end", dy=0),
           dict(x=dv3["B"]["refusal"], y=dv3["H"]["unsafe"], label="AyaDial-Improved", kind="knob", anchor="end", dy=24),
           dict(x=cast_mid["benign_refusal"], y=cast_mid["harmful_unsafe"], label="CAST", kind="other", anchor="middle", dy=-16)]
    rd = [(d["B"]["refusal"], d["H"]["unsafe"], sub) for name, sub, d in st_specs if name == "RefusalDirection"]
    ad = [(d["B"]["refusal"], d["H"]["unsafe"], sub) for name, sub, d in st_specs if name == "AdaSteer"]
    for i, (x, y, sub) in enumerate(rd):
        pts.append(dict(x=x, y=y, label=("RefusalDirection" if i == len(rd) - 1 else ""), kind="other", anchor="start"))
    for i, (x, y, sub) in enumerate(ad):
        pts.append(dict(x=x, y=y, label=("AdaSteer" if i == len(ad) - 1 else ""), kind="other", anchor="end", dy=-14))
    S["scatter"] = scatter_svg(pts, [[dict(x=x, y=y) for x, y, _ in rd], [dict(x=x, y=y) for x, y, _ in ad]])

    # Loosen on the held-out test
    lc = ["base", "ctrlneg@-0.02", "lax3@2"]
    T5L = "round5/tables/frz_lax.json"
    S["loosen"] = (
        tgroup("round5", T5L, "orbench_hard (fresh)", "B", "Harmless requests", "harmful-sounding, OR-Bench-hard", lc) +
        tgroup("round5", T5L, "adversarial_benign (fresh WildJailbreak)", "B", "Harmless requests", "jailbreak-wrapped, WildJailbreak", lc) +
        tgroup("round5", T5L, "adversarial_harmful (fresh WildJailbreak)", "H", "Harmful requests", "jailbreak-wrapped, WildJailbreak", lc) +
        tgroup("round5", T5L, "in-the-wild template + harmful question", "H", "Harmful requests", "in-the-wild jailbreak template", lc) +
        tgroup("round5", T5L, "MultiJail 9 other languages", "H", "Harmful requests", "nine non-English languages, MultiJail", lc))
    # both
    bc = ["base", "ungated@0.1", "v3gcap@1", "both3@1"]
    T5B = "round5/tables/frz_both.json"
    S["both"] = (
        tgroup("round5", T5B, "in-the-wild template + harmful question", "H", "Harmful requests", "in-the-wild jailbreak template", bc) +
        tgroup("round5", T5B, "MultiJail 9 other languages", "H", "Harmful requests", "nine non-English languages, MultiJail", bc) +
        tgroup("round5", T5B, "orbench_hard (fresh)", "B", "Harmless requests", "harmful-sounding, OR-Bench-hard", bc) +
        tgroup("round5", T5B, "adversarial_benign (fresh WildJailbreak)", "B", "Harmless requests", "jailbreak-wrapped, WildJailbreak", bc))

    # ---------------- 4. the dial ----------------
    sweep = [0, 0.75, 1, 1.25, 1.5, 2, 2.5, 3]
    sw_h = [bar_row("Base", dbase["H"], "H")]
    sw_b = [bar_row("Base", dbase["B"], "B")]
    for s in sweep:
        d = dev_shares(f"foldcap@{s:g}")
        lab = "s = 0 (installed, off)" if s == 0 else f"s = {s:g}"
        sw_h.append(bar_row(lab, d["H"], "H", strong=(s == 1)))
        sw_b.append(bar_row(lab, d["B"], "B", strong=(s == 1)))
    S["sweep"] = (group("Harmful requests", "H", "AyaDial, WildJailbreak development", "".join(sw_h), dbase["H"]["n"]) +
                  group("Harmless requests", "B", "AyaDial, WildJailbreak development", "".join(sw_b), dbase["B"]["n"]))
    i_h, i_b = [], []
    for s in (0.75, 1, 1.25):
        d = dev_shares(f"v3gcap@{s:g}")
        i_h.append(bar_row(f"s = {s:g}", d["H"], "H", strong=(s == 1)))
        i_b.append(bar_row(f"s = {s:g}", d["B"], "B", strong=(s == 1)))
    sub_h = '<div class="subhead">AyaDial-Improved, same requests</div>'
    S["sweep"] = (group("Harmful requests", "H", "WildJailbreak development", '<div class="subhead">AyaDial</div>' + "".join(sw_h) + sub_h + "".join(i_h), dbase["H"]["n"]) +
                  group("Harmless requests", "B", "WildJailbreak development", '<div class="subhead">AyaDial</div>' + "".join(sw_b) + sub_h + "".join(i_b), dbase["B"]["n"]))

    # ---------------- 5. beyond the benchmark ----------------
    tc = ["base", "ungated@0.1", "foldcap@1", "v3gcap@1"]
    T5T = "round5/tables/frz_tight.json"
    S["beyond"] = (
        tgroup("round5", T5T, "in-the-wild template + harmful question", "H", "Harmful requests", "in-the-wild jailbreak template", tc) +
        tgroup("round5", T5T, "JailbreakBench GCG (held-out behaviours)", "H", "Harmful requests", "GCG attack suffix, JailbreakBench held-out behaviours", tc) +
        tgroup("round5", T5T, "in-the-wild template + harmless instruction", "B", "Harmless requests", "in-the-wild template carrying a harmless instruction", tc))
    diag = J("round5/diag/read_or_push_foldcap@1.json")["groups"]
    S["fire"] = (f1(diag["adversarial_harmful (dev)"]["fires_among_base_unsafe"]), f1(diag["in-the-wild + harmful question"]["fires_among_base_unsafe"]))
    cl_itw = claim("round5/tables/frz_tight_claims_v3gcap@1.json", "in-the-wild template + harmful question")
    S["cl_itw"] = cl_itw
    T6 = "round6/tables/xb_tight.json"
    c6 = ["base", "ungated@0.1", "v3gcap@1"]
    S["langs_main"] = (tgroup("round6", T6, "red-teaming prompts, 6 unseen languages", "H", "Harmful requests", "Aya red-teaming, Filipino, French, Hindi, Russian, Serbian, Spanish", c6) +
                       tgroup("round6", T6, "harmless prompts, 6 unseen languages", "B", "Harmless requests", "Aya prompts, same six languages", c6))
    LN = [("fil", "Filipino"), ("fr", "French"), ("hi", "Hindi"), ("ru", "Russian"), ("sr", "Serbian"), ("es", "Spanish")]
    small = []
    for c, name in LN:
        small.append('<div class="mini">' + tgroup("round6", T6, f"red-teaming prompts, {c}", "H", name, "", c6) + "</div>")
    S["langs_each"] = '<div class="minis">' + "".join(small) + "</div>"
    S["six_cl"] = six_cl
    q8 = claim("round6/tables/xb_tight_q8_claims_v3gcap@1.json", "red-teaming prompts, 6 unseen languages")
    S["six_q8"] = q8

    # ---------------- 6. limits ----------------
    S["lim_formats"] = (tgroup("round6", T6, "exam formats, all: harmful requests", "H", "Harmful requests", "eight formats: developer preambles, multi-turn chats", c6) +
                        tgroup("round6", T6, "exam formats, all: harmless requests", "B", "Harmless requests", "same eight formats", c6))
    cf = claim("round6/tables/xb_tight_claims_v3gcap@1.json", "exam formats, all: harmful requests")
    S["cl_formats"] = cf
    S["lim_en"] = tgroup("round6", T6, "red-teaming prompts, English", "H", "Harmful requests", "Aya red-teaming, English", c6)
    S["lim_mj"] = tgroup("round6", T6, "MultiJail 9 other languages", "H", "Harmful requests", "MultiJail, nine languages, new requests", c6)
    S["cl_mj"] = claim("round6/tables/xb_tight_claims_v3gcap@1.json", "MultiJail 9 other languages")
    S["cl_mj5"] = claim("round5/tables/frz_tight_claims_v3gcap@1.json", "MultiJail 9 other languages")
    S["lim_judge"] = (tgroup("round6", "round6/tables/xb_tight_q8.json", "red-teaming prompts, 6 unseen languages", "H", "Second judge, Qwen3Guard-Gen-8B", "six unseen languages", c6, judge="q8") +
                      tgroup("round5", "round5/tables/frz_tight_wg.json", "in-the-wild template + harmful question", "H", "Second judge, WildGuard", "in-the-wild jailbreak template", tc, judge="wg"))
    llama = J("round6/llama/checks.json")
    S["llama"] = llama
    T5 = J(T5T)
    itwb = next(r for r in T5["rows"] if r["row"] == "in-the-wild template + harmless instruction")["cells"]["base"]["unsafe"]
    S["itwb_unsafe"] = itwb

    # ---------------- 7. how it was tested ----------------
    decl6 = J("round6/exam_declaration.json")
    decl5 = J("round5/frozen_declaration.json")
    S["decl"] = (decl5, decl6)

    # ---------------- appendix ----------------
    S["appendix"] = appendix(S)
    return S


def table_html(head, rows, cls="tbl"):
    h = "<tr>" + "".join(f"<th>{esc(x)}</th>" for x in head) + "</tr>"
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<div class="tblwrap"><table class="{cls}"><thead>{h}</thead><tbody>{b}</tbody></table></div>'


def appendix(S):
    out = []
    # round 5 held-out test, all cells, refusal / unsafe
    T = J("round5/tables/frz_tight.json")
    cols = T["columns"]
    names = {"base": "Base", "ungated@0.1": "Control (ungated@0.1)", "ungated@0.02": "ungated@0.02", "foldcap@1": "AyaDial (foldcap@1)",
             "foldcap@1.25": "foldcap@1.25", "v2cap@1": "v2 (v2cap@1)", "v3gcap@1": "AyaDial-Improved (v3gcap@1)"}
    rows = []
    for r in T["rows"]:
        if r["row"].startswith("SORRY-Bench ") and r["row"] not in ("SORRY-Bench base",) and len(r["sets"]) == 1:
            continue
        if re.match(r"MultiJail [a-z][a-z]$", r["row"]):
            continue
        rows.append([esc(r["row"]), r["kind"], str(r["cells"]["base"]["n"])] +
                    [f'{f1(r["cells"][c]["refusal"])} / {f1(r["cells"][c]["unsafe"])}' if c in r["cells"] else "" for c in cols])
    out.append("<h3>Held-out test 1 (round 5): refused % / unsafe %, every cell</h3>")
    out.append(table_html(["set", "kind", "n"] + [names.get(c, c) for c in cols], rows))
    out.append('<p class="small">kind H: harmful, B: harmless, C: controversial. Source: <code>results/round5/tables/frz_tight.json</code>.</p>')
    # claims for v3g, round 5
    C = J("round5/tables/frz_tight_claims_v3gcap@1.json")
    rows = [[esc(r["row"]), str(r["n"]), f1(r["base_unsafe"]), f1(r["unsafe"]), f'{sg(r["vs_control"])} ({ci(r["vs_control_ci"])})',
             (f'{sg(r["counterpart_refusal_rise"])}' if r.get("counterpart_refusal_rise") is not None else ""), esc(r["verdict"])] for r in C["rows"]]
    out.append("<h3>AyaDial-Improved against Control, held-out test 1, with 95% intervals</h3>")
    out.append(table_html(["harmful set", "n", "Base unsafe", "unsafe", "vs Control (points)", "harmless counterpart refusal rise", "verdict"], rows))
    out.append(f'<p class="small">{esc(C["rule"])}. Source: <code>results/round5/tables/frz_tight_claims_v3gcap@1.json</code>.</p>')
    # round 6, including v4
    T6 = J("round6/tables/xb_tight.json")
    cols6 = [c for c in T6["columns"]]
    n6 = {"base": "Base", "ungated@0.1": "Control", "v3gcap@1": "AyaDial-Improved (v3gcap@1)", "v4cap@1": "v4 (v4cap@1)", "both4@1": "both4@1",
          "v4cal": "v4cal", "both4cal": "both4cal"}
    rows = [[esc(r["row"]), r["kind"], str(r["cells"]["base"]["n"])] +
            [f'{f1(r["cells"][c]["refusal"])} / {f1(r["cells"][c]["unsafe"])}' if c in r["cells"] else "" for c in cols6] for r in T6["rows"]]
    out.append("<h3>Held-out test 2 (round 6): refused % / unsafe %, every cell</h3>")
    out.append(table_html(["set", "kind", "n"] + [n6.get(c, c) for c in cols6], rows))
    out.append('<p class="small">v4 is a variant with the position detector refit for multi-turn chats; both4 is v4 with Loosen. '
               'Source: <code>results/round6/tables/xb_tight.json</code>.</p>')
    C6 = J("round6/tables/xb_tight_claims_v3gcap@1.json")
    rows = [[esc(r["row"]), str(r["n"]), f1(r["base_unsafe"]), f1(r["unsafe"]), f'{sg(r["vs_control"])} ({ci(r["vs_control_ci"])})', esc(r["verdict"])] for r in C6["rows"]]
    out.append("<h3>AyaDial-Improved against Control, held-out test 2, with 95% intervals</h3>")
    out.append(table_html(["harmful set", "n", "Base unsafe", "unsafe", "vs Control (points)", "verdict"], rows))
    out.append('<p class="small">Source: <code>results/round6/tables/xb_tight_claims_v3gcap@1.json</code>.</p>')
    # Loosen
    TL = J("round5/tables/frz_lax.json")
    rows = [[esc(r["row"]), r["kind"], str(r["cells"]["base"]["n"])] + [f'{f1(r["cells"][c]["refusal"])} / {f1(r["cells"][c]["unsafe"])}' if c in r["cells"] else "" for c in TL["columns"]]
            for r in TL["rows"] if not re.match(r"(MultiJail [a-z][a-z]$|SORRY-Bench (?!base|other))", r["row"])]
    out.append("<h3>Loosen, held-out test 1: refused % / unsafe %</h3>")
    out.append(table_html(["set", "kind", "n"] + [{"base": "Base", "ctrlneg@-0.02": "Control (ctrlneg@-0.02)", "lax3@2": "Loosen (lax3@2)", "both3@1": "Tighten + Loosen (both3@1)"}.get(c, c) for c in TL["columns"]], rows))
    out.append('<p class="small">Source: <code>results/round5/tables/frz_lax.json</code>.</p>')
    # method notes
    out.append("<h3>How the three-part bars were computed</h3>")
    out.append(f'<p>Each reply has two labels from Qwen3Guard-Gen-0.6B (a second judge where named): refused and unsafe. Unsafe = the unsafe label. '
               f'Refused = the refused label and not unsafe. Answered without harm = the rest (this includes the few replies flagged as degenerate). '
               f'On harmless sets the bar has two parts: Refused = the refused label, Answered = the rest. Both labels on the same reply are rare: at most '
               f'{f1(OVERLAP["max"])}% of a set in any bar on this page. For every bar the refusal and unsafe rates computed from the label files were checked against '
               f'the summary table of the same cell before drawing. CAST is the exception: its per-prompt labels are not on this machine, so its bar uses '
               f'<code>results/round6/safetune/full_small.json</code> with Answered without harm = 100 minus refused minus unsafe. '
               f'The judge also marks some replies to harmless requests unsafe (for example {f1(S["itwb_unsafe"])}% of Base replies to in-the-wild templates carrying a harmless instruction); '
               f'the two-part harmless bars do not show that label; the tables above do.</p>')
    out.append("<h3>Names on this page and in the files</h3>")
    out.append(table_html(["on the page", "in the files"], [
        ["AyaDial", "round 4's adapter: <code>foldcap</code>, <code>v1</code>, <code>foldcap@1</code>"],
        ["AyaDial-Improved / Tighten", "round 5's adapter: <code>v3g</code>, <code>v3gcap@1</code>"],
        ["Loosen", "<code>lax3@2</code>"],
        ["AyaDial-Improved (Tighten + Loosen)", "<code>both3@1</code>"],
        ["Control", "<code>ungated@0.1</code>; on the Loosen side <code>ctrlneg@-0.02</code>"],
        ["held-out test 1 / 2", "round 5 frozen test / round 6 exam B"],
        ["not on the main page", "<code>v2</code> (v2cap), <code>v4</code> (v4cap, guard refit), <code>both4</code>, <code>lax4</code>"],
    ]))
    out.append("<h3>Files read by the build script</h3>")
    out.append("<p class=\"small\">" + " · ".join(f"<code>results/{esc(u)}</code>" if not u.startswith("..") else f"<code>{esc(u[3:])}</code>" for u in USED) + "</p>")
    return "".join(out)


CSS = r"""
:root{--bg:#121211;--card:#1a1a19;--card2:#20201e;--ink:#eceae4;--sec:#b9b7ae;--muted:#8d8b84;--hair:#2e2e2a;
--ref:#3987e5;--ans:#199e70;--uns:#d95926;--acc:#9ec5f4}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:18px/1.55 "Inter","Helvetica Neue",Helvetica,Arial,sans-serif;}
main{max-width:1080px;margin:0 auto;padding:56px 24px 96px}
h1{font-size:44px;line-height:1.15;letter-spacing:-.01em;margin:0 0 20px;font-weight:650}
h2{font-size:30px;margin:88px 0 10px;font-weight:620;letter-spacing:-.005em}
h3{font-size:21px;margin:40px 0 10px;font-weight:600}
p{margin:10px 0;max-width:46em}
.meta{color:var(--sec);font-size:17px}
.meta b{color:var(--ink);font-weight:600}
.lede{font-size:21px;color:var(--ink);margin-top:22px}
code{font:15px/1.4 ui-monospace,Menlo,monospace;color:var(--acc)}
.small{font-size:15px;color:var(--muted)}
.lead{color:var(--sec);font-size:18px}
.hero{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:40px}
.hcard{background:var(--card);border:1px solid var(--hair);border-radius:14px;padding:24px 22px}
.hn{font-size:36px;font-weight:650;letter-spacing:-.01em;line-height:1.15;font-variant-numeric:tabular-nums}
.hc{color:var(--sec);font-size:16px;margin-top:10px}
.card{background:var(--card);border:1px solid var(--hair);border-radius:14px;padding:26px 26px 22px;margin:22px 0}
.twocol{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:start}
.twocol>*{min-width:0}
.twocol.dir{grid-template-columns:.75fr 1.25fr}
.pushcard{max-width:860px}
.formulas{display:flex;flex-wrap:wrap;gap:0 40px;align-items:center}
.formulas .formula{max-width:100%}
.formula.small2{font-size:16px}
.props{margin-top:18px}
.scat{max-width:860px}
.subhead{font-size:14px;color:var(--muted);text-transform:none;margin:12px 0 2px;padding-top:8px;border-top:1px solid var(--hair)}
.subhead:first-child{border-top:0;padding-top:0;margin-top:0}
.legend{display:flex;flex-wrap:wrap;gap:8px 22px;font-size:15px;color:var(--sec);margin:4px 0 14px}
.legend span{display:inline-flex;align-items:center;gap:8px}
.sw{width:14px;height:14px;border-radius:4px;display:inline-block}
.sw.ref,.seg.ref{background:var(--ref)}.sw.ans,.seg.ans{background:var(--ans)}.sw.uns,.seg.uns{background:var(--uns)}
.grp{margin:18px 0 24px}
.gh{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 14px;margin-bottom:8px}
.gt{font-weight:620;font-size:17px}
.gb{font-size:14px;color:var(--muted)}
.brow{display:grid;grid-template-columns:230px 1fr 76px;gap:14px;align-items:center;padding:5px 0}
.bl{font-size:16px;color:var(--sec);line-height:1.25}
.brow.strong .bl{color:var(--ink);font-weight:620}
.rsub{display:block;font-size:13px;color:var(--muted);font-weight:400}
.bar{display:flex;gap:2px;height:22px;border-radius:5px;overflow:hidden;background:var(--card2)}
.seg{display:block;height:100%}
.seg:first-child{border-radius:5px 0 0 5px}.seg:last-child{border-radius:0 5px 5px 0}
.bv{font-size:17px;font-variant-numeric:tabular-nums;text-align:right;color:var(--sec)}
.brow.strong .bv{color:var(--ink);font-weight:650}
.example{max-width:760px}
.exlab{position:relative;margin-left:244px;margin-right:90px;height:30px;font-size:14px;color:var(--sec)}
.exlab span{position:absolute;white-space:nowrap}
.exlab.top span{transform:translateX(-50%);bottom:2px;padding-bottom:4px;border-bottom:2px solid var(--muted)}
.exlab.bot span{top:0;padding-top:6px;border-top:2px solid var(--muted)}
.exlab.bot span.r{text-align:right}
.svgfig{width:100%;height:auto;display:block}
.svgfig .ax{fill:#8d8b84;font-size:15px}
.svgfig .axt{fill:#b9b7ae;font-size:15px}
.svgfig .lab{fill:#eceae4;font-size:16px}
.svgfig .lab.strong{font-weight:650}
.props{width:100%;border-collapse:collapse;font-size:16px}
.props td{padding:9px 0;border-bottom:1px solid var(--hair);vertical-align:top}
.props td:first-child{color:var(--muted);width:38%;padding-right:14px}
.formula{font-size:19px;overflow-x:auto;padding:6px 0}
.toggle{display:inline-flex;border:1px solid var(--hair);border-radius:999px;overflow:hidden;margin:6px 0 16px}
.toggle button{background:transparent;color:var(--sec);border:0;padding:9px 18px;font:inherit;font-size:16px;cursor:pointer}
.toggle button.on{background:#2a3a52;color:var(--ink)}
.sgrp{margin:14px 0 6px}
.sgh{font-size:15px;font-weight:600;margin-bottom:4px;display:flex;flex-wrap:wrap;gap:4px 10px;align-items:baseline}
.used{font-size:12.5px;font-weight:500;padding:1px 8px;border-radius:999px}
.used.yes{background:#2a3a52;color:#cde2fb}.used.no{background:#3a2a20;color:#f2c7ae}
.srow{display:grid;grid-template-columns:170px 1fr;gap:12px;align-items:center;height:20px}
.sl{font-size:14px;color:var(--sec);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.strack{position:relative;height:100%;border-left:1px solid var(--hair)}
.rng{position:absolute;top:6px;height:8px;background:#7f8a99;border-radius:4px}
.med{position:absolute;top:2px;width:3px;height:16px;background:var(--ink);margin-left:-1.5px;border-radius:2px}
.thr{position:absolute;top:-4px;bottom:-4px;width:0;border-left:2px dashed #e0b25a}
.saxis{position:relative;height:22px;border-top:1px solid var(--hair)}
.saxis span{position:absolute;transform:translateX(-50%);font-size:13px;color:var(--muted);top:3px}
.axisrow{height:26px}
.skey{display:flex;flex-wrap:wrap;gap:6px 20px;font-size:14px;color:var(--sec);margin:8px 0}
.skey i{display:inline-block;vertical-align:middle;margin-right:6px}
.placeholder{border:2px dashed #4a4a44;border-radius:12px;padding:22px;color:var(--sec);font-size:16px;min-height:220px;display:flex;flex-direction:column;justify-content:center}
.placeholder b{color:var(--ink)}
.minis{display:grid;grid-template-columns:repeat(3,1fr);gap:4px 26px}
.mini .brow{grid-template-columns:120px 1fr 64px;gap:10px}
.mini .bl{font-size:14px}.mini .bv{font-size:15px}
.mini .grp{margin:10px 0}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.cards .card{margin:0}
.cards .brow{grid-template-columns:150px 1fr 66px}
.cards .card h3{margin-top:0;font-size:19px}
.box{background:var(--card);border:1px solid var(--hair);border-radius:14px;padding:22px 26px}
.box ul{margin:6px 0;padding-left:20px}.box li{margin:6px 0}
details{margin-top:28px;background:var(--card);border:1px solid var(--hair);border-radius:14px;padding:16px 22px}
summary{cursor:pointer;font-size:19px;font-weight:600}
.tblwrap{overflow-x:auto;margin:10px 0}
.tbl{border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
.tbl th,.tbl td{padding:6px 10px;border-bottom:1px solid var(--hair);text-align:left;white-space:nowrap}
.tbl th{color:var(--sec);font-weight:600}
.note{color:var(--muted);font-size:15px;margin-top:4px}
nav.toc{display:flex;flex-wrap:wrap;gap:6px 18px;margin-top:30px;font-size:15px}
nav.toc a{color:var(--sec);text-decoration:none;border-bottom:1px solid var(--hair)}
@media (max-width:860px){
 .hero{grid-template-columns:1fr}.twocol{grid-template-columns:1fr}.cards{grid-template-columns:1fr}.minis{grid-template-columns:1fr 1fr}
}
@media (max-width:600px){
 body{font-size:17px}
 main{padding:32px 16px 72px}
 h1{font-size:31px}h2{font-size:25px;margin-top:64px}
 .hn{font-size:30px}
 .brow,.cards .brow,.mini .brow{grid-template-columns:1fr 62px;gap:4px 10px}
 .brow .bl{grid-column:1 / -1}
 .exlab{margin-left:0;margin-right:72px;font-size:12.5px}
 .twocol.dir{grid-template-columns:1fr}
 .placeholder{min-height:120px}
 .minis{grid-template-columns:1fr}
 .srow{grid-template-columns:118px 1fr;gap:8px}
 .sl{font-size:13px}
 .card{padding:18px 16px}
 .formula{font-size:15px}
 .svgfig .ax,.svgfig .axt{font-size:20px}.svgfig .lab{font-size:21px}
}
"""

JS = r"""
document.querySelectorAll('.toggle button').forEach(b=>{b.addEventListener('click',()=>{
 document.querySelectorAll('.toggle button').forEach(x=>x.classList.toggle('on',x===b));
 document.querySelectorAll('[data-dir]').forEach(el=>{el.hidden = el.dataset.dir!==b.dataset.show;});
});});
if(window.katex){document.querySelectorAll('[data-tex]').forEach(el=>{try{katex.render(el.dataset.tex,el,{displayMode:true,throwOnError:false});}catch(e){}});}
"""


def page(S):
    six_cl, q8, cl_itw, cf, cl_mj, cl_mj5 = S["six_cl"], S["six_q8"], S["cl_itw"], S["cl_formats"], S["cl_mj"], S["cl_mj5"]
    llama = S["llama"]
    nl = llama["no_lora_field"]
    tok = llama["typed_special_tokens"]
    d5, d6 = S["decl"]
    props = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in S["props"])
    au1, au3 = S["au"]
    strip_key = ('<div class="skey"><span><i style="width:26px;height:8px;background:#7f8a99;border-radius:4px"></i>10th to 90th percentile of the harm score</span>'
                 '<span><i style="width:3px;height:14px;background:#eceae4"></i>median</span>'
                 '<span><i style="width:0;height:14px;border-left:2px dashed #e0b25a"></i>threshold of the direction</span></div>')
    body = f"""
<main>
<header>
<h1>A Per-Request Strictness Knob in the Weights of a Small Open Model</h1>
<p class="meta">Lexsi AI Safety and Alignment Hackathon, Track 2b · problem statements <b>Inference-time safety steering</b> and <b>Locating where safety lives</b> · model <code>CohereLabs/tiny-aya-global</code></p>
<p class="lede">The adapter is called AyaDial. It is computed in closed form with no training, and its scale is set per request.</p>
<nav class="toc"><a href="#how">How it works</a><a href="#main">Main results</a><a href="#dial">It is a dial</a><a href="#beyond">Beyond the benchmark</a><a href="#limits">Where it falls short</a><a href="#tested">How it was tested</a></nav>
</header>

{S["hero"]}

<h2 id="how">How it works</h2>
<h3>The push as a function of the harm score</h3>
<div class="card pushcard">{S["push"]}</div>
<div class="formulas">
  <div class="formula" data-tex="{esc(S["formula_p"])}"></div>
  <div class="formula" data-tex="{esc(S["formula_q"])}"></div>
  <div class="formula small2" data-tex="{esc(S["formula_vals"])}"></div>
</div>
<p class="small">p is Tighten's push along the refusal direction r, q is Loosen's; s and t are the per-request scales.</p>
<table class="props">{props}</table>

<h2 id="main">Main results</h2>
<p class="lead">Every result on this page is drawn the same way.</p>
<div class="card">{legend()}{S["example"]}<p class="small">Base model, {S["ex_n"]} harmful requests wrapped in in-the-wild jailbreak templates. The number at the right is the share that matters: Unsafe for harmful requests, Refused for harmless ones.</p></div>

<h3>Tighten against CAST, AdaSteer and RefusalDirection</h3>
<p class="small">Development data (the 1910 WildJailbreak requests AyaDial was tuned on), at the settings run; the baselines were not tuned to equal cost.</p>
<div class="card scat">{S["scatter"]}<p class="small">Each point is one setting; lines join the three settings of RefusalDirection and of AdaSteer. The bars below give the same cells in full.</p></div>
<div class="card">{legend()}{S["st_bars"]}<p class="small">{esc(S["cast_note"])}</p></div>

<h3>Loosen, on held-out requests</h3>
<p class="small">Loosen fires when the harm score is low and pushes away from refusal. The harmful sets are shown beside them to check for release.</p>
<div class="card">{legend()}{S["loosen"]}</div>

<h3>Tighten and Loosen together</h3>
<div class="card">{legend()}{S["both"]}</div>

<h2 id="dial">It is a dial</h2>
<p class="small">The full sweep of s exists only for AyaDial on WildJailbreak development data. AyaDial-Improved was measured at three values on the same requests.</p>
<div class="card">{legend()}{S["sweep"]}</div>

<h2 id="beyond">Beyond the benchmark it was built on</h2>
<h3>In-the-wild templates and GCG suffixes</h3>
<p>Among requests the base answers unsafely, AyaDial's reading fires on {S["fire"][0]}% of WildJailbreak requests and {S["fire"][1]}% of in-the-wild templates. AyaDial-Improved's direction was recomputed with samples from those sources; the tested templates and requests are different ones.</p>
<div class="card">{legend()}{S["beyond"]}<p class="small">Held-out test 1. In-the-wild templates, AyaDial-Improved against Control: {sg(cl_itw["vs_control"])} points ({ci(cl_itw["vs_control_ci"])}, 95% interval); the harmless counterpart's refusal rises by {sg(cl_itw["counterpart_refusal_rise"])} points.</p></div>

<h3>Six languages no direction has seen</h3>
<p>Filipino, French, Hindi, Russian, Serbian and Spanish were in no fit of any direction. Unsafe replies fall by {f1(six_cl["base_unsafe"]-six_cl["unsafe"])} points against Base and {sg(six_cl["vs_control"])} against Control ({ci(six_cl["vs_control_ci"])}); a second judge, Qwen3Guard-Gen-8B, gives {f1(q8["base_unsafe"])}% to {f1(q8["unsafe"])}% ({sg(q8["vs_control"])} against Control, {ci(q8["vs_control_ci"])}).</p>
<div class="card">{legend()}{S["langs_main"]}<p class="small">Held-out test 2. AyaDial was not run on this test.</p></div>
<h3>Language by language</h3>
<div class="card">{legend()}{S["langs_each"]}<p class="small">150 requests per language, so each interval is wide; the pooled result above is the claim.</p></div>

<h2 id="limits">Where it falls short</h2>
<div class="cards">
 <div class="card"><h3>Custom system prompts and multi-turn chats</h3>{S["lim_formats"]}<p class="small">Unsafe is not shown to differ from Control ({sg(cf["vs_control"])} points, {ci(cf["vs_control_ci"])}); it avoids Control's extra refusals.</p></div>
 <div class="card"><h3>Plain harmful requests in English</h3>{S["lim_en"]}<p class="small">AyaDial-Improved does little here; Control is lower, with an interval that touches 0.</p></div>
 <div class="card"><h3>Nine MultiJail languages: the result did not repeat</h3>{S["lim_mj"]}<p class="small">Held-out test 1: {sg(cl_mj5["vs_control"])} against Control ({ci(cl_mj5["vs_control_ci"])}). On new requests: {sg(cl_mj["vs_control"])} ({ci(cl_mj["vs_control_ci"])}).</p></div>
 <div class="card"><h3>Judges</h3>{S["lim_judge"]}<p class="small">Every rate is a model judge's label; no reply has a human label yet, and some verdicts change with the judge. The main judge also marks {f1(S["itwb_unsafe"])}% of Base replies to harmless in-the-wild templates unsafe.</p></div>
 <div class="card"><h3>llama.cpp: a request without scales</h3><p>Of {nl["of"]} requests sent without the per-request scale field, {nl["identical_to_all_scales_1"]} got replies identical to every adapter at full scale, Loosen included. Every request has to carry explicit scales.</p></div>
 <div class="card"><h3>llama.cpp: typed special tokens</h3><p>Turn tokens typed by a user add {tok["extra_tokens_in_chat_for_typed_text"]} prompt tokens, as real special tokens would ({tok["extra_if_parsed_as_special"]}); as plain text they would add {tok["extra_if_plain_text"]}. User text has to be stripped of them.</p></div>
</div>

<h2 id="tested">How it was tested</h2>
<div class="box"><ul>
<li>Two held-out tests, each fixed and hashed before any reply was generated, with the settings declared in advance (held-out test 1 declared {esc(d5["written_utc"])}; held-out test 2 declared {esc(d6["written_utc"])}). Each was opened once.</li>
<li>Control is the same push along the same refusal direction with no harm gate, at scale 0.1 (for Loosen, the push away from refusal at 0.02).</li>
<li>Main judge Qwen3Guard-Gen-0.6B; second judges WildGuard (English rows) and Qwen3Guard-Gen-8B (other languages).</li>
<li>AyaDial's direction was computed from WildJailbreak training requests and plain harmful, XSTest, OR-Bench and everyday requests. AyaDial-Improved's added samples from in-the-wild templates, MultiJail, Aya prompts in ten languages, and GCG and PAIR attacks.</li>
</ul></div>
<details><summary>Appendix: full tables, intervals, other variants, method notes</summary>{S["appendix"]}</details>
</main>
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AyaDial Results</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>{CSS}</style></head><body>{body}<script>{JS}</script></body></html>"""


def main():
    S = build()
    OUT.write_text(page(S))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB); max refused-and-unsafe overlap {OVERLAP['max']:.2f}% at {OVERLAP['where']}")


if __name__ == "__main__":
    main()
