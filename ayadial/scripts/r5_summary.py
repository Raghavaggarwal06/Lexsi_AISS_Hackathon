"""Write results/round5/summary_numbers.md: every table and number of the round 5 summary, generated from the JSON result files.
The prose of summary.md is written around these blocks; no number in summary.md is typed by hand (charter section 5).
  python scripts/r5_summary.py            (runtime or Mac; pure Python + numpy)
Each block is skipped with a visible note if its source file does not exist yet."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/round5"
L = []


def J(rel):
    p = OUT / rel
    return json.loads(p.read_text()) if p.exists() else None


def h(t):
    L.extend(["", t, ""])


def miss(rel):
    L.append(f"_(not available: `{rel}`)_")


def f1(x):
    return "" if x is None else f"{x:.1f}"


def ci(v, c):
    return "" if v is None else f"{v:+.1f} ({c[0]:+.1f} to {c[1]:+.1f})"


def table_block(name, cols=None, rows=None, title=None):
    """One fixed-format table from tables/<name>.json: refusal % / unsafe % per column."""
    T = J(f"tables/{name}.json")
    if T is None:
        return miss(f"tables/{name}.json")
    cols = [c for c in (cols or T["columns"]) if c in T["columns"]]
    if title:
        L.append(f"**{title}**")
        L.append("")
    L.append("| set | n | " + " | ".join(cols) + " |")
    L.append("| --- | --- |" + " --- |" * len(cols))
    for r in T["rows"]:
        if rows and not any(r["row"].startswith(x) for x in rows):
            continue
        n = max(v["n"] for v in r["cells"].values())
        L.append(f"| {r['row']} | {n} | " + " | ".join(f"{r['cells'][c]['refusal']:.1f} / {r['cells'][c]['unsafe']:.1f}" if c in r["cells"] else "" for c in cols) + " |")
    L.append("")
    L.append(f"Cells are refusal % / unsafe % ({T['judge']}). Source: `tables/{name}.json`.")
    L.append("")


def change_block(name, cell, kinds=("H",), rows=None, title=None, measure="unsafe"):
    """Paired changes of one cell against base and against the table's control, with cluster-bootstrap 95% intervals."""
    T = J(f"tables/{name}.json")
    if T is None:
        return miss(f"tables/{name}.json")
    if title:
        L.append(f"**{title}**")
        L.append("")
    L.append(f"| set | n | base {measure} | {cell} {measure} | change vs base | change vs {T['control'] or 'control'} |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for r in T["rows"]:
        if r["kind"] not in kinds or cell not in r["cells"] or (rows and not any(r["row"].startswith(x) for x in rows)):
            continue
        c, b = r["cells"][cell], r["cells"]["base"]
        vb, vc = c.get("vs_base"), c.get("vs_control")
        L.append(f"| {r['row']} | {c['n']} | {b[measure]:.1f} | {c[measure]:.1f} | {ci(vb[measure], vb[measure + '_ci']) if vb else ''} | {ci(vc[measure], vc[measure + '_ci']) if vc else ''} |")
    L.append("")
    L.append(f"Changes are in points, with 95% intervals from a bootstrap over clusters. Source: `tables/{name}.json`.")
    L.append("")


def claims_block(name, title=None):
    C = J(f"tables/{name}.json")
    if C is None:
        return miss(f"tables/{name}.json")
    if title:
        L.append(f"**{title}**")
        L.append("")
    L.append("| harmful set | n | unsafe base to cell | vs base | vs control | harmless counterpart: refusal rise | verdict |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    for e in C["rows"]:
        L.append(f"| {e['row']} | {e['n']} | {e['base_unsafe']:.1f} to {e['unsafe']:.1f} | {ci(e['vs_base'], e['vs_base_ci'])} | {ci(e['vs_control'], e['vs_control_ci']) if e['vs_control'] is not None else ''} | "
                 f"{(e['counterpart'] + ': ' + ci(e['counterpart_refusal_rise'], e['counterpart_refusal_rise_ci'])) if e['counterpart_refusal_rise'] is not None else ''} | {e['verdict']} |")
    L.append("")
    L.append(f"Rule ({C['cell']} against {C['control']}): {C['rule']}. Source: `tables/{name}.json`.")
    L.append("")


def read_block(vecs=("v1", "v2"), frozen=False, title=None):
    R = {v: J(f"read/{v}{'_frozen' if frozen else ''}.json") for v in vecs}
    R = {v: r for v, r in R.items() if r}
    if not R:
        return miss("read/*.json")
    if title:
        L.append(f"**{title}**")
        L.append("")
    names = list(next(iter(R.values()))["pairs"])
    L.append("| pair (harmful vs harmless) | n | " + " | ".join(f"{v} AUROC | {v} among base-answered | {v} fires on harmful / harmless" for v in R) + " |")
    L.append("| --- | --- |" + " --- | --- | --- |" * len(R))
    for n in names:
        e0 = next(iter(R.values()))["pairs"][n]
        cells = []
        for v, r in R.items():
            e = r["pairs"].get(n)
            if not e:
                cells += ["", "", ""]
                continue
            aa = "" if e["auroc_answered"] is None else f"{e['auroc_answered']:.3f} ({e['auroc_answered_ci95'][0]:.3f} to {e['auroc_answered_ci95'][1]:.3f}; {e['n_h_answered']}/{e['n_b_answered']})"
            cells += [f"{e['auroc']:.3f} ({e['auroc_ci95'][0]:.3f} to {e['auroc_ci95'][1]:.3f})", aa, f"{e['fires_h_s1'] * 100:.0f}% / {e['fires_b_s1'] * 100:.0f}%"]
        L.append(f"| {n} | {e0['n_h']}/{e0['n_b']} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("z = w.x at the reply-start token; 'fires' = z above theta0 (the unit starts to push at s = 1). Source: `read/<version>.json`.")
    L.append("")


import re as _re


def _get(obj, path):
    for k in path.split("/"):
        if isinstance(obj, list):
            obj = obj[int(k)]
        else:
            obj = obj[k]
    return obj


def _fmt(v, fmt):
    if fmt == "ci":
        return f"{v[0]:.1f} to {v[1]:.1f}"
    if fmt == "sci":
        return f"{v[0]:+.1f} to {v[1]:+.1f}"
    if fmt == "pct":
        return f"{v * 100:.1f}"
    if fmt == "str":
        return str(v)
    return format(v, fmt)


MAIN_ROWS = [("adversarial_harmful (fresh", "fresh WildJailbreak"), ("in-the-wild template + harmful", "in-the-wild templates"), ("MultiJail English", "MultiJail English"), ("MultiJail 9 other", "MultiJail in 9 other languages"),
             ("MultiJail high-resource", "MultiJail zh/it/vi"), ("MultiJail medium-resource", "MultiJail ar/ko/th"), ("MultiJail low-resource", "MultiJail bn/sw/jv"), ("JailbreakBench GCG", "GCG"), ("JailbreakBench PAIR", "PAIR"), ("JailbreakBench DSN", "DSN"),
             ("JailbreakBench JBC", "JBC"), ("JailbreakBench RS", "random search"), ("harmful request + transplanted", "harmful request + transplanted suffix"), ("SORRY-Bench base", "SORRY-Bench base questions"), ("SORRY-Bench other", "SORRY-Bench other styles")]
PRETTY = {"SORRY-Bench translateta": "SORRY-Bench, Tamil translation", "SORRY-Bench translateml": "SORRY-Bench, Malayalam translation", "SORRY-Bench translatemr": "SORRY-Bench, Marathi translation", "SORRY-Bench translatefr": "SORRY-Bench, French translation",
          "SORRY-Bench translatezhcn": "SORRY-Bench, Chinese translation", "SORRY-Bench uncommondialects": "SORRY-Bench, uncommon dialects", "SORRY-Bench authorityendorsement": "SORRY-Bench, authority endorsement",
          "SORRY-Bench expertendorsement": "SORRY-Bench, expert endorsement", "SORRY-Bench misrepresentation": "SORRY-Bench, misrepresentation", "SORRY-Bench roleplay": "SORRY-Bench, role play", "SORRY-Bench technicalterms": "SORRY-Bench, technical terms",
          "MultiJail jv": "MultiJail, Javanese", "MultiJail vi": "MultiJail, Vietnamese", "MultiJail zh": "MultiJail, Chinese", "MultiJail bn": "MultiJail, Bengali", "MultiJail sw": "MultiJail, Swahili"}


def fill(text):
    """Resolve the number tokens of a prose template from the result files. Unresolvable tokens stay visible as [MISSING: ...].
    {{J:rel/path.json|key/sub/0|fmt}}                       a value in a JSON file under results/round5 (keys separated by '/')
    {{T:table|row label prefix|cell|field/sub|fmt}}         a value of one cell of one row of tables/<table>.json
    fmt: a Python format such as .1f or +.1f, or ci (a to b), sci (signed a to b), pct (fraction as percent), str."""
    cache = {}
    def load(rel):
        if rel not in cache:
            cache[rel] = J(rel)
        return cache[rel]
    def one(m):
        tok = m.group(1)
        try:
            parts = tok.split("|")
            if parts[0].startswith("J:"):
                return _fmt(_get(load(parts[0][2:]), parts[1]), parts[2] if len(parts) > 2 else ".1f")
            if parts[0].startswith("C:"):  # {{C:claims file|row label prefix|field/sub|fmt}}: one row of a claims file
                Cm = load(f"tables/{parts[0][2:]}.json")
                row = next(r for r in Cm["rows"] if r["row"].startswith(parts[1]))
                return _fmt(_get(row, parts[2]), parts[3] if len(parts) > 3 else ".1f")
            if parts[0].startswith(("VN:", "VL:", "VX:")):  # verdict words over the main harmful rows of a claims file: count / names that hold / names that do not, each with its verdict
                kind, Cm = parts[0][:2], load(f"tables/{parts[0][3:]}.json")
                got = [(nm, next((r for r in Cm["rows"] if r["row"].startswith(pre)), None)) for pre, nm in MAIN_ROWS]
                got = [(nm, r) for nm, r in got if r is not None]
                want = parts[1] if len(parts) > 1 else "holds"
                if kind == "VN":
                    return f"{sum(r['verdict'] == want for _, r in got)} of {len(got)}"
                if kind == "VL":
                    return ", ".join(nm for nm, r in got if r["verdict"] == want)
                return "; ".join(f"{nm} ({r['verdict']})" for nm, r in got if r["verdict"] != want)
            if parts[0].startswith("CAP:"):  # {{CAP:capability file|cell|change or fires or letters}}: the range of one quantity over the languages of a capability file
                Cp = load(f"capability/{parts[0][4:]}.json")
                e = Cp["cells"][parts[1]]
                key = {"change": "change_vs_base", "fires": "fires_abs_gt_half", "letters": "letter_changed"}[parts[2]]
                v = [e[l][key] for l in Cp["langs"]]
                return f"{min(v):+.1f} to {max(v):+.1f}" if parts[2] == "change" else f"{min(v):.1f}% to {max(v):.1f}%" if parts[2] == "letters" else f"{min(v):.0f}% to {max(v):.0f}%"
            if parts[0].startswith("CAPSIG:"):  # languages in which a cell's accuracy change has a 95% interval that excludes 0, with the change; "none" if there is none
                Cp = load(f"capability/{parts[0][7:]}.json")
                e = Cp["cells"][parts[1]]
                sig = [f"{l.split('_')[0]} {e[l]['change_vs_base']:+.1f} ({e[l]['change_ci95'][0]:+.1f} to {e[l]['change_ci95'][1]:+.1f})" for l in Cp["langs"] if e[l]["change_ci95"][0] > 0 or e[l]["change_ci95"][1] < 0]
                return "; ".join(sig) if sig else "none"
            if parts[0].startswith("CHG:"):  # {{CHG:cell|changed_pct or refusal_label_changed_pct}}: range over the harmless sets of capability/replies_changed.json
                Ch = load("capability/replies_changed.json")
                v = [g_[parts[0][4:]][parts[1]] for g_ in Ch["groups"].values() if parts[0][4:] in g_ and g_[parts[0][4:]].get(parts[1]) is not None]
                return f"{min(v):.0f}% to {max(v):.0f}%" if parts[1] == "changed_pct" else f"{min(v):.1f}% to {max(v):.1f}%"
            if parts[0].startswith("MAXB:"):  # {{MAXB:table|cell|excluded row prefix}}: largest refusal rise against base over the harmless rows (kind B), leaving one row out
                T = load(f"tables/{parts[0][5:]}.json")
                ex = parts[2] if len(parts) > 2 and parts[2] else None
                return f"{max(r['cells'][parts[1]]['vs_base']['refusal'] for r in T['rows'] if r['kind'] == 'B' and parts[1] in r['cells'] and not (ex and r['row'].startswith(ex))):+.1f}"
            if parts[0].startswith(("RISE:", "RISEN:")):  # {{RISE:table|cell|threshold}}: harmful rows whose unsafe rate rises by more than the threshold against base, largest first
                kind, T = parts[0].split(":")[0], load(f"tables/{parts[0].split(':', 1)[1]}.json")
                rs = sorted(((r["cells"][parts[1]]["vs_base"]["unsafe"], r) for r in T["rows"] if r["kind"] == "H" and parts[1] in r["cells"] and r["cells"][parts[1]]["vs_base"]["unsafe"] > float(parts[2])), key=lambda x: -x[0])
                if kind == "RISEN":
                    return str(len(rs))
                if not rs:
                    return "none"
                return "; ".join(f"{PRETTY.get(r['row'], r['row'])} {d:+.1f} (n = {r['cells'][parts[1]]['n']}, {r['cells'][parts[1]]['vs_base']['unsafe_ci'][0]:+.1f} to {r['cells'][parts[1]]['vs_base']['unsafe_ci'][1]:+.1f}"
                                 + (f"; control {r['cells'][T['control']]['vs_base']['unsafe']:+.1f}" if T.get("control") in r["cells"] else "") + ")" for d, r in rs)
            if parts[0].startswith("T:"):
                T = load(f"tables/{parts[0][2:]}.json")
                row = next(r for r in T["rows"] if r["row"].startswith(parts[1]))
                return _fmt(_get(row["cells"][parts[2]], parts[3]), parts[4] if len(parts) > 4 else ".1f")
        except Exception as e:
            return f"[MISSING: {tok} ({type(e).__name__})]"
        return f"[MISSING: {tok}]"
    return _re.sub(r"\{\{(.+?)\}\}", one, text)


def write_summary():
    """summary.md = the prose template (results/round5/summary_prose.md, tokens resolved) followed by the number blocks."""
    tpl = OUT / "summary_prose.md"
    if not tpl.exists():
        return
    body = fill(tpl.read_text())
    nums = (OUT / "summary_numbers.md").read_text().split("\n", 1)[1]
    (OUT / "summary.md").write_text(body.rstrip() + "\n\n---\n\n# Appendix: every table, written by script\n" + nums)
    missing = _re.findall(r"\[MISSING: .+?\]", body)
    print(f"summary.md written; unresolved tokens: {len(missing)}")
    for x in missing[:20]:
        print("  ", x)


def main():
    L.append("# Round 5: numbers written by script")
    L.append("")
    L.append("Generated by `scripts/r5_summary.py` from the JSON files under `results/round5/`. Do not edit by hand.")
    m = J("sets_manifest.json")
    h("## Sets")
    if m:
        nd = m.get("near_duplicate_drop", {})
        L.append(f"- Round 4 rows carried over: {m['round4_rows_carried']}. New rows: {m['new_rows']} (development {m['sides']['development']}, frozen {m['sides']['frozen']}, fit-only {m['sides']['fit_only']}).")
        L.append(f"- Frozen ids sha256: `{m['frozen_ids_sha256']}` ({m['frozen_n']} rows). Before the near-duplicate drop: `{nd.get('frozen_hash_before', '')}` ({nd.get('frozen_n_before', '')} rows; {nd.get('n_dropped', 0)} dropped).")
        ex = m["exact_text_clashes"]
        L.append(f"- Exact-text clashes: fit vs test {ex['test_rows_equal_to_a_fit_row']}, development vs frozen {ex['frozen_rows_equal_to_a_development_row']}, new frozen vs round 4 frozen {ex['frozen_rows_equal_to_a_round4_frozen_row']}.")
    v = J("v1_check.json")
    h("## v1 rebuilt from the adapter files")
    if v:
        L.append(f"- All frozen files match `v1_freeze.json`: {all(v['files_match_freeze'].values())}. norm(w) {v['norm_w']:.6f} (fit.json {v['norm_w_in_fit_json']:.6f}); second gate row vs formula, max difference {v['second_gate_row_vs_formula_max_abs_diff']:.1e}; "
                 f"down columns vs r/5, cosine {v['down_first_vs_r_over_5']['cos']:.6f}; formula path vs shipped rows after rounding to bf16, max difference {v['formula_vs_shipped_rows_max_abs_diff_bf16']:.1e}. Pass: {v['pass']}.")
    else:
        miss("v1_check.json")
    r = J("reuse_check.json")
    h("## Reuse of round 4 development cells")
    if r:
        for su, e in r["suites"].items():
            L.append(f"- `{su}` base re-generated tonight, n = {e['n']}: identical replies {e['identical_responses'] * 100:.1f}%; label agreement refusal {e['label_agreement']['refusal'] * 100:.1f}%, unsafe {e['label_agreement']['unsafe'] * 100:.1f}%; "
                     f"rates last night {e['rates_last_night'][0]:.1f} / {e['rates_last_night'][1]:.1f}, tonight {e['rates_tonight'][0]:.1f} / {e['rates_tonight'][1]:.1f}. Pass: {e['pass']}.")
        e = r["v1_from_files_vs_last_night_foldcap@1"]
        L.append(f"- v1 from the adapter files at s = 1 against last night's foldcap@1, n = {e['n']}: identical replies {e['identical_responses'] * 100:.1f}%; label agreement refusal {e['label_agreement']['refusal'] * 100:.1f}%, unsafe {e['label_agreement']['unsafe'] * 100:.1f}%; "
                 f"fire value max difference {e['fire_max_abs_diff']:.3f}; rates last night {e['rates_last_night'][0]:.1f} / {e['rates_last_night'][1]:.1f}, tonight {e['rates_tonight'][0]:.1f} / {e['rates_tonight'][1]:.1f}.")
        L.append(f"- Reused: {r['reuse']}." + (f" Labels changed by tonight's degeneracy rule on the reused cells: {r['labels_changed_vs_round4_files']['total']}." if "labels_changed_vs_round4_files" in r else ""))
    else:
        miss("reuse_check.json")
    for sec in SECTIONS:
        sec()
    (OUT / "summary_numbers.md").write_text("\n".join(L) + "\n")
    if "--quiet" not in sys.argv:
        print("\n".join(L))
    write_summary()


def sec_lab():
    h("## Read lab: refits scored on cached activations (no generation)")
    S = J("lab/selection.json")
    if S is None:
        return miss("lab/selection.json")
    order = [c for c in ("v1", "v1again", "v2", "v3itw", "v3mj", "v3", "v3w", "v2g", "v3g") if c in S["candidates"]]
    L.append("| candidate | mean share of base-unsafe prompts pushed, six held-out groups | same on round 4 adversarial_harmful | largest rise in harmless prompts pushed vs v1 | passes the lab rule |")
    L.append("| --- | --- | --- | --- | --- |")
    for c in order:
        v = S["candidates"][c]
        L.append(f"| {c} | {v['mean_pushed_base_unsafe_held_out']:.1f} | {v['round4_pushed_base_unsafe']:.1f} | +{v['largest_rise_in_harmless_pushed']:.1f} ({v['on_group']}) | {'yes' if v['passes'] else 'no'} |")
    L.append("")
    L.append(f"Rule: {S['rule']}. Source: `lab/selection.json`, `lab/<candidate>.json`.")
    cols = [c for c in ("v1", "v2", "v3", "v3g") if J(f"lab/{c}.json")]
    D = {c: J(f"lab/{c}.json")["groups"] for c in cols}
    L.append("")
    L.append("| group | n base-unsafe | " + " | ".join(f"{c}: pushed among base-unsafe / all harmful / harmless" for c in cols) + " |")
    L.append("| --- | --- |" + " --- |" * len(cols))
    for g in D[cols[0]]:
        f = lambda v: "-" if v is None else f"{v:.1f}"
        L.append(f"| {g} | {D[cols[0]][g]['n_base_unsafe']} | " + " | ".join(f"{f(D[c][g]['pushed_base_unsafe'])} / {f(D[c][g]['pushed_harmful'])} / {f(D[c][g]['pushed_harmless'])}" for c in cols) + " |")
    L.append("")
    L.append("Shares in %, at s = 1, with the real unit formulas (guard included). 'Pushed' = more than 0.5 r at the reply-start token.")


def sec_guard():
    h("## Position guard")
    G = J("guard/g2.json")
    if G is None:
        return miss("guard/g2.json")
    new, old = G["new_guard_on_held_out_positions"], G["v1_guard_on_the_same_positions"]
    L.append(f"Held-out positions: {G['n_eval_positions']} from {G['n_eval_prompts']} prompts. Margins (marker against every other position): v1's guard m {old['margin_m']:+.3f}, q {old['margin_q']:+.3f}; refit guard m {new['margin_m']:+.3f}, q {new['margin_q']:+.3f}.")
    L.append("")
    L.append("| prompt family | n markers | v1 guard: m, q, threshold shift in z | refit guard: m, q, threshold shift in z |")
    L.append("| --- | --- | --- | --- |")
    for fam, o in old["by_family_at_marker"].items():
        n_ = new["by_family_at_marker"][fam]
        L.append(f"| {fam} | {o['n']} | {o['m_mean']:.3f}, {o['q_mean']:+.3f}, {o['threshold_shift_in_z']:+.3f} | {n_['m_mean']:.3f}, {n_['q_mean']:+.3f}, {n_['threshold_shift_in_z']:+.3f} |")
    L.append("")
    L.append("Source: `guard/g2.json`.")
    rows = [(v, J(f"guard/writes_{v}.json")) for v in ("v1", "v2g", "v3", "v3g")]
    rows = [(v, w) for v, w in rows if w]
    if rows:
        L.append("")
        L.append("| version | largest write away from the marker at s = 1 / 1.25 / 2 / 3 | share of positions above 0.05 r at s = 2 / 3 | lax unit at t = 1: largest write away from the marker |")
        L.append("| --- | --- | --- | --- |")
        for v, w in rows:
            st = w["strict"]
            L.append(f"| {v} | {st['s=1']['elsewhere_max']:.3f} / {st['s=1.25']['elsewhere_max']:.3f} / {st['s=2']['elsewhere_max']:.3f} / {st['s=3']['elsewhere_max']:.3f} | {st['s=2']['elsewhere_share_above_0.05'] * 100:.3f}% / {st['s=3']['elsewhere_share_above_0.05'] * 100:.3f}% | {w['lax']['t=1']['elsewhere_max']:.3f} |")
        L.append("")
        L.append("Writes in units of r, on held-out positions. Source: `guard/writes_<version>.json`.")


def sec_dev():
    h("## Development: v1 unchanged on data its read never saw")
    table_block("dev_v1", title="Development slices and round 4 development sets")
    claims_block("dev_v1_claims_foldcap@1", title="v1 at s = 1 against the control at 0.1")
    claims_block("dev_v1_claims_foldcap@1.25", title="v1 at s = 1.25 against the control at 0.1")
    read_block(("v1", "v2", "v3g"), title="The read on each benchmark")
    D = J("diag/read_or_push_foldcap@1.json")
    if D:
        L.append("**Read or push? v1 at s = 1**")
        L.append("")
        L.append("| harmful set | n | unit fires | base-unsafe prompts | fires among them | still unsafe when it fired | still unsafe when it did not |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        f = lambda v: "-" if v is None else f"{v:.1f}%"
        for g, v in D["groups"].items():
            L.append(f"| {g} | {v['n']} | {v['fires']:.1f}% | {v['base_unsafe_n']} | {f(v['fires_among_base_unsafe'])} | {f(v['still_unsafe_among_base_unsafe_fired'])} (n={v['n_base_unsafe_fired']}) | {f(v['still_unsafe_among_base_unsafe_missed'])} (n={v['n_base_unsafe_missed']}) |")
        L.append("")
        L.append("Source: `diag/read_or_push_foldcap@1.json`.")


def sec_lax_exposure():
    h("## Loosening: where the lax unit could fire (before any generation)")
    for v in ("v1", "v3g"):
        E = J(f"lax/exposure_{v}.json")
        if E is None:
            miss(f"lax/exposure_{v}.json")
            continue
        L.append(f"**Read {v}**, threshold z < {E['thresholds']['h01']:.3f} (1st percentile of harmful fit prompts). Share of base-refused prompts below the threshold:")
        L.append("")
        hs = [(n, e) for n, e in sorted(E["sets"].items()) if e.get("harm") == 1 or n in ("harmful", "harmful_jb", "wj_adversarial_harmful", "dev_ho_h", "dev_in_h")]
        bs = [(n, e) for n, e in sorted(E["sets"].items()) if (n, e) not in hs]
        f = lambda e: "-" if e["share_of_refused_below_h01"] is None else f"{e['share_of_refused_below_h01'] * 100:.0f}% of {e['base_refused']}"
        L.append("- harmful sets: " + "; ".join(f"{n} {f(e)}" for n, e in hs))
        L.append("- harmless sets: " + "; ".join(f"{n} {f(e)}" for n, e in bs if e["base_refused"] >= 5))
        L.append("")


def board_block(name, title):
    Bd = J(f"tables/{name}.json")
    if Bd is None:
        return miss(f"tables/{name}.json")
    L.append(f"**{title}** ({Bd['method']} against {Bd['control']}; {Bd['n_harmful']} adversarial_harmful, {Bd['n_benign']} adversarial_benign development prompts)")
    L.append("")
    L.append("| budget (points of adversarial_benign refusal) | method: setting, unsafe, benign rise | control: setting, unsafe, benign rise | control minus method, 95% interval | verdict |")
    L.append("| --- | --- | --- | --- | --- |")
    for b_, e in Bd["boards"].items():
        m, c = e["method_pick"], e["control_pick"]
        L.append(f"| {b_} | {m['setting']}, {m['objective']:.1f}%, +{m['budget_rise'][0]:.1f} | {c['setting']}, {c['objective']:.1f}%, +{c['budget_rise'][0]:.1f} | {e['diff_vs_control']:.1f} ({e['ci95'][0]:.1f} to {e['ci95'][1]:.1f}) | {e['verdict']} |")
    L.append("")
    L.append(f"Base unsafe {next(iter(Bd['boards'].values()))['method_pick']['base_objective']:.1f}%. Selection is redone in every bootstrap resample. Source: `tables/{name}.json`.")
    L.append("")


def sec_v2v3():
    h("## Development: the two refits")
    board_block("board_v1", "v1 on the round 4 scoreboard")
    board_block("board_v2", "v2 on the round 4 scoreboard")
    board_block("board_v3g", "v3g on the round 4 scoreboard")
    P = J("picks.json")
    if P:
        L.append("**Settings fixed on development data** (`picks.json`; rule: " + P["rule_tightening"] + ")")
        L.append("")
        L.append("| version | grid | scoreboard pick at budget 5 (unsafe there) | setting by the list as first written | frozen setting | sets over 5 points at the settings tried |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for m, v in P["tightening"].items():
            over = "; ".join(f"s = {st['setting']:g}: " + (", ".join(f"{k} +{x}" for k, x in st["over_5_points"].items()) or "none") for st in v["steps"])
            L.append(f"| {m} | {', '.join(f'{g:g}' for g in v['grid'])} | {v['scoreboard_pick_budget5']} ({v['scoreboard_unsafe_at_pick']}%) | {v['setting_by_the_list_as_first_written']} | {v['frozen_setting']} | {over} |")
        L.append("")
        if P.get("loosening"):
            x = P["loosening"]
            L.append(f"Loosening ({P['rule_loosening']}): lax setting {x['lax_frozen_setting']}, refusal on the three harmless sets {x['lax_refusal_base_to_pick'][0]}% to {x['lax_refusal_base_to_pick'][1]}%, unsafe rises {x['lax_unsafe_rises']}; control setting {x['control_frozen_setting']} ({x['control_refusal_at_pick']}%, unsafe rises {x['control_unsafe_rises']}); control minus lax {x['diff_vs_control']} ({x['ci95'][0]} to {x['ci95'][1]}), {x['verdict']}.")
            L.append("")
    U = J("tables/cells_used.json") or {}
    table_block("dev_main", title="Fixed-format table, development (v3g saw the 'a' half of the in-the-wild, MultiJail and Aya rows; see the next table)")
    table_block("dev_main_b", title="Same, held-out half only for the in-the-wild, MultiJail and Aya rows")
    for c in ("foldcap@1", U.get("v2"), U.get("v3g")):
        if c:
            claims_block(f"dev_main_b_claims_{c}", title=f"{c} against the control at 0.1, development, held-out halves")


def sec_lax():
    h("## Loosening")
    Bd = J("tables/laxboard.json")
    if Bd is None:
        miss("tables/laxboard.json")
    else:
        L.append(f"**Mirror budget rule** ({Bd['method']} against {Bd['control']}; unsafe may rise by at most the budget on harmful (n={Bd['n_harmful_sets'][0]}), harmful_jb (n={Bd['n_harmful_sets'][1]}) and adversarial_harmful (n={Bd['n_harmful_sets'][2]}))")
        L.append("")
        L.append("| harmless set | budget | method: setting, refusal base to pick, unsafe rises | control: setting, refusal | control minus method, 95% interval | verdict |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for lab, bs in Bd["boards"].items():
            for b_, e in bs.items():
                m, c = e["method_pick"], e["control_pick"]
                L.append(f"| {lab} | {b_} | {m['setting']}, {m['base_objective']:.1f}% to {m['objective']:.1f}%, " + ", ".join(f"{x:+.1f}" for x in m["budget_rise"]) + f" | {c['setting']}, {c['objective']:.1f}% | {e['diff_vs_control']:.1f} ({e['ci95'][0]:.1f} to {e['ci95'][1]:.1f}) | {e['verdict']} |")
        L.append("")
        L.append("Source: `tables/laxboard.json`.")
        L.append("")
    table_block("dev_lax", title="Loosening, development, fixed-format table")
    table_block("dev_lax_b", title="Same, held-out half only for the in-the-wild, MultiJail and Aya rows")
    Pr = J("parity/units.json")
    if Pr:
        L.append("**Parity: float32 hook against weights** (mean next-token KL; bar 1e-3)")
        L.append("")
        L.append("| units | KL first token, hook vs weights | KL reply positions, hook vs weights | KL first token, base vs weights | pass |")
        L.append("| --- | --- | --- | --- | --- |")
        for n, e in Pr["cells"].items():
            L.append(f"| {n} | {e['hook_vs_weights']['kl_first_token']:.2e} | {e['hook_vs_weights']['kl_reply_positions']:.2e} | {e['base_vs_weights']['kl_first_token']:.3f} | {e['pass_kl_1e-3']} |")
        z = Pr["base_vs_four_units_zeroed"]
        L.append("")
        L.append(f"{Pr['n_prompts']} prompts. Base against the four dormant units zeroed (both adapters at scale 0): KL first token {z['kl_first_token']:.2e}, reply positions {z['kl_reply_positions']:.2e}, same first token on {z['argmax_agree_first'] * 100:.1f}% of prompts. Source: `parity/units.json`.")
        L.append("")
    R = J("lax/orbench_reply_review.json")
    if R:
        L.append(f"**The judge's blind side on orbench_hard** (`lax/orbench_reply_review.json`): {R['n_newly_answered']} replies went from refusal to answer under last night's loosening hook, {R['n_judged_unsafe']} of them labelled unsafe; read by the model that ran the round, {R['replies_with_operational_harmful_detail_in_visible_part']} give working detail in the 128 tokens generated. Categories: " + "; ".join(f"{c['category']} ({c['n']}, {c['n_judged_unsafe']} flagged)" for c in R["categories"]) + ".")
        L.append("")


def sec_frozen():
    h("## Frozen test (opened once)")
    D = J("frozen_declaration.json")
    if D is None:
        return miss("frozen_declaration.json")
    L.append(f"Declared {D['written_utc']} UTC, before any frozen prompt was generated: {D['frozen_n']} prompts, ids sha256 `{D['frozen_ids_sha256']}`. Rule: {D['rule']}")
    L.append("")
    L.append("| cell | role |")
    L.append("| --- | --- |")
    for c in D["cells"]:
        L.append(f"| {c['name']} | {c['role']} |")
    L.append("")
    for name, title in (("frz_tight", "Tightening, frozen, fixed-format table"), ("frz_lax", "Loosening and the combined setting, frozen"), ("frz_tight_wg", "Tightening, frozen, second judge (WildGuard)"), ("frz_lax_wg", "Loosening and the combined setting, frozen, second judge (WildGuard)"),
                        ("frz_both_wg", "The combined setting on the tightening rule, frozen, second judge (WildGuard)"), ("frz_tight_q8", "Tightening, frozen, rows in other languages, second judge (Qwen3Guard-Gen-8B)"),
                        ("frz_lax_q8", "Loosening, frozen, rows in other languages, second judge (Qwen3Guard-Gen-8B)")):
        if "_wg" in name or "_q8" in name:
            if J(f"tables/{name}.json") is None:
                continue
        table_block(name, title=title)
    U = J("tables/cells_used.json") or {}
    for c in ("foldcap@1", "foldcap@1.25", U.get("v2"), U.get("v3g")):
        if c:
            claims_block(f"frz_tight_claims_{c}", title=f"{c} against the control at 0.1, frozen")
            if J(f"tables/frz_tight_wg_claims_{c}.json"):
                claims_block(f"frz_tight_wg_claims_{c}", title=f"{c} against the control at 0.1, frozen, WildGuard")
            if J(f"tables/frz_tight_q8_claims_{c}.json"):
                claims_block(f"frz_tight_q8_claims_{c}", title=f"{c} against the control at 0.1, frozen rows in other languages, Qwen3Guard-Gen-8B")
    for c in (U.get("lax"), U.get("both")):
        if c and J("tables/frz_lax.json"):
            change_block("frz_lax", c, kinds=("B", "C"), title=f"{c}: refusal on harmless sets, frozen", measure="refusal")
            change_block("frz_lax", c, kinds=("H",), title=f"{c}: unsafe on harmful sets, frozen", measure="unsafe")
            for suf, jn in (("_wg", "WildGuard, English rows"), ("_q8", "Qwen3Guard-Gen-8B, rows in other languages")):
                Tj = J(f"tables/frz_lax{suf}.json")
                if Tj and c in Tj["columns"]:
                    change_block(f"frz_lax{suf}", c, kinds=("B", "C"), title=f"{c}: refusal on harmless sets, frozen, {jn}", measure="refusal")
                    change_block(f"frz_lax{suf}", c, kinds=("H",), title=f"{c}: unsafe on harmful sets, frozen, {jn}", measure="unsafe")
    if U.get("both"):
        for suf, jn in (("", "main judge"), ("_wg", "WildGuard, English rows")):
            if J(f"tables/frz_both{suf}_claims_{U['both']}.json"):
                claims_block(f"frz_both{suf}_claims_{U['both']}", title=f"{U['both']} (v3g strict with the lax pair) against the control at 0.1, frozen, {jn}")
        if J(f"tables/frz_tight_q8_claims_{U['both']}.json"):
            claims_block(f"frz_tight_q8_claims_{U['both']}", title=f"{U['both']} against the control at 0.1, frozen rows in other languages, Qwen3Guard-Gen-8B")


def sec_checks():
    h("## Robustness, capability, judges, export")
    if J("read/v1_frozen.json"):
        read_block(("v1", "v2", "v3g"), frozen=True, title="The read on the frozen benchmarks (computed after the frozen test; no setting depends on it)")
    for v in ("v1", "v3g"):
        R = J(f"robust/positions_{v}.json")
        if R is None:
            miss(f"robust/positions_{v}.json")
            continue
        L.append(f"**Where the units fire, by format ({v}; forward passes, 200 requests per format)**")
        L.append("")
        L.append("| format | m at the true marker: min / median | q there: median | strict s = 1 fires: harmful / harmless | lax fires: harmful / harmless | largest strict write elsewhere (s = 1 / 2) | other reply-start tokens: prompts, strict fires there, lax fires there |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        for f_, e in R["formats"].items():
            o = e.get("at_other_reply_start_tokens")
            L.append(f"| {f_} | {e['m_at_marker_min_p1_p50_p99_max'][0]:.2f} / {e['m_at_marker_min_p1_p50_p99_max'][2]:.2f} | {e['q_at_marker_min_p1_p50_p99_max'][2]:+.2f} | {e['strict_s1_fires_harmful'] * 100:.0f}% / {e['strict_s1_fires_harmless'] * 100:.0f}% | {e['lax_fires_harmful'] * 100:.0f}% / {e['lax_fires_harmless'] * 100:.0f}% | "
                     f"{e['strict_s1_write_elsewhere_max']:.2f} / {e['strict_s2_write_elsewhere_max']:.2f} | " + (f"{e['prompts_with_other_reply_start_tokens']}, {o['strict_s1_fires'] * 100:.0f}%, {o['lax_fires'] * 100:.0f}%" if o else "0") + " |")
        L.append("")
        L.append(f"Source: `robust/positions_{v}.json`.")
        L.append("")
    G = J("robust/robust_cells.json")
    if G:
        cells = [c for c in next(iter(G.values())) if c != "base"]
        L.append("**Behaviour by format (generation; harmful refusal / harmful unsafe / harmless refusal, %)**")
        L.append("")
        L.append("| format | base | " + " | ".join(cells) + " |")
        L.append("| --- | --- |" + " --- |" * len(cells))
        for f_, e in G.items():
            L.append(f"| {f_} | " + " | ".join(f"{e[c]['harmful_refusal']:.1f} / {e[c]['harmful_unsafe']:.1f} / {e[c]['harmless_refusal']:.1f}" for c in ["base"] + cells if c in e) + " |")
        L.append("")
        L.append("Source: `robust/robust_cells.json`.")
        L.append("")
    for name in ("cap_strict", "cap_all"):
        C = J(f"capability/{name}.json")
        if C is None:
            continue
        L.append(f"**Capability ({C['task']}; {C['n_per_language']} questions per language)**")
        L.append("")
        L.append("| cell | " + " | ".join(C["langs"]) + " |")
        L.append("| --- |" + " --- |" * len(C["langs"]))
        L.append("| base accuracy | " + " | ".join(f"{C['base'][l]['accuracy']:.1f}" for l in C["langs"]) + " |")
        for c, e in C["cells"].items():
            L.append(f"| {c}: change, 95% interval; letters changed; fires | " + " | ".join(f"{e[l]['change_vs_base']:+.1f} ({e[l]['change_ci95'][0]:+.1f} to {e[l]['change_ci95'][1]:+.1f}); {e[l]['letter_changed']:.1f}%; {e[l]['fires_abs_gt_half']:.0f}%" for l in C["langs"]) + " |")
        L.append("")
        L.append(f"Source: `capability/{name}.json`.")
        L.append("")
    Ch = J("capability/replies_changed.json")
    if Ch:
        cells = []
        for v in Ch["groups"].values():
            cells += [c for c in v if c not in cells]
        L.append("**How many harmless replies change at all** (share of replies that differ from the base reply by at least one character; same rows, same batching, 128 new tokens; in brackets the share whose refusal label changed)")
        L.append("")
        L.append("| harmless set | n | " + " | ".join(cells) + " |")
        L.append("| --- | --- |" + " --- |" * len(cells))
        for g_, v in Ch["groups"].items():
            L.append(f"| {g_} | {max(e['n'] for e in v.values())} | " + " | ".join((f"{v[c]['changed_pct']:.1f}%" + (f" ({v[c]['refusal_label_changed_pct']:.1f}%)" if v[c].get('refusal_label_changed_pct') is not None else "")) if c in v else "" for c in cells) + " |")
        L.append("")
        L.append("Source: `capability/replies_changed.json`.")
        L.append("")
    for name in sorted(p_.stem for p_ in (OUT / "judge2").glob("*.json")) if (OUT / "judge2").exists() else []:
        Jd = J(f"judge2/{name}.json")
        if "cells" not in Jd or "judge" not in Jd:
            continue
        L.append(f"**Second judge {Jd['judge']} against the main judge (`judge2/{name}.json`)**")
        L.append("")
        L.append("| cell | n | agreement on refusal | agreement on unsafe | refusal main / second | unsafe main / second |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for c, e in Jd["cells"].items():
            a_ = e["ALL"]
            L.append(f"| {c} | {a_['n']} | {a_['agree_refusal'] * 100:.1f}% | {a_['agree_unsafe'] * 100:.1f}% | {a_['refusal_main']:.1f} / {a_['refusal_second']:.1f} | {a_['unsafe_main']:.1f} / {a_['unsafe_second']:.1f} |")
        L.append("")
    Jl = J("judge2/by_language.json")
    for key, what in (("languages", "development replies"), ("languages_frozen", "frozen replies")):
        if not Jl or not Jl.get(key):
            continue
        L.append(f"**Main judge against Qwen3Guard-Gen-8B by language ({what}; cells pooled: {', '.join(Jl.get('cells_dev' if key == 'languages' else 'cells_frz', []))})**")
        L.append("")
        L.append("| language or style | n | agreement on refusal | agreement on unsafe | harmful requests: unsafe %, main / 8B | harmful requests: refusal %, main / 8B | harmless requests (Aya): refusal %, main / 8B |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        for lg, e in Jl[key].items():
            hm = e.get("multijail") or e.get("sorry") or e.get("in_the_wild")
            ay = e.get("aya")
            L.append(f"| {lg} | {e['n']} | {e['agree_refusal'] * 100:.1f}% | {e['agree_unsafe'] * 100:.1f}% | " + (f"{hm['unsafe_main']:.1f} / {hm['unsafe_second']:.1f} | {hm['refusal_main']:.1f} / {hm['refusal_second']:.1f}" if hm else " | ") + " | " + (f"{ay['refusal_main']:.1f} / {ay['refusal_second']:.1f}" if ay else "") + " |")
        L.append("")
    Gg = J("gguf/export_both_sides.json")
    if Gg:
        L.append(f"**GGUF, four adapters, per-request scales** (`gguf/export_both_sides.json`; lax threshold {Gg['theta']:.3f}, t = {Gg['t']:g})")
        L.append("")
        L.append("| run | harmful refusal | harmful unsafe | base-refused harmless: refusal | other harmless: refusal | agreement of refusal labels with HF (f16 / Q4_K_M) |")
        L.append("| --- | --- | --- | --- | --- | --- |")
        for n, e in Gg["rates"].items():
            ag = f"{e.get('refusal_label_agreement_with_f16', float('nan')):.1f} / {e.get('refusal_label_agreement_with_q4km', float('nan')):.1f}" if n.startswith("hfref5") else ""
            L.append(f"| {n} | {e['harmful_refusal']:.1f} | {e['harmful_unsafe']:.1f} | {e['base_refused_harmless_refusal']:.1f} | {e['other_harmless_refusal']:.1f} | {ag} |")
        L.append("")
    for name in ("screen", "screen8", "full"):
        St = J(f"safetune/{name}.json")
        if St is None:
            continue
        L.append(f"**SafeTune baselines, {name}** (SafeTune {St.get('safetune_version')}; `safetune/{name}.json`)")
        L.append("")
        L.append("| cell | adversarial_harmful refusal | adversarial_harmful unsafe | adversarial_benign refusal | degenerate |")
        L.append("| --- | --- | --- | --- | --- |")
        b_ = St.get("base", {})
        if b_:
            L.append(f"| base | {b_['harmful_refusal']:.1f} | {b_['harmful_unsafe']:.1f} | {b_['benign_refusal']:.1f} | |")
        for c, e in St["cells"].items():
            L.append(f"| {c} | {e['harmful_refusal']:.1f} | {e['harmful_unsafe']:.1f} | {e['benign_refusal']:.1f} | {e['degenerate']:.1f} |")
        if St.get("failures"):
            L.append("")
            L.append("Failures: " + "; ".join(f"{k}: {v[:120]}" for k, v in St["failures"].items()))
        L.append("")


SECTIONS = [sec_dev, sec_lab, sec_guard, sec_v2v3, sec_lax_exposure, sec_lax, sec_frozen, sec_checks]

if __name__ == "__main__":
    main()
