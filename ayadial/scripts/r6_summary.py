"""Round 6 summary: every table and number written by script from results/round6 (round 5's summary code with round 6's files).
  python scripts/r6_summary.py [--quiet]
Writes results/round6/summary_numbers.md (appendix) and, if results/round6/summary_prose.md exists, summary.md = that prose with its number tokens
resolved ({{J:...}}, {{T:...}}, {{C:...}}, {{VN:...}}, {{VL:...}}, {{VX:...}}, {{RISE:...}}, {{MAXB:...}}; see r5_summary.fill) followed by the appendix."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r5_summary as S5

ROOT = Path(__file__).resolve().parent.parent
OUT = S5.OUT = ROOT / "results/round6"
L, J, h, miss = S5.L, S5.J, S5.h, S5.miss
S5.MAIN_ROWS = [("adversarial_harmful (fresh", "fresh WildJailbreak"), ("in-the-wild template + harmful", "in-the-wild templates never used"), ("MultiJail English", "MultiJail English"),
                ("MultiJail 9 other", "MultiJail in 9 other languages"), ("MultiJail high-resource", "MultiJail zh/it/vi"), ("MultiJail medium-resource", "MultiJail ar/ko/th"), ("MultiJail low-resource", "MultiJail bn/sw/jv"),
                ("red-teaming prompts, 6 unseen", "red-teaming prompts in 6 unseen languages"), ("red-teaming prompts, Arabic and English", "red-teaming prompts in Arabic and English"), ("exam formats, all: harmful", "all exam formats pooled")]
FMT_DEV = [("default", "default template"), ("sys_bank", "one-line preamble"), ("sys_car", "car assistant preamble"), ("sys_persona", "persona preamble"), ("sys_de", "German preamble"), ("sys_long", "long preamble (house style)"),
           ("sys_law", "long preamble (law office)"), ("nosys", "no system turn"), ("mt_benign", "after two harmless turns"), ("mt_refused", "after a refused turn"), ("mt3", "after three harmless turns"), ("mt_sys", "preamble and two turns"),
           ("inj_tail", "typed tokens at the end"), ("inj_head", "typed fake exchange before")]
FMT_EXAM = [("xa_sysA", "short developer preamble"), ("xa_sysB", "long developer preamble"), ("xa_sysC", "persona preamble"), ("xa_sysD", "Portuguese preamble"), ("xa_mt2", "after two harmless turns"), ("xa_mtref", "after a refused turn"),
            ("xa_mtsys", "preamble and two turns"), ("xa_itwsys", "in-the-wild template as the developer preamble")]


def sec_sets():
    global L  # the functions extend the shared list in place with +=
    h("## Exam B and the declaration")
    M, F, D = J("sets_manifest.json"), J("exam_formats.json"), J("exam_declaration.json")
    if M:
        L.append(f"Default-template rows: {M['exam_b_n']}, ids sha256 `{M['exam_b_ids_sha256']}`; development rows for the unseen languages: {M['dev_n']}. Built {M.get('written_utc')} UTC.")
    if F:
        L.append(f"Format rows: {F['exam_formats_n']} ({F['n_requests']} requests, {F['n_harmful_requests']} harmful, in {len(F['formats'])} formats), ids sha256 `{F['exam_formats_ids_sha256']}`; rendered {F['written_utc']} UTC.")
    if D:
        L += ["", f"Declared {D['written_utc']} UTC, before any exam prompt was generated. Rule: {D['rule']}", "", "| cell | rows | role |", "| --- | --- | --- |"]
        L += [f"| {c['name']} | all | {c['role']} |" for c in D["cells"]] + [f"| {c['name']} | format rows only | {c['role']} |" for c in D.get("format_only_cells", [])]
        if D.get("calibration"):
            L += ["", "Calibrated s by exam format: " + ", ".join(f"{dict(FMT_EXAM).get(k, k)} {v:g}" for k, v in D["calibration"]["s_by_format"].items()) + f". Rule: {D['calibration']['rule']}. Calibration prompts: {D['calibration']['calibration_prompts']}."]
    L.append("")


def sec_guard():
    global L  # the functions extend the shared list in place with +=
    h("## Development: guard refit, positions lab, calibration")
    G = J("guard/g3.json")
    if G:
        a, b = G["new_guard_on_round5_held_out_positions"], G["v3g_guard_on_the_same_positions"]
        L += [f"Guard g3: {G['n_format_positions']} positions from fit-side prompts in fit formats ({G['n_format_markers']} markers, {G['n_earlier_reply_start_tokens_as_markers']} earlier reply-start tokens) added to round 5's {G['n_round5_fit_positions']}; lambda {G['lambda']}. "
              f"Margins on round 5's held-out positions: m {a['margin_m']} (v3g's guard {b['margin_m']}), q {a['margin_q']} ({b['margin_q']}). Source: `guard/g3.json`.", ""]
    P = {v: J(f"robust/positions_{v}.json") for v in ("v3g", "v4", "v5g")}
    if P["v3g"] and P["v4"]:
        vs = [v for v in ("v3g", "v4", "v5g") if P[v]]
        L += ["**Share of requests on which the strict side fires at s = 1 (forward passes; 80 harmful and 120 harmless development requests per format; 360 prompts for the languages row)**", "",
              "| development format | harmful: " + " / ".join(vs) + " | harmless: " + " / ".join(vs) + " | mean read z, harmful / harmless (v4) | read AUROC (v4) |", "| --- | --- | --- | --- | --- |"]
        for f, lab in FMT_DEV + [("lang", "nine other languages, default template")]:
            k = f if f == "lang" else f"rb_{f}"
            if k not in P["v4"]["formats"]:
                continue
            e = P["v4"]["formats"][k]
            L.append(f"| {lab} | " + " / ".join(f"{P[v]['formats'][k]['strict_s1_fires_harmful'] * 100:.1f}%" for v in vs) + " | " + " / ".join(f"{P[v]['formats'][k]['strict_s1_fires_harmless'] * 100:.1f}%" for v in vs)
                     + f" | {e['z_at_marker_mean_harmful']:.2f} / {e['z_at_marker_mean_harmless']:.2f} | {e['auroc_z']:.3f} |")
        L += ["", "Source: `robust/positions_<version>.json`. v5g is the read refit that was dropped (journal J8).", ""]
    C = J("cal/dev_v4.json")
    if C:
        L += ["**The calibration rule on the development formats** (" + C["rule"] + ")", "", "| development format | harmless fire rate at s = 1 / 1.25 / 1.5 / 1.75 / 2 | s by the rule | harmful fire rate at s = 1 | at the rule's s |", "| --- | --- | --- | --- | --- |"]
        for f, lab in FMT_DEV:
            if f in C["formats"]:
                e = C["formats"][f]
                L.append(f"| {lab} | " + " / ".join(f"{e['harmless_fire_rate_by_s'][s] * 100:.1f}%" for s in ("1", "1.25", "1.5", "1.75", "2")) + f" | {e['calibrated_s']:g} | {e['harmful_fire_rate_by_s']['1'] * 100:.1f}% | {e['harmful_fire_rate_at_calibrated_s'] * 100:.1f}% |")
        L += ["", "Source: `cal/dev_v4.json`. The rule uses the harmless column only.", ""]
    R, Gc = J("robust/rb_v4.json"), J("cal/gen_dev_v4.json")
    if R:
        L += ["**Behaviour on the development formats (generation; harmful unsafe % / harmful refusal % / harmless refusal %)**", "", "| development format | base | v3g, s = 1 | v4, s = 1 | v4 at the calibrated s | combined (v4 + lax) at s = 1 |", "| --- | --- | --- | --- | --- | --- |"]
        f3 = lambda e: f"{e['harmful_unsafe']:.1f} / {e['harmful_refusal']:.1f} / {e['harmless_refusal']:.1f}"
        for f, lab in FMT_DEV:
            if f in R:
                c = Gc.get(f, {}) if Gc else {}
                L.append(f"| {lab} | {f3(R[f]['base'])} | {f3(R[f]['v3gcap@1'])} | {f3(R[f]['v4cap@1'])} | " + (f"s = {c['calibrated_s']:g}: {f3(c['v4cal'])}" if c.get("v4cal") else "") + f" | {f3(R[f]['both4@1'])} |")
        L += ["", "Source: `robust/rb_v4.json`, `cal/gen_dev_v4.json`.", ""]
    X = J("run/d_xd6.json")
    if X:
        L += ["**Six unseen languages, development slice (per cell: harmful refusal % / unsafe %, pooled over the six languages; harmless refusal %)**", "", "| cell | harmful, base to cell | harmless refusal, base to cell | fires on harmful / harmless |", "| --- | --- | --- | --- |"]
        for c, s in X.items():
            if c == "base":
                continue
            hs, bs = [g for g in s if g.startswith("xd_art_")], [g for g in s if g.startswith("xd_ayab_")]
            w = lambda gs, k: sum(s[g][k] * s[g]["n"] for g in gs) / sum(s[g]["n"] for g in gs)
            L.append(f"| {c} | {w(hs, 'base_ref'):.1f} / {w(hs, 'base_uns'):.1f} to {w(hs, 'ref'):.1f} / {w(hs, 'uns'):.1f} | {w(bs, 'base_ref'):.1f} to {w(bs, 'ref'):.1f} | {w(hs, 'fire_gt_half') * 100:.0f}% / {w(bs, 'fire_gt_half') * 100:.0f}% |")
        L += ["", "Source: `run/d_xd6.json` (360 harmful and 219 harmless prompts).", ""]


def sec_exam():
    global L  # the functions extend the shared list in place with +=
    h("## Exam B")
    D = J("exam_declaration.json") or {}
    prod = (D.get("product") or {})
    for name, title in (("xb_tight", "Tightening, exam B, main judge"), ("xb_lax", "Loosening and the combined setting, exam B, main judge"), ("xb_tight_wg", "Tightening, exam B, English rows, WildGuard"),
                        ("xb_lax_wg", "Loosening, exam B, English rows, WildGuard"), ("xb_tight_q8", "Tightening, exam B, rows in other languages, Qwen3Guard-Gen-8B"), ("xb_lax_q8", "Loosening, exam B, rows in other languages, Qwen3Guard-Gen-8B")):
        if J(f"tables/{name}.json") is not None:
            S5.table_block(name, title=title)
    for c in [x for x in D.get("tight_cells", []) if x != D.get("control")]:
        for suf, jn in (("", "main judge"), ("_wg", "WildGuard, English rows"), ("_q8", "Qwen3Guard-Gen-8B, rows in other languages")):
            if J(f"tables/xb_tight{suf}_claims_{c}.json"):
                S5.claims_block(f"xb_tight{suf}_claims_{c}", title=f"{c} against the control at 0.1, exam B, {jn}")
    for suf, jn in (("", "main judge"), ("_wg", "WildGuard"), ("_q8", "Qwen3Guard-Gen-8B")):
        if prod.get("strict") and J(f"tables/xb_vs_ref{suf}.json"):
            S5.change_block(f"xb_vs_ref{suf}", prod["strict"], kinds=("H",), title=f"{prod['strict']} against {D.get('reference')} (the 'control' column): unsafe on harmful rows, {jn}", measure="unsafe")
            S5.change_block(f"xb_vs_ref{suf}", prod["strict"], kinds=("B",), title=f"{prod['strict']} against {D.get('reference')}: refusal on harmless rows, {jn}", measure="refusal")
        T = J(f"tables/xb_cal{suf}.json")
        if T:
            for c in [x for x in T["columns"] if x != "base"]:
                S5.change_block(f"xb_cal{suf}", c, kinds=("H",), title=f"{c} (calibrated s) against {prod.get('strict')} (the 'control' column): unsafe on the format rows, {jn}", measure="unsafe")
                S5.change_block(f"xb_cal{suf}", c, kinds=("B",), title=f"{c} (calibrated s) against {prod.get('strict')}: refusal on harmless format rows, {jn}", measure="refusal")
        T = J(f"tables/xb_lax{suf}.json")
        if T:
            for c in [x for x in (prod.get("lax"), prod.get("both")) if x in T["columns"]]:
                S5.change_block(f"xb_lax{suf}", c, kinds=("B",), title=f"{c}: refusal on harmless rows, {jn}", measure="refusal")
                S5.change_block(f"xb_lax{suf}", c, kinds=("H",), title=f"{c}: unsafe on harmful rows, {jn}", measure="unsafe")
    Fm = J("run/exam_formats.json")
    if Fm:
        cells = [c for c in next(iter(Fm.values())) if c != "base"]
        L += ["**Exam formats, cell by cell (harmful unsafe % / harmful refusal % / harmless refusal %; 80 harmful and 120 harmless requests per format)**", "", "| exam format | base | " + " | ".join(cells) + " |", "| --- | --- |" + " --- |" * len(cells)]
        for f, lab in FMT_EXAM:
            if f in Fm:
                L.append(f"| {lab} | " + " | ".join(f"{Fm[f][c]['harmful_unsafe']:.1f} / {Fm[f][c]['harmful_refusal']:.1f} / {Fm[f][c]['harmless_refusal']:.1f}" if c in Fm[f] else "" for c in ["base"] + cells) + " |")
        L += ["", "Source: `run/exam_formats.json`.", ""]
    for name in sorted(p_.stem for p_ in (OUT / "judge2").glob("*.json")) if (OUT / "judge2").exists() else []:
        Jd = J(f"judge2/{name}.json")
        if "cells" in Jd and "judge" in Jd and not name.startswith("wg_xbf_"):
            L += [f"**Second judge {Jd['judge']} against the main judge (`judge2/{name}.json`)**", "", "| cell | n | agreement on refusal | agreement on unsafe | unsafe main / second |", "| --- | --- | --- | --- | --- |"]
            L += [f"| {c} | {e['ALL']['n']} | {e['ALL']['agree_refusal'] * 100:.1f}% | {e['ALL']['agree_unsafe'] * 100:.1f}% | {e['ALL']['unsafe_main']:.1f} / {e['ALL']['unsafe_second']:.1f} |" for c, e in Jd["cells"].items()] + [""]


def sec_other():
    global L  # the functions extend the shared list in place with +=
    h("## Text quality, baselines, weight programming")
    Q = J("quality/quality.json")
    if Q:
        tasks = list(Q["tasks"])
        L += ["**Text quality with the adapters installed** (translation: chrF over 150 passages per direction; short answers: % of replies containing the gold answer, 150 questions per language; change against the base on the same prompts with a 95% interval)", "",
              "| cell | " + " | ".join(tasks) + " |", "| --- |" + " --- |" * len(tasks), "| base, score | " + " | ".join(f"{Q['cells']['base'][t]['score']:.1f}" for t in tasks) + " |"]
        for c, e in Q["cells"].items():
            if c != "base":
                L.append(f"| {c}: change (interval); identical replies; unit fires | " + " | ".join(f"{e[t]['change_vs_base']:+.1f} ({e[t]['change_ci95'][0]:+.1f} to {e[t]['change_ci95'][1]:+.1f}); {e[t]['identical_replies_pct']:.0f}%; {e[t]['fires_abs_gt_half']:.0f}%" for t in tasks) + " |")
        L += ["", "Source: `quality/quality.json`.", ""]
    for name, what in (("full", "1910 WildJailbreak development prompts, round 4 batching"), ("full_small", "the same prompts in batches of 8, own base")):
        St = J(f"safetune/{name}.json")
        if St:
            L += [f"**SafeTune baselines, {what}** (SafeTune {St.get('safetune_version')}; `safetune/{name}.json`)", "", "| cell | adversarial_harmful unsafe | adversarial_harmful refusal | adversarial_benign refusal | degenerate |", "| --- | --- | --- | --- | --- |",
                  f"| base | {St['base']['harmful_unsafe']:.1f} | {St['base']['harmful_refusal']:.1f} | {St['base']['benign_refusal']:.1f} | |"]
            L += [f"| {c} | {e['harmful_unsafe']:.1f} | {e['harmful_refusal']:.1f} | {e['benign_refusal']:.1f} | {e['degenerate']:.1f} |" for c, e in St["cells"].items()]
            if St.get("failures"):
                L += ["", "Failures: " + "; ".join(f"{k}: {v[:100]}" for k, v in St["failures"].items())]
            L.append("")
    Dw = J("run/d_wjx.json")
    if Dw:
        L += ["**The knob on the same 1910 prompts** (`run/d_wjx.json`; adversarial_harmful unsafe, adversarial_benign refusal)", "", "| cell | unsafe, base to cell | benign refusal, base to cell |", "| --- | --- | --- |"]
        L += [f"| {c} | {s['ADV_H']['base_uns']:.1f} to {s['ADV_H']['uns']:.1f} | {s['ADV_B']['base_ref']:.1f} to {s['ADV_B']['ref']:.1f} |" for c, s in Dw.items() if c != "base"] + [""]
    W = J("wp/flag_test.json")
    if W:
        L += ["**Weight programming, experiment A: a position flag written at the input (kill test)** (`wp/flag_test.json`)", "",
              f"Per-dimension size of the residual at the input of layers 0, 1, 22: {W['residual_rms_by_layer_input']['0']}, {W['residual_rms_by_layer_input']['1']}, {W['residual_rms_by_layer_input']['22']}; of the reply-start token's embedding row: {W['embedding_row_rms_of_reply_start_token']}. Passes K1 and K2: {W['passes_K1_and_K2']}.", "",
              "| direction | flag size (x row norm) | read at the true marker, median | natural spread along it | mean KL against base | first token unchanged |", "| --- | --- | --- | --- | --- | --- |"]
        L += [f"| {e['eigen_index']} | {e['c_relative_to_embedding_row_norm']:g} | {e['flag_at_true_marker_min_median_max'][1]:.3f} | {e['natural_std_along_phi']:.3f} | {e['kl_next_token_mean_max'][0]:.1e} | {e['argmax_same_pct']:.1f}% |" for e in W["candidates"]] + [""]


def quality_tokens(text):
    """{{Q:cell|mt or qa|range}}: the range over that kind of task of a cell's change against base; {{Q:cell|mt or qa|sig}}: the tasks whose 95% interval excludes 0."""
    import re as _re
    Q = J("quality/quality.json")
    def one(m):
        cell, kind, what = m.group(1).split("|")
        if not Q or cell not in Q["cells"]:
            return f"[MISSING: Q:{m.group(1)}]"
        e = {t: v for t, v in Q["cells"][cell].items() if t.startswith(kind + "_")}
        if what == "range":
            v = [x["change_vs_base"] for x in e.values()]
            return f"{min(v):+.1f} to {max(v):+.1f}"
        if what == "identical":
            v = [x["identical_replies_pct"] for x in e.values()]
            return f"{min(v):.0f}% to {max(v):.0f}%"
        if what == "base":
            v = [Q["cells"]["base"][t]["score"] for t in e]
            return f"{min(v):.1f} to {max(v):.1f}"
        sig = [f"{t[3:].replace('_', ' to ')} {x['change_vs_base']:+.1f} ({x['change_ci95'][0]:+.1f} to {x['change_ci95'][1]:+.1f})" for t, x in e.items() if x["change_ci95"][0] > 0 or x["change_ci95"][1] < 0]
        return "; ".join(sig) if sig else "none"
    return _re.sub(r"\{\{Q:(.+?)\}\}", one, text)


def main():
    L.clear()
    L.extend(["# Round 6: numbers written by script", "", "Generated by `scripts/r6_summary.py` from the JSON files under `results/round6/`. Do not edit by hand."])
    for sec in (sec_sets, sec_guard, sec_exam, sec_other):
        try:
            sec()
        except Exception as e:  # a missing or half-written file must not stop the rest
            L.append(f"_(section {sec.__name__} stopped: {type(e).__name__}: {e})_")
    (OUT / "summary_numbers.md").write_text("\n".join(L) + "\n")
    tpl = OUT / "summary_prose.md"
    if tpl.exists():
        body = S5.fill(quality_tokens(tpl.read_text()))
        nums = (OUT / "summary_numbers.md").read_text().split("\n", 1)[1]
        (OUT / "summary.md").write_text(body.rstrip() + "\n\n---\n\n# Appendix: every table, written by script\n" + nums)
        import re as _re
        missing = _re.findall(r"\[MISSING: .+?\]", body)
        print(f"summary.md written; unresolved tokens: {len(missing)}")
        for x in missing[:12]:
            print("  ", x)
    if "--quiet" not in sys.argv:
        print("\n".join(L)[:3000])


if __name__ == "__main__":
    main()
