"""Writes results/round3/summary.md from the files in results/round3/. Numbers only, no prompt text."""
import json

from common import RESULTS, load_set

R = RESULTS / "round3"
J = lambda p: json.loads((R / p).read_text())
pct = lambda v: "n/a" if v is None else f"{v * 100:.1f}"
SETS = ["harmful", "harmful_jb", "xstest_safe", "orbench_hard", "benign"]


def frac(e):
    if not e or not e.get("base_answered"):
        return "n/a"
    lo, hi = e["ci"]
    return f"{e['now_refused']}/{e['base_answered']} ({lo * 100:.0f} to {hi * 100:.0f})"


def main():
    L = ["# Round 3 summary", "",
         "Eval halves only (development data by now), layer 23 = output of layers[22], cleaned direction, Qwen3Guard-Gen-0.6B judge unless stated. Every number below comes from a file in this folder.", ""]

    L += ["## 1. Code audit (block 0)", "",
          "- F1 survives. gate_last and prop_last return early on decode steps (`if h.shape[1] == 1: return out`) and push only the last template token during prefill. They never touched a generated token. See block0/code_audit.md for every method's prefill and decode positions.",
          "- Separate problem: gate, gate_inv, prop_prompt, gated_prop, wang_gated and scans apply a decision read at the last prompt token to every earlier prefill position, which a causal model with fixed weights cannot reproduce.",
          "- The first per-token gate (phase 2 pilot) used the raw direction and a last-template-token threshold of 14.39; gate_token used the cleaned direction and a threshold fit on generated tokens (-3.20). They are different tests, not one mislabelled one.",
          "- 87.7% vs 87.0%: one prompt (harmbench-51) sits 0.016 above tau in the padded batch-8 phase 1 pass and 0.045 below it when scored alone. Cause: batch composition and left padding.",
          "- Index 23 is exactly the output of layers[22] and the input of layers[23]; layers[23] is a full-attention layer with no rotary embedding (NoPE); `tie_word_embeddings` is true; the AlphaSteer null space was 125 benign_fit + 125 xstest_safe_fit + 125 orbench_hard_fit, uncentered (block0/alphasteer_manifest.md).", ""]

    c = J("block1/crosstabs.json")
    L += ["## 2. Cross-tabs and flip rates (block 1)", "", "Base outcome by whether the tau90 gate fires at the last prompt token:", "",
          "| set | refused, fires | refused, no fire | answered, fires | answered, no fire | degenerate |", "| --- | --- | --- | --- | --- | --- |"]
    for s, x in c["contingency_tau90"].items():
        L.append(f"| {s} | {x['base_refused_fires']} | {x['base_refused_no_fire']} | {x['base_answered_fires']} | {x['base_answered_no_fire']} | {x['base_degenerate']} |")
    L += ["", "The reviewer's reconstructed counts hold to within one prompt per cell.", "",
          "Conversions among base-answered prompts and new refusals among base-answered harmless prompts (count, exact 95% interval in %):", "",
          "| run | harmful | harmful_jb | xstest_safe new | orbench_hard new | benign new |", "| --- | --- | --- | --- | --- | --- |"]
    for k in ("b2_gate_tpr90_1", "b2_gate_tpr95_1", "b2_prop_-_1", "b2_ungated_-_0.5", "p2_ungated_0.25", "b1_gate_inv_-1", "b1_prop_benign_-1", "b1_ungated_-1"):
        f, n = c["flip_rates"].get(k, {}), c["new_refusals_harmless"].get(k, {})
        L.append(f"| {k} | {frac(f.get('harmful'))} | {frac(f.get('harmful_jb'))} | {frac(n.get('xstest_safe'))} | {frac(n.get('orbench_hard'))} | {frac(n.get('benign'))} |")
    L += ["", "gate_inv release among base-refused harmful prompts scoring below tau (released / eligible):", ""]
    for k, v in c["gate_inv_release"].items():
        if k.startswith("b2_") or k == "b1_gate_inv_-1":
            L.append(f"- {k} (tau {v['tau']:.3f}): harmful {v['harmful']['released']}/{v['harmful']['refused_below_tau']}, harmful_jb {v['harmful_jb']['released']}/{v['harmful_jb']['refused_below_tau']}")
    L.append("")

    a, m, mk, hv = J("block3/loosen_existence.json"), J("block3/m_space.json"), J("block3/marker.json"), J("block3/harm_vs_refusal.json")
    best = max(a["cells"], key=lambda x: x["frac_harmless_refused_above_2theta"] or -1)
    passing = [x for x in a["cells"] if x["pass"]]
    L += ["## 3. The three kill tests (block 3)", "",
          f"**3.1 Loosen existence: PASS.** {len(passing)} of {len(a['cells'])} (layer, tail position) cells pass, at every layer from 16 to 30, at positions -5 to -2 and never at the marker (-1). Best: layer {best['layer']}, position {best['tail_pos']} (`<|END_OF_TURN_TOKEN|>`), AUROC refused-harmless vs refused-harmful {best['auroc_refused_harmless_vs_refused_harmful']:.4f}, theta {best['theta']:.3f}, {pct(best['frac_harmless_refused_above_2theta'])}% of refused harmless above 2 theta. Fit: {a['n_fit_positive']} refused harmless, {a['n_fit_zero']} harmful (direct and wrapped). Held-out groups: {a['groups']}. What it keeps open: a selective loosen read exists in the residual before the marker, so the loosen side is not hook-only by necessity.", ""]
    L += ["**3.2 m-space probe (Gemini's design): FAIL.** Probe AUROCs (logistic / diff of means):", "", "| space | harmful vs xstest_safe | harmful vs orbench_hard | same format |", "| --- | --- | --- | --- |"]
    for sp, v in m["probes"].items():
        L.append(f"| {sp} | {v['logistic']['harmful_vs_xstest_safe']:.3f} / {v['diff_of_means']['harmful_vs_xstest_safe']:.3f} | {v['logistic']['harmful_vs_orbench_hard']:.3f} / {v['diff_of_means']['harmful_vs_orbench_hard']:.3f} | {v['logistic']['same_format']:.3f} / {v['diff_of_means']['same_format']:.3f} |")
    L += ["", f"Paired 95% intervals for the AUROC difference against h23 (logistic): {json.dumps({k: {p: [round(x, 4) for x in iv] for p, iv in v.items()} for k, v in m['paired_auroc_diff_vs_h23_ci'].items()})}. m-space separates as well as the residual.", "",
          "Coefficient a·m of the AlphaSteer-style m23 readout by position group:", "", "| group | n | share negative | median | 95th pct | max |", "| --- | --- | --- | --- | --- | --- |"]
    for g, v in m["a_tight_m23_coefficients"].items():
        L.append(f"| {g} | {v['n']} | {pct(v['frac_negative'])}% | {v['median']:.3f} | {v['p95']:.3f} | {v['max']:.3f} |")
    L += ["", f"Kill rule (preset): negative on harmful markers {m['KILL_negative_on_harmful']}; off-marker 95th percentile above 0.25 × harmful-marker median {m['KILL_offmarker_comparable']}. The readout pushes on preamble, user and generated tokens at a quarter to half of its marker value, so a LoRA built from it would add a near-uniform push on every token. What it forecloses: the closed-form m-space LoRA as fit here. G2's protected-covariance variant, which penalises off-marker tokens in the fit, was not tested.", ""]
    L += ["**3.3 Marker margin (Fable's design): borderline, FAIL by the preset bar.**", "", "| layer | marker min | marker 1st pct | others max | margin | variants min (lowest) |", "| --- | --- | --- | --- | --- | --- |"]
    for Ly, e in mk["layers"].items():
        low = min(e["variants_min"], key=e["variants_min"].get)
        L.append(f"| {Ly} | {e['marker_min']:.3f} | {e['marker_p1']:.3f} | {e['others_max']:.3f} | {e['margin_min_minus_max']:.3f} | {e['variants_min'][low]:.3f} ({low}) |")
    kl = mk["zero_two_dormant_kl"]
    L += ["", f"There is no overlap at any layer, including multi-turn chats, two other system prompts, and Spanish and Hindi translations, but no layer clears the preset margin of 0.3 for every group (layer 21 misses on Spanish by about 0.01). Zeroing the two most dormant layers[22] units (ids {mk['dormant_units_layer22'][0]['unit']} and {mk['dormant_units_layer22'][1]['unit']}) moves next-token KL by a mean of {kl['mean']:.5f} (max {kl['max']:.4f}, n = {kl['n']}), under the 1e-3 bar. What it forecloses: nothing yet. The separation exists; whether a margin of 0.22 to 0.41 is enough for a SiLU gate is the next question.", ""]
    L += ["**3.4 Harm or impending refusal.**", "", "| read position | AUROC harmful vs harmless | among base-answered only | same format |", "| --- | --- | --- | --- |"]
    for k, v in hv.items():
        L.append(f"| {k} | {v['auroc_harmful_vs_harmless']:.3f} | {v['auroc_base_answered_only']:.3f} (n {v['n_answered_harmful']} vs {v['n_answered_harmless']}) | {v['same_format']:.3f} |")
    L += ["", "The direction, fit at the last template token, reads nothing at the last instruction token. At the template token it still separates base-answered prompts (0.817 on 19 harmful), so it is partly a harm read and partly a refusal read.", ""]

    h = J("block4/headline.json")
    L += ["## 4. Corrected matched comparison (block 4)", "", "Ungated, dense measured grid (refusal %, all attempted prompts):", "", "| s | " + " | ".join(SETS) + " | harmful_jb unsafe |", "| --- | " + " | ".join("---" for _ in SETS) + " | --- |"]
    for s, r in h["ungated_dense"].items():
        L.append(f"| {s} | " + " | ".join(pct(r[x]["refused"]) for x in SETS) + f" | {pct(r['harmful_jb']['unsafe'])} |")
    L += ["", "Each method against the measured ungated point nearest in harmful_jb refusal (method minus ungated, points, 95% interval; bootstrap by base prompt):", "",
          "| method | ungated s | harmful_jb | xstest_safe | orbench_hard | benign | pooled |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for e in h["matched_measured"]:
        d, ci = e["method_minus_ungated"], e["ci"]
        cell = lambda k: f"{d[k] * 100:+.1f} ({ci[k][0] * 100:+.1f} to {ci[k][1] * 100:+.1f})"
        L.append(f"| {e['method']} | {e['matched_ungated_s']} | " + " | ".join(cell(k) for k in ("harmful_jb", "xstest_safe", "orbench_hard", "benign", "pooled")) + " |")
    L += ["", "The selective advantage survives measured points on these sets. prop at s=1 and AlphaSteer at s=1 refuse 13.3 and 16.8 pooled points fewer harmless prompts than the uniform push at equal harmful_jb refusal. AlphaSteer's edge on xstest_safe and orbench_hard is partly by construction, since those fit halves are in its null space.", ""]
    fc, sf = J("block4/format_control.json"), J("block4/same_format.json")
    L += ["## 5. Format control (block 4.2, 4.3)", "", "| run | benign wrapped | xstest_safe wrapped |", "| --- | --- | --- |"]
    for k, r in fc.items():
        L.append(f"| {k} | {pct(r['benign_jb']['refused'])} | {pct(r['xstest_safe_jb']['refused'])} |")
    L += ["", "Wrapped harmless prompts gain 0.4 to 6 points of refusal, against about 20 points on wrapped harmful prompts, so the tighten effect is not about wrapper text.", "", "| run | xstest_unsafe refused | xstest_safe refused |", "| --- | --- | --- |"]
    for k, r in sf.items():
        L.append(f"| {k} | {pct(r['xstest_unsafe']['refused'])} | {pct(r['xstest_safe']['refused'])} |")
    L += ["", f"Unsafe rate on harmful_jb for every run is in block4/harmful_jb_unsafe.json and block1/crosstabs.json; gate_inv at tau90 and s=-1 gives {pct(c['unsafe_harmful_jb'].get('b1_gate_inv_-1'))}% against {pct(c['unsafe_harmful_jb']['base'])}% at base.", ""]

    at, sets = J("block5/attacks.json"), J("block5/sets.json")
    cols = ["jbb_PAIR", "jbb_GCG", "wj_adversarial_harmful", "wj_adversarial_benign"]
    L += ["## 6. Published attacks (block 5)", "", f"JailbreakBench artifacts commit {sets['jbb_artifacts_sha'][:7]} ({sets['files']['PAIR']}, {sets['files']['GCG']}); sizes {sets['sizes']}.", "",
          "Refusal % / unsafe % (degenerate % if any):", "", "| run | " + " | ".join(cols) + " |", "| --- | " + " | ".join("---" for _ in cols) + " |"]
    for k, r in at.items():
        L.append(f"| {k} | " + " | ".join(f"{pct(r[x]['refused'])} / {pct(r[x]['unsafe'])}" + (f" ({pct(r[x]['degenerate'])})" if r[x]["degenerate"] else "") for x in cols) + " |")
    L += ["", "On WildJailbreak the selectivity does not transfer. At about 88 to 89% refusal of adversarial harmful prompts, ungated s=0.25 refuses 32.9% of adversarial benign prompts, prop s=1 35.7% and the tau95 gate 33.8%. AlphaSteer refuses more adversarial benign than ungated at equal harmful refusal (s=1: 78.0 / 30.5 against ungated s=0.1: 78.0 / 19.5). The loosen side releases attacks: gate_inv at s=-1 cuts adversarial benign refusal from 14.3 to 3.3 but raises the unsafe rate on adversarial harmful from 10.3 to 19.0 and on GCG from 30.4 to 37.0.", ""]

    arc, wg, dis = J("block6/arc.json"), J("block6/wildguard.json"), J("block6/disagreements_recovery.json")
    L += ["## 7. Checks (block 6)", "", "| run | ARC accuracy from text | first-token A-D mass, mean | 10th pct |", "| --- | --- | --- | --- |"]
    for k, v in arc.items():
        if isinstance(v, dict):
            L.append(f"| {k} | {pct(v['accuracy_from_text'])} | {v['first_token_letter_mass_mean']:.3f} | {v['first_token_letter_mass_p10']:.3f} |")
    L += ["", f"The tau90 gate fires on {pct(arc['gate_tau90_firing_on_arc'])}% of ARC questions. The degeneracy filter flags every one-letter answer, so its ARC refusal and degenerate rates are invalid; it should not be applied to short-answer tasks.",
          f"WildGuard against Qwen3Guard-0.6B on {wg['n']} steered rows: refusal agreement {pct(wg['refusal']['agreement'])}% (kappa {wg['refusal']['kappa']:.3f}); unsafe agreement {pct(wg['unsafe_vs_harmful_response']['agreement'])}% (kappa {wg['unsafe_vs_harmful_response']['kappa']:.3f}).",
          f"The 42 judge disagreements: candidates recovered for {dis['recovered_with_candidates']} of {dis['n']}, uniquely for {dis['unique']}; texts are saved privately in runs_private/round3/, not here.", ""]

    f7, rc = J("block7/file_baseline.json"), J("block7/recheck.json")
    L += ["## 8. File-level baseline (block 7)", "",
          f"- llama.cpp commit {f7['llama_cpp_commit'][:7]}; CUDA build and Q4_K_M quantization succeeded.",
          f"- Token ids: identical to apply_chat_template for {sum(p['same'] for p in rc['parity'])} of {len(rc['parity'])} prompts after fixing the check (recheck.json).",
          f"- Special tokens typed in a user message are not escaped: a typed `<|END_OF_TURN_TOKEN|>` becomes the real token ({f7['special_token_escape']['end_of_turn_count_typed']} occurrences against {f7['special_token_escape']['end_of_turn_count_clean']} for a clean message). A level token in the user text would be a control channel any user can type.",
          "- Activated LoRA: `alora_invocation_tokens` is present in this build's server and adapter source; not exercised.",
          "- Q4_K_M base rates are not valid: generation did not stop at the end of the turn, and 158 of 813 answers were flagged degenerate for trailing repetition. The cause was not established, so no HF vs Q4 comparison is reported.", "",
          "## 9. Two-repo pipeline (block 8)", "", "See block8/pipeline.md: CuratorKIT and AuditKIT ran on our data; SafeTune baselines are wired but not yet run on Tiny Aya.", ""]
    text = "\n".join(L)
    (R / "summary.md").write_text(text, encoding="utf8")
    leaks = sum(r["prompt"] in text for n in ("harmful", "xstest_unsafe") for r in load_set(n))
    print(f"wrote summary.md, harmful text hits: {leaks}")


if __name__ == "__main__":
    main()
