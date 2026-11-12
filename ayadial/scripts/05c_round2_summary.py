"""Writes results/round2/summary.md from the files in results/round2/. Numbers only, no prompt text."""
import csv
import json

from common import RESULTS, load_set

R = RESULTS / "round2"
SETS = ["harmful", "harmful_jb", "xstest_safe", "orbench_hard", "benign"]
HEAD = "| method | s | harmful | harmful_jb | xstest_safe | orbench_hard | benign |\n| --- | --- | --- | --- | --- | --- | --- |"


def pct(v):
    return "n/a" if v in (None, "") else f"{float(v) * 100:.1f}"


def interval(c):
    return "n/a" if not c or c[0] is None else f"{pct(c[0])} to {pct(c[1])}"


def load(block):
    return json.loads((R / block / "results.json").read_text())


def main():
    C, F, S = load("blockC"), load("blockF"), load("scoring")
    T = json.loads((R / "blockF" / "token_scores.json").read_text())
    rows = {(r["method"], float(r["s"]), r["set"]): r for r in csv.DictReader(open(R / "all_methods.csv"))}
    base = S["base"]
    ref = lambda m, s, n: float(rows[(m, s, n)]["refusal"])
    line = lambda m, s: " | ".join(pct(rows[(m, s, n)]["refusal"]) if (m, s, n) in rows else "n/a" for n in SETS)
    table = lambda name, m, s: f"| {name} | {s:g} | {line(m, s)} |"

    loosen = (-0.25, -0.5, -0.75, -1.0, -1.5)
    keeps = [s for s in loosen if ref("prop_benign_prompt", s, "harmful") >= 0.80 and base["harmful_jb"] - ref("prop_benign_prompt", s, "harmful_jb") <= 0.05]
    L = [
        "# Round 2 summary", "",
        "Layer 23, cleaned direction, eval halves only, Qwen3Guard-Gen-0.6B judge, degenerate responses excluded, 1000-resample bootstrap CIs. Base, ungated, prop, gate and gate_inv generations were reused from phase 2 and overnight.", "",
        "Base refusal (%): " + ", ".join(f"{n} {pct(base[n])}" for n in SETS) + ".", "",
        "## Block A: prop_benign by prompt", "", HEAD,
    ]
    L += [table("prop_benign_prompt", "prop_benign_prompt", s) for s in loosen]
    L += ["", f"Scales keeping harmful >= 80 and harmful_jb within 5 of base: {keeps if keeps else 'none'}. Ramp or step: a ramp. orbench_hard falls at every scale, but harmful and harmful_jb fall with it.", "",
          "## Block B: Wang false-refusal direction vs gate_inv at matched harmful", "", HEAD]
    L += [table(m, m, s) for m in ("wang", "wang_gated") for s in (0.5, 1.0)]
    L += ["", "Loosen comparison on the harmful axis: points removed from base, interpolated at the first crossing of the target.", ""]
    for target, d in S["loosen_matched_on_harmful"].items():
        L.append(f"**At harmful refusal {float(target) * 100:.0f}%**")
        for name, label in (("loosened_xs", "xstest_safe"), ("loosened_or", "orbench_hard")):
            L.append(f"- {label}: " + "; ".join(f"{m} {pct(v['value'])} (CI {interval(v['ci'])})" if v["value"] is not None else f"{m} does not reach this level" for m, v in d[name].items()))
        L.append("")
    L += ["Wang never lowers harmful refusal to 85% or 83%, so it has no matched value; its table row is the comparison. The first-crossing rule understates gate_inv's plateau, where harmful refusal stays near 82 to 83% while orbench_hard continues to fall (see all_methods.csv).", "",
          "## Block C: AlphaSteer vs prop at matched harmful_jb", "",
          f"Single layer 23. Fit: {C['fit']['benign_rank_removed']} benign directions removed, ridge {C['fit']['ridge']:.4f}. M is rank 1: h' = h + s (h . w) r_raw.", "", HEAD]
    L += [table("alphasteer", "alphasteer", s) for s in (0.5, 1.0, 1.5)]
    L.append("")
    for target, d in S["tighten_matched_on_harmful_jb"].items():
        L.append(f"**Harmless refusal at harmful_jb refusal {float(target) * 100:.1f}%**")
        L.append("- " + "; ".join(f"{m} {pct(v['value'])} (CI {interval(v['ci'])})" if v["value"] is not None else f"{m} does not reach this level" for m, v in d["harmless"].items()))
        L.append("")
    L += ["## Block D: SCANS vs gate and gate_inv", "", HEAD]
    for s in (0.5, 1.0):
        L += [table("scans", "scans", s), table("gate", "gate", s), table("gate_inv", "gate_inv", -s)]
    L += ["", "SCANS does not coincide with either method alone, as expected: it applies +s above tau and -s below tau, so each prompt gets one of the two pushes. Comparing the two partitions separately needs the raw responses and was not done.", "",
          "## Block F1: carry-through from the last prompt token", "", HEAD]
    L += [table(m, m, s) for m in ("prop_last", "gate_last") for s in (1.0, 2.0, 4.0)]
    L += [table("prop (all positions)", "prop", 1.0), table("gate (all positions)", "gate", 1.0), "",
          f"Carry-through: yes. gate_last at s=1 gives harmful_jb {pct(ref('gate_last', 1.0, 'harmful_jb'))} against {pct(ref('gate', 1.0, 'harmful_jb'))} for all positions, and xstest_safe {pct(ref('gate_last', 1.0, 'xstest_safe'))} against {pct(ref('gate', 1.0, 'xstest_safe'))}. prop keeps part of the effect at s=1 (harmful_jb {pct(ref('prop_last', 1.0, 'harmful_jb'))} against {pct(ref('prop', 1.0, 'harmful_jb'))}).", "",
          "## Block F2: per-token gate on generated tokens", "",
          f"tau_gen {F['f2']['tau_gen']:.3f} at 90% TPR on {F['f2']['n_tokens']['harmful']} generated harmful tokens; it fires on {pct(F['f2']['gate_fpr_on_generated_benign'])}% of {F['f2']['n_tokens']['benign']} generated benign tokens.", "", HEAD]
    L += [table("gate_token", "gate_token", s) for s in (0.5, 1.0)]
    gain = (ref("gate_token", 0.5, "harmful_jb") - base["harmful_jb"]) * 100
    cost = [(ref("gate_token", s, "xstest_safe") - base["xstest_safe"]) * 100 for s in (0.5, 1.0)]
    L += ["", f"Per-token gate: yes on harmful_jb (s=0.5 raises it by {gain:.1f} points), but it costs {cost[0]:.1f} points on xstest_safe at s=0.5 and {cost[1]:.1f} at s=1. Scores on generated tokens barely separate harmful from benign completions (harmful mean {T['harmful_mean']:.2f}, benign mean {T['benign_mean']:.2f}; blockF/token_scores.json).", "",
          "## Recommendation for phase 3", "",
          "Fold AlphaSteer. It gives the lowest harmless refusal at matched harmful_jb among the methods here, leaves xstest_safe near base, and its matrix is rank 1 and per token, so it needs no memory across positions. It tightens only. Parity between a fold and this hook is still unproven: the overnight down_proj adapter did not reproduce the prop hook.", "",
          "## Notes", "",
          "- Eval halves cap the sets at 146 harmful, 292 harmful_jb and 125 for each of the others.",
          "- AlphaSteer, Wang and SCANS were implemented from the descriptions in the run prompt; the papers were not re-read for this run.",
          "- The run script (round2.py) and the raw responses (runs_round2/) stayed on the Colab runtime.", ""]
    text = "\n".join(L)
    (R / "summary.md").write_text(text, encoding="utf8")

    check = text + (R / "all_methods.csv").read_text(encoding="utf8")
    leaks = sum(r["prompt"] in check for name in ("harmful", "xstest_unsafe") for r in load_set(name))
    print(f"wrote {R / 'summary.md'}; harmful text hits: {leaks}; methods: {sorted({k[0] for k in rows})}")


if __name__ == "__main__":
    main()
