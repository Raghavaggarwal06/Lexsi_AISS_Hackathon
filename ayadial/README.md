# ayadial: a per-request strictness knob in the weights of a small open model

AyaDial makes [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) refuse harmful
requests more often by rewriting **two dormant MLP units of layer 22**. Nothing is trained. The edit is computed in
closed form from the model's own activations and ships as LoRA files, so strictness `s` is a number the caller sets
on each request.

**Model:** `CohereLabs/tiny-aya-global`, `Cohere2ForCausalLM`, 3.35B, bf16, 36 layers, CC-BY-NC-4.0, auto-gated.
**Judge:** `Qwen/Qwen3Guard-Gen-0.6B` (Apache-2.0).

**[The report (PDF)](report/ayadial_report.pdf)** · [extended result tables](report/ayadial_results.pdf)

## Status (2026-11-12)

- Report and extended tables compiled. Every number in them is read from a file in `results/`.
- Both variants measured: **AyaDial** (closed-form edit) and **AyaDial-WSP** (the same two units, edited by a
  frontier model that read the weights directly; harness in `wsp/`).
- The four GGUF adapters were exported on Colab and **are not in this repo yet**. The vectors, thresholds, unit
  indices, pinned llama.cpp commit and expected SHA256 are all here; one re-export run rebuilds them byte-for-byte.
  See [`adapter/README.md`](adapter/README.md).

## What it does

Share of **unsafe replies**, base model against AyaDial, with the best competing method for scale:

| test | base | AyaDial | best other method |
| --- | --- | --- | --- |
| in-the-wild jailbreak templates | 28.0% | **9.4%** | 20.9% |
| random-search attack | 71.4% | **9.1%** | 39.0% |
| WildJailbreak behind a long system prompt (AyaDial-WSP) | 40.6% | **16.0%** | — |

Harmless requests are refused about as often as with the other methods.

## How it works

The hidden state entering layer 22, at the token where the reply starts, is projected onto a direction `w` fitted by
linear discriminant. That gives a harm score `z`, near 0 for a typical harmless request and near 1 for a harmful one.
Units 6666 and 1896 are dormant in the base model, so rewriting them changes nothing else. They are set so that
together they compute

```
push(z) = min(1, max(0, β·(s·z − θ))) · r        β = 4, θ = 0.537
```

along the model's own refusal direction `r`. Below the threshold nothing is written at all, and a second vector keeps
the units shut at every token except the reply start.

`s` is the dial: `s = 0` writes nothing, `s = 1` is the setting used throughout the report, higher is stricter. A
second pair of units (1785 and 4466) implements **Loosen**, which lowers refusals on harmless-but-scary requests, so
the dial goes both ways. `adapter/README.md` has the mapping from one user-facing `s` onto the four adapter scales.

## Layout

```
report/     the paper and the extended tables, with LaTeX sources
adapter/    the fitted vectors, how to rebuild the four LoRA files and how to serve them
results/    the 73 JSON files every number in the report is read from
scripts/    the code that produced them, in run order
wsp/        the edit loop used for AyaDial-WSP (runs and caches stripped)
docs/       the runbook for the round that produced the SafeTune rows
data/       the prompt sets that can be redistributed
```

`requirements.txt` is the loose floor; `requirements-frozen.txt` is the exact environment behind the
numbers, and `docs/RUNBOOK-round7.md` has the Colab pins, which differ.

## Reproducing

### What runs from this repo alone

```bash
pip install -r requirements.txt     # requirements-frozen.txt has the exact versions
hf auth login                       # tiny-aya-global, XSTest and HarmBench are gated
python scripts/00_env_check.py
python scripts/01_fetch_data.py     # rebuilds data/, including the sets not shipped here
python scripts/02_extract.py
python scripts/03_direction.py      # the separation test; about 1 h on a Mac
```

That reproduces phase 1 end to end and should land on the AUROC in `results/day1/`.

**Rebuild the four adapters** (needs a CUDA GPU for `prep`, which downloads the base model and builds
llama.cpp; the adapters themselves are written in seconds):

```bash
python scripts/r5_gguf.py prep
python scripts/r5_gguf.py test 0.3747 2.0
```

Check the output against the hashes in [`adapter/README.md`](adapter/README.md). The export is
deterministic, so matching hashes mean you have the artifacts the report measured.

**Recompute the report's tables without a GPU.** `results/round5|6|7/labels/` hold the per-row
judgements as ids and 0/1 only, so the scoreboards and paired differences can be recomputed from
them with `scripts/r5_report.py` and `r6_report.py`, and the report itself rebuilt with
`scripts/report/build_report_tex.py`.

### What needs more than this repo

Rerunning the **generation** half of rounds 4 to 7 needs the prompt suites, and those hold request
text: `runs_private/*/sets.jsonl`, the in-the-wild system templates, the format requests, and the
harmful sets. They are deliberately not published here. `scripts/r7_bundle.py` packs them into the
private bundle the runbook expects, shared over Drive rather than through this repo. With the bundle,
[`docs/RUNBOOK-round7.md`](docs/RUNBOOK-round7.md) has the literal Colab cells and the expected
output of every stage.

So: the fits, the adapters and every table recompute from what is here; regenerating replies from
scratch needs the bundle and a GPU.

## Relation to the other folders here

`../wsp-safety` is the team's reimplementation of *Sparks of Weight-Space Programming* and scores that method on its
own safety tasks. AyaDial-WSP borrows the idea, not the code: `wsp/` here is a separate edit loop aimed at one
target, the two units above. `../cast-safety` covers conditional activation steering, which appears in this report as
one of the competing methods.

## What is deliberately not here

- **Harmful prompt sets and raw generations.** `data/harmful.jsonl`, `data/xstest_unsafe.jsonl` and every raw reply
  log stay local. `scripts/01_fetch_data.py` rebuilds them from the original gated sources, and anything published
  carries row ids only, never prompt text.
- **Run caches.** `wsp/runs/` and `wsp/cache/` are 1.4 GB of intermediate state, including a slice of the base model
  weights that cannot be redistributed.
- **The literature review.** The paper reading and idea selection that led here are a separate body of work.

## Licence

Code here is the team's to license. The **adapter weights are a derivative of Tiny Aya and inherit CC-BY-NC-4.0**:
attribution required, non-commercial only. That is not a choice we can make differently. See [NOTICE.md](NOTICE.md)
for the full breakdown, including each prompt set's own terms.
