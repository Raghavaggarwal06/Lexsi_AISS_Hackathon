# CAST replication: results (2026-11-09)

Models: Qwen 1.5 Chat 1.8B and Hermes 2 Pro (Llama 3 8B), at the paper's pinned revisions. Data: the release's
`docs/demo-data`. Refusal judge: paper App D.2 (classifier OR keywords). Metric files: `results/<model>/<run>/`.
All numbers below use the release's pooling (the default; see "Pooling" for why).

## Sec 4: conditioned refusal (450 Sorry-Bench harmful / 500 Alpaca harmless, refusal %)

| model | setting | ours harmful / harmless | paper harmful / harmless |
|---|---|---|---|
| Qwen 1.5 1.8B | base | 49.1 / 3.2 | 45.8 / 0.0 |
| | +refusal, every prompt (L10–20, α=4) | 100.0 / 95.8 | 100.0 / 96.4 |
| | **CAST**, our grid point (L6, sim > 0.195, F1 .90) | **88.9 / 3.8** | **90.7 / 2.2** |
| | CAST, the paper's point (L8, sim > 0.031) | 94.0 / 5.0 | 90.7 / 2.2 |
| Hermes 2 Pro 8B | base | 21.6 / 2.6 | 19.3 / 1.0 |
| | +refusal, every prompt (L15,17–24, α=1.7) | 98.2 / 91.4 | 98.0 / 91.8 |
| | **CAST**, our grid point (L4, sim < 0.052, F1 .92) | **81.3 / 3.0** | **83.3 / 2.4** |
| | CAST, the paper's point (L7, sim < 0.048) | 97.8 / 73.8 | 83.3 / 2.4 |

The replication works. On Qwen the paper's own condition point transfers to our vectors. On Hermes the paper's
point fires on too many prompts at layer 7, but our grid point, with a similar threshold at layer 4, reproduces
the paper's numbers.

## Sec 5: category conditions (500 test prompts per category, refusal %)

Hermes 2 Pro (the paper's Fig 5–7 model):

| rule | base | sex | lgl | hat | crm | hth |
|---|---|---|---|---|---|---|
| no steering | 3.0 | 5.2 | 6.2 | 24.6 | 11.6 | 11.8 |
| if sex | 11.0 | **82.6** | 13.8 | 38.0 | 17.6 | 28.0 |
| if lgl | 11.2 | 7.8 | **71.0** | 25.0 | 25.6 | 45.0 |
| if hat | 8.6 | 16.4 | 8.6 | **92.2** | 16.2 | 13.0 |
| if crm | 9.0 | 7.4 | 16.2 | 26.4 | **88.8** | 15.6 |
| if hth | 10.8 | 14.0 | 29.0 | 25.8 | 14.0 | **85.0** |
| if hat ∨ lgl ∨ sex | 24.2 | **84.6** | **75.6** | **93.4** | 34.6 | 56.4 |
| if ¬hth (health-only assistant) | 84.0 | 81.2 | 64.4 | 94.8 | 90.4 | **18.2** |
| if ¬crm | 85.8 | 87.8 | 77.2 | 94.2 | **15.6** | 87.6 |

- Single conditions, OR composition and domain constraints by negation all behave as the paper's Fig 5a / 6a / 7a
  show.
- The comparison direction we find (<, >) matches the paper's Table 3 for all five categories. The layers differ,
  except hate (L7).
- Legal is the weakest category (F1 .83 Hermes, .70 Qwen), as the paper reports for constraining.
- Qwen 1.8B shows the same pattern (target 91–96% except legal 89% with heavy leakage). Its base model already
  refuses about 50% of hate and crime prompts.

## Pooling: an undocumented detail the replication depends on
Our first implementation averaged each text over its own real tokens, which looks like the "correct" version of
the release's code. The release instead averages over the padded batch, and pairs share a batch. The difference
decides whether the method works:

| | masked pooling (ours, first try) | release pooling |
|---|---|---|
| PC1 share of pair-centred variance, harmful condition | 0.97 (Qwen L6–13), ~1.00 (Hermes L1–14) | 0.33–0.69 |
| Qwen +refusal harmless | 70.6 | 95.8 (paper 96.4) |
| Hermes Sec 5 grid search | F1 0.667 for every category (= "always on"): rules refuse everything | works (table above) |
| one neutral sentence prepended to harmless prompts → harmful gate fires | Qwen 72%, Hermes 97% | Qwen 9%, Hermes 6% |

The mechanism: the first token's attention-sink state is huge. With per-text averaging it enters each pooled
state with weight 1/(own length), so PC1 becomes a length direction. With padded-batch averaging both members of
a pair get the same weight, and pair-centring cancels it.

## Finding: the harmful gate is still partly a length detector
In the paper's Sec 4 data, harmful prompts are longer than harmless ones (46 vs 30 tokens with the chat template).
Length alone separates the two test classes with AUC 0.91. The release-pooling gate reaches 0.93 on Qwen and 0.96
on Hermes. It is length-sensitive, and the content of the prompt is unchanged:

| neutral filler sentences prepended to the 500 harmless prompts | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| Qwen gate fires | 0.6% | 8.6% | 8.0% | 24.8% | 48.8% |
| Hermes gate fires | 0.8% | 5.8% | 24.8% | 19.0% | 79.8% |

On Qwen the gate fires on 0% of the 18 harmful test prompts under 30 tokens, against 100% of those over 60.
Appending the filler gives similar numbers (`length_gate.json`). The safety reading: a terse harmful request can
pass the gate, and a long, polite benign request can be refused. This is the first Stage-2 lead (docs/plan.md).
