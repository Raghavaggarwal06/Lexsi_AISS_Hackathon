# cast-safety: Conditional Activation Steering, reimplemented

Reimplementation of *Programming Refusal with Conditional Activation Steering* (CAST; Lee, Padhi, Natesan
Ramamurthy, Miehling, Dognin, Nagireddy, Dhurandhar; ICLR 2025 spotlight; https://arxiv.org/abs/2409.05907).
It is the first step toward the safety experiments in `docs/plan.md`. All compute is local, on GPU, with no paid
APIs.

## Status (2026-11-09)
**The replication works on both paper models**; full tables are in `docs/results_replication.md`. Sec 4 CAST
(harmful / harmless refusal %): Qwen 1.5 1.8B **88.9 / 3.8** (paper 90.7 / 2.2); Hermes 2 Pro **81.3 / 3.0** (paper
83.3 / 2.4). Sec 5 single categories, OR rules and domain constraints by negation reproduce the paper's figures on
Hermes, with legal the weakest category, as in the paper.
- The result depends on an undocumented pooling detail. See "Faithfulness".
- **CAST compiles into a LoRA** (`docs/results_lora.md`). A plain-model adapter distilled from the CAST-hooked
  model matches CAST with no hooks: Qwen 90.7 / 5.4 vs CAST 88.7 / 4.2, Hermes 82.2 / 3.2 vs 80.7 / 3.0, with the
  same decision on 88–97% of test prompts. Training on length-invariant teacher data removes CAST's length bias
  (harmless + 8 filler sentences: 4.8% refusal vs CAST 47.6%) without opening a filler jailbreak.
- First safety lead: the harmful gate is partly a length detector. Eight neutral sentences prepended to harmless
  prompts make it fire 49% (Qwen) / 80% (Hermes) of the time. Short harmful prompts pass it.

## Method in one paragraph
A **behaviour vector** v_l (here: refusal) and a **condition vector** c_l (here: "harmful", or a category such as
"legal opinion") are taken per layer as the first principal component of pair-centred contrast activations. At
inference, the prompt's mean hidden state h at one condition layer gives sim = cos(h, tanh(P_c h)), with
P_c = c cᵀ / c·c. If sim passes a threshold θ (direction < or >), α·v_l is added to the residual stream at the
behaviour layers for every later token. θ, the layer and the direction come from a grid search that maximises F1 on
the training prompts. Conditions combine with OR/AND, and flipping the direction steers on the complement
("respond only to health questions").

## Code
| file | what |
|---|---|
| `cast/model.py` | model loading and chat formatting, following the release's conventions |
| `cast/data.py` | the paper's released contrast and test data (read from `refs/activation-steering/docs/demo-data`) |
| `cast/vectors.py` | pooled hidden states → `pca_pairwise` first PC + sign fix (behaviour: suffix tokens; condition: all tokens) |
| `cast/steer.py` | `Steerer`: forward pre-hooks for condition checks (prompt pass) and behaviour addition (all passes), with rules, OR/AND and negation; batched greedy generation; offline `condition_sims` |
| `cast/search.py` | grid search of the condition point (layer, θ, direction) by F1, in the release's iteration order |
| `cast/judge.py` | refusal = `protectai/distilroberta-base-rejection-v1` says REJECTION, or one of the paper's keywords appears |
| `scripts/sec4.py` | paper Sec 4: base / +refusal / +CAST (grid point and paper point) on 450 Sorry-Bench harmful + 500 Alpaca harmless |
| `scripts/sec5.py` | paper Sec 5: 6 category condition vectors; single, OR and negated rules on 6 x 500 test prompts |
| `scripts/length_gate.py` | the harmful gate vs prompt length: AUCs, firing rate by length bin, neutral-filler intervention |
| `cast/lora.py` | minimal LoRA (fp32 A, B; B = 0 at init; switch on/off so the same weights also serve as the teacher) |
| `scripts/distill.py` | **CAST → LoRA**: CAST-hooked teacher answers the train prompts; the LoRA student (no hooks) is trained on token-level KL to the teacher; Sec 4 test + filler test. `--augment` adds extra harmless prompts and filler-padded prompts whose teacher decision is the gate's decision on the unpadded prompt |

## Faithfulness to the release (IBM/activation-steering)
Kept on purpose, because the paper's numbers and layer indices depend on them:
- A vector "for layer l" is read from `hidden_states[l+1]` (the output of decoder layer l) but is checked or added at
  the **input** of decoder layer l, a one-layer offset.
- Texts are chat-templated and then tokenised with special tokens, so Hermes / Llama 3 gets a double BOS.
- Behaviour texts are `chat(user=q) + suffix` with no assistant header. Condition texts have no generation prompt;
  inference and the grid search do.
- Generation is greedy, 50 new tokens, repetition penalty 1.1. The behaviour is also added on the prompt pass.

- **Pooling (default) is the release's exactly**: pairs batched together, right padding with id 0, and the mean
  over every position of the padded batch (condition), or over the last k positions, where k is the token length
  of the first refusal suffix (behaviour). This looks like a bug, but the method depends on it. The first token's
  huge attention-sink state then enters both members of a pair with the same weight and cancels under
  pair-centring.
  `--masked-pooling` (mean over each text's own tokens) makes PC1 a 1/length direction (97–100% of the variance).
  Qwen's +refusal then drops to 70.6% harmless refusal (paper 96.4%), and Hermes Sec 5 fails completely
  (`results/*/sec4_masked`, `sec5_masked`).

Deviations:
- Similarities are computed in fp32 (the release uses fp16).
- Generation is batched with left padding, and the pad token is not EOS. The release generates one prompt at a
  time. On a sample of unsteered prompts, batched and single-prompt greedy outputs were identical.
- Responses are decoded with `skip_special_tokens=True`.

## Storage
`results/` holds the small metric files (copied from runs). `runs/` and `refs/` are symlinks to `/scratch/scholar/pande1/cast/`, and models are in
`/scratch/scholar/pande1/models` (`CAST_MODELS`). The paper's pinned revisions are Qwen/Qwen1.5-1.8B-Chat@e482ee3,
NousResearch/Hermes-2-Pro-Llama-3-8B@8ab73a6 and protectai/distilroberta-base-rejection-v1@6558496.

## Running
```bash
PY=~/.conda/envs/mech-interp/bin/python
git clone https://github.com/IBM/activation-steering refs/activation-steering     # data
CUDA_VISIBLE_DEVICES=0 $PY scripts/sec4.py --model qwen1.5-1.8b-chat              # vectors, search, eval (~5 min)
CUDA_VISIBLE_DEVICES=0 $PY scripts/sec5.py --model qwen1.5-1.8b-chat              # needs sec4 vectors (~8 min)
CUDA_VISIBLE_DEVICES=0 $PY scripts/length_gate.py --model qwen1.5-1.8b-chat       # needs sec4 (~2 min)
CUDA_VISIBLE_DEVICES=0 $PY scripts/distill.py --model qwen1.5-1.8b-chat                  # CAST -> LoRA (~15 min)
CUDA_VISIBLE_DEVICES=0 $PY scripts/distill.py --model qwen1.5-1.8b-chat --augment --tag aug_r16 --grad-ckpt
# Hermes 2 Pro 8B: about 3x longer; one A30 (24 GB) is enough (distill: --bs 4 --epochs 1 --lr 1e-4 --grad-ckpt)
```
