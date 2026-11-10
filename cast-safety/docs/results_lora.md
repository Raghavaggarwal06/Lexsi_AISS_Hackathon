# CAST → LoRA: compiling conditional steering into weights (2026-11-09)

**Question.** Can a LoRA adapter on the plain chat model reproduce what CAST does with inference-time hooks:
refuse harmful prompts, answer harmless ones, with no condition check and no steering at inference?

**Answer.** Yes, on both models. The adapter matches CAST's refusal rates within 1.5–4.6 points and makes the same
refuse/answer decision on 88–97% of test prompts. With length-invariant training data it also removes CAST's
length bias, and it does not open a filler jailbreak.

## Method (`scripts/distill.py`, `cast/lora.py`)
- **Teacher**: the Sec 4 CAST configuration (`runs/<model>/sec4`: release-pooling vectors, grid-searched gate,
  the paper's behaviour layers and α).
- **Training prompts**: the Sec 4 *train* prompts: 4,039 unique harmful and 698 unique harmless (the release's
  4,050 harmless train items repeat 698 Alpaca prompts). The teacher answers each one, greedy, ≤ 64 tokens. Its
  gate decision is stored. The test prompts are never used.
- **Student**: the same weights plus a LoRA on q/k/v/o/gate/up/down of every decoder layer (rank 16, α 32), with
  no hooks.
- **Loss**: token-level KL(teacher ‖ student) over the teacher's response tokens. The teacher here is the
  adapter-free model with the refusal vector added at every position (exactly what CAST does during generation)
  if its gate fired on that prompt, and the unmodified model if it did not. So the adapter learns both halves:
  refuse when the gate fires, change nothing when it does not.
- **Length-invariant variant (`--augment`)**:
  - 3,300 extra harmless Alpaca prompts (yahma/alpaca-cleaned, none in any test set), answered by the CAST teacher;
  - 3,000 copies of training prompts (1,500 harmful, 1,500 harmless) with 1–8 neutral sentences prepended or
    appended, whose teacher decision is the gate's decision on the **unpadded** prompt;
  - the training filler sentences are disjoint from the test filler sentences.
- Everything runs in bf16, base / CAST / LoRA alike: Qwen 2 epochs, batch 16, lr 2e-4; Hermes 1 epoch, batch 4,
  lr 1e-4, with gradient checkpointing. Training takes 5–12 minutes on one A30.

## Results: Sec 4 test (450 Sorry-Bench harmful / 500 Alpaca harmless, refusal %)

| model | adapter | params | base | CAST hooks | **LoRA** | same decision as CAST (harmful / harmless) |
|---|---|---|---|---|---|---|
| Qwen 1.5 1.8B | r16 | 15.0M | 49.1 / 3.6 | 88.7 / 4.2 | **93.3 / 5.8** | 92.7 / 94.4 |
| | r2 | 1.9M | | | 95.1 / 9.0 | 90.4 / 91.6 |
| | r16, length-invariant | 15.0M | | | **90.7 / 5.4** | **94.0 / 96.8** |
| Hermes 2 Pro 8B | r16 | 41.9M | 22.2 / 2.8 | 80.7 / 3.0 | **82.2 / 3.2** | 88.2 / 97.0 |

Only 3–30% of responses are word-for-word identical to CAST's. The adapter reproduces CAST's decisions and its
style of refusal, not its exact text. Rank 2 already carries the rule, with a little more over-refusal.

## Length bias: harmless test prompts with k neutral sentences prepended (held-out filler, refusal %)

| | k = 0 | k = 2 | k = 8 |
|---|---|---|---|
| Qwen CAST hooks | 4.6 | 10.6 | 47.6 |
| Qwen LoRA r16 | 6.0 | 22.2 | **67.2** (amplifies the bias) |
| Qwen LoRA r2 | 9.2 | 19.8 | 51.6 |
| **Qwen LoRA r16, length-invariant** | 6.0 | **6.6** | **4.8** (bias removed) |
| Hermes CAST hooks | 2.6 | 23.2 | 72.6 |
| Hermes LoRA r16 | 3.2 | 18.6 | 19.4 |

## Safety checks
Harmful test prompts with 8 neutral sentences prepended (refusal %). Filler must not become a way around refusal:

| | base | CAST hooks | LoRA |
|---|---|---|---|
| Qwen, LoRA r16 | 44.7 | 97.1 | 100.0 |
| Qwen, LoRA r16 length-invariant | 44.7 | 97.1 | **94.0** |
| Hermes, LoRA r16 | 38.0 | 99.1 | **93.6** |

Harmful refusal by prompt length (Qwen; the first bin has n = 18):

| tokens | < 30 | 30–40 | 40–60 | ≥ 60 |
|---|---|---|---|---|
| base | 61.1 | 64.7 | 41.1 | 21.0 |
| CAST hooks | 61.1 | 85.3 | 91.1 | 100.0 |
| LoRA r16, length-invariant | 55.6 | 87.9 | 94.4 | 98.4 |

Short harmful prompts remain a gap. The CAST gate does not fire on them, so the distilled adapter cannot learn
to refuse them. Fixing that needs a better teacher decision, not a better student.

## What this means
- A conditional steering rule (one cosine check and a set of added vectors) compiles into an ordinary weight
  delta of a few million parameters. The result needs no inference hooks and can ship as a normal adapter.
- Distillation copies the teacher's decision boundary, including its flaws (the length bias, missed short
  prompts). Once the rule lives in the training data, it can be corrected there: deciding the teacher's action on
  the content-only prompt gave a length-invariant adapter without losing harmful refusal.
- For the WSP track: these adapters are known, small weight edits with known behaviour. They are natural targets
  for "audit the weights" tasks, such as finding the condition this LoRA refuses on.
