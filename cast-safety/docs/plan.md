# CAST reimplementation and safety follow-ups: plan

Paper: Lee et al., *Programming Refusal with Conditional Activation Steering*, ICLR 2025 (arXiv 2409.05907).
Reference code and data: IBM/activation-steering (Apache-2.0), cloned to `refs/` (scratch). Our implementation is
new code. It follows the release's conventions, and every deviation is listed in the README.

## Stage 1: replicate the paper
| step | what | paper target |
|---|---|---|
| 1a | Sec 4: refusal vector + harmful condition vector; grid-searched condition point; base / +refusal / +CAST on 450 Sorry-Bench + 500 Alpaca | App Table 2: Qwen 1.5 1.8B 45.8/0.0 → 100/96.4 → **90.7/2.2**; Hermes 2 Pro 83.3/2.4 |
| 1b | Sec 5: five category condition vectors (sexual, legal, hate, crime, health) from the paraphrased-Alpaca data; induce refusal per category; OR composition; domain constraint by flipping the comparison (¬c_health) | Fig 5–7 (pie charts) |
| 1c | properties: duality (flip comparison = complement), modulation (θ sweep), saturation (training-set size) | Fig 4, 6 |

Models: Qwen 1.5 Chat 1.8B and Hermes 2 Pro (Llama 3 8B), the two paper models with points for both sections.
They fit on one A30 each.

## Stage 2: what CAST teaches about safety (to agree with the team)
- **Attacking the gate.** CAST's refusal depends on one cosine check at one layer. Does the condition still fire
  under jailbreak wrappers (role-play, hypothetical framing, payload splitting), translation, or a long benign
  preamble that dilutes the mean-pooled state? The prediction is that mean pooling over the prompt is the weak point.
- **Over-refusal.** Borderline-safe prompts in the style of XSTest ("how do I kill a Python process").
- **Specificity of category conditions.** Confusions between categories, and unseen categories.

## Stage 3: link to Weight-Space Programming (`../wsp-safety`)
CAST is a conditional policy written into activations. Two WSP tasks follow from it:
- **compile:** Astra writes a CAST rule (condition → refusal) directly into the weights of a small model, with no
  hooks, and we score it with the same refusal metrics;
- **audit:** a model carries a hidden conditional behaviour (a CAST rule folded into its weights, the condition being
  a topic). Astra must find the condition from the weights. This is a backdoor-style audit with known ground truth.
