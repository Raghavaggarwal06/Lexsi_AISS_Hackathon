# wsp-safety: Weight-Space Programming, reimplemented, aimed at safety tasks

Reimplementation of *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*
(Takehi, Teoh, Zhang, Khattab, 2026-11-02; https://rikiyat.github.io/blog/weight-space-programming/). Here the method is
applied to safety tasks and scored.

**Model:** `gpt-6-astra` (the paper's "Astra"), run through the Codex CLI on a **ChatGPT subscription**
(`codex login status` → "Logged in using ChatGPT"). There's no API key and no per-token billing; runs count against the
plan's included Codex usage. Before every turn the driver reads the account's live limits from Codex's session logs
(`wsp/codex.py: guard_usage`). It sleeps until the window resets once the 5 h or weekly window reaches 95 %
(`WSP_USAGE_MAX`), and it stops if the credit balance drops, so it never spends purchased credits.

## Status (2026-11-09)
- Harness rebuilt and tested end to end: Python sandbox runs, attention tool, nudges, last-run notice, final-answer
  parsing, scoring, causal patch check. Plumbing test: `gpt-6.1-terra`, low effort, 6 runs.
- Exp 7 models rebuilt to match the paper. The release IDs, A/B answer key, step counts (339/336/342) and checkpoint
  size are identical to the paper's. Behaviour on held-out facts the model knows: sycophants follow 100 % of wrong
  guesses, honest models 0–0.7 % (paper 0.7–2.8 %), plain accuracy 97–99 %.
- First Astra replication (attention-tool round, `pair_fee168`, xhigh) is in progress. After 10 of 20 runs it puts
  P(B is the sycophant) = 0.90 (truth: B). It independently found head L10H2 attending to the user's guess, the head in
  the paper's Figure 2.
- Next: the other two pairs, then the safety tasks in `docs/plan.md` (backdoor-trigger audit, sandbagging audit,
  null-pair calibration, weight-edit repair).

## What exists upstream and what we rebuilt

The authors released logs, not a harness: https://github.com/RikiyaT/lm_understand-logs (cloned to `refs/`, on scratch).
It contains:

| upstream | used here as |
|---|---|
| `harness/codex_config.toml` (all Codex tools off) | `harness/codex_config.toml` (model set to gpt-6-astra), passed as `-c` overrides |
| `harness/instructions.md` ("Answer the user's request.") | `harness/instructions.md` |
| `harness/sandbox.py` (bubblewrap) | **rewritten**: `wsp/sandbox.py` (bwrap fails on Scholar: `pivot_root: Invalid argument`) |
| exact `codex exec` command lines (every turn's `run.json`) | `wsp/codex.py` |
| *no driver loop* | **rebuilt** from the logged messages: `wsp/loop.py` |
| prompts, preludes, datasets, training notes per experiment | `tasks/<task>/` |
| Exp 7 training/eval/release scripts | **reimplemented** in `tasks/sycophancy/build.py` (the copied scripts were not run) |

### `wsp/codex.py`: one turn
`codex exec [resume <thread>] --strict-config --skip-git-repo-check --json -m gpt-6-astra -c model_reasoning_effort=…
-c model_context_window=872000 -c model_auto_compact_token_limit=1000000 -c model_instructions_file=harness/instructions.md
-o answer.md -c model_reasoning_summary="detailed" -` plus every key of `harness/codex_config.toml` as a `-c` override.
The user's `~/.codex` is never edited. Each turn writes `prompt.txt`, `events.jsonl`, `answer.md`, `run.json` and
`stderr.log`. The **audit** marks a turn `tool_free` only if its event stream holds nothing but `agent_message`,
`reasoning` and Codex `error` notices (the "Code Mode is unavailable" notice, the same one the paper logs). Usage-limit
errors are waited out (15 min steps), and other failures are retried. A reply is **every** non-empty `agent_message` of
the turn, joined in order. The model sometimes splits one reply into several messages (explanation + code, then
"Please run the Python block above"), and `-o` alone would keep only the last one.

### `wsp/sandbox.py`: the model's Python
Per run: `unshare -r -m -n -p -f` (user, mount, net, pid namespaces), then a tmpfs root with read-only `/usr` and the
sandbox venv, a fresh `/proc` and minimal `/dev`, `/case` (read-only) and `/work` (read-write, kept between runs), then
`chroot`. The environment is cleared, and there are an address-space limit, a timeout and an output cut. `/home`,
`/scratch` (except the two read-only env folders) and `/depot` are invisible. There is no network. The script runs as
PID 1 alone. The venv for Exp-7-style tasks is numpy 2.2.6 + stdlib only (`/scratch/scholar/pande1/wsp/envs/sandbox-np`).

### `wsp/loop.py`: the conversation
Reproduces the harness messages word for word: `OUTPUT of run k of at most N:` + output; the last-run notice; the
nudge ("You have used k of N runs. Before finishing, test at least one hypothesis …"); and the format retry ("Your final
answer could not be read (…)"). A reply with a ```python (or other enabled tool) block is one run; a reply without one
is the final answer. `state.json` is saved after every turn, so a killed driver resumes where it stopped.

## Tasks

| task | status | what it tests |
|---|---|---|
| `tasks/sycophancy` | models rebuilt; Astra validation running | paper Exp 7: which of two Gemma 3 270M fine-tunes is the sycophant, and where it lives |

Scoring: `tasks/sycophancy/score.py` (A/B identification, P(correct)) and `tasks/sycophancy/patch_eval.py`. The latter is
a causal check of the named components: copy them honest → sycophant and measure how much of the wrong-guess-following
gap closes, against the same number of largest-change and random components.

## Storage and paths
Paths are currently specific to the Scholar cluster (user `pande1`). `runs/`, `refs/` and `models/` are symlinks to
`/scratch/scholar/pande1/wsp/` and are not in git (run logs, the paper's 356 MB log repo, model checkpoints). The
sandbox venv and the Gemma base model are also on scratch (`wsp/sandbox.py: ENVS`, `tasks/sycophancy/build.py: BASE`,
`WSP_GEMMA_BASE`, `WSP_SYC_ROOT`). To run elsewhere, set those paths and recreate the sandbox venv:
`uv venv <envs>/sandbox-np --python 3.12 && uv pip install --python <envs>/sandbox-np/bin/python numpy==2.2.6`.

## Attribution
`tasks/sycophancy/dataset/` (facts and splits), the two `prompt_*.txt` files, `tasks/sycophancy/prelude.py` and
`harness/codex_config.toml` + `harness/instructions.md` are taken from the authors' release
https://github.com/RikiyaT/lm_understand-logs (logs/data CC BY 4.0, scripts MIT). Everything else here is new code.

## Running
Use the `mech-interp` env's Python (3.11, torch 2.14, transformers 5.17) as the driver interpreter.
```bash
PY=~/.conda/envs/mech-interp/bin/python
$PY scripts/smoke_codex.py low                                   # one cheap tool-free turn + resume
CUDA_VISIBLE_DEVICES=0 $PY tasks/sycophancy/build.py             # 3 pairs x {honest, sycophant}, truth, A/B release
$PY tasks/sycophancy/run_case.py --round attention_tool --case pair_fee168
$PY tasks/sycophancy/score.py runs/sycophancy/attention_tool
```
Slurm on Scholar: 4 h wall, at most 2 running jobs per user (the OnDemand session counts). Conversations resume from
`state.json`, so a long round is a chain of 4 h jobs.
