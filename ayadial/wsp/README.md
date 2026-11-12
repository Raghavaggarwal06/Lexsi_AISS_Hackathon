# wsp: the weight-space-programming harness

A driver lets an LLM editor (Claude Fable 5.2, through the Anthropic API) try to improve the strictness knob of `CohereLabs/tiny-aya-global` by editing layer 22's MLP directly. The editor is blind: it sees only our messages. It works through a sandboxed Python kernel on this Mac and a forward-pass tool served by a GPU worker. The design follows `notes/weight-programming/HARNESS-NOTES.md` section 4, with the owner's decisions of 11 Nov 2026. The MIT Exp 8 loop it copies is described in section 1 of the same file.

Status on 11 Nov 2026: the loop is built and tested with a scripted fake editor and fake forward passes. The GPU worker (`worker.py`) is tested on this Mac's MPS with tiny jobs through the real client. **No API call has been made, Colab was not used, and nothing was committed, pushed or uploaded.**

## Files

| file | what it is |
| --- | --- |
| `driver.py` | the loop: API client, reply parsing, budgets, dollar cap, handoffs, notices, logging, resume, `--mock`, `--dry-run` |
| `kernel.py` | the editor's persistent Python kernel (server side, runs inside `sandbox-exec`) and the driver-side client (sandbox profile, scrubbed environment, time and memory limits, restarts) |
| `edits.py` | edit format and validation; shared by the kernel (fast feedback) and the driver (the check that counts) |
| `passes.py` | forward-pass request schema and validation, `StubBackend` (fake numbers, `--mock` only), `RemoteBackend` (the client of the GPU worker's file queue; its docstring is the contract) |
| `worker.py`, `WORKER-COLAB.md`, `refusal_openers.json` | the GPU worker that serves read and behave passes from the queue; how to run it on Colab (files, commands); the token ids of the first-token refusal score |
| `case.py` | builds each run's read-only case folder: weight slice (layers 18 to 22 only), the starting knob as an edit, vectors, set and format catalogue, facts |
| `build_first_message.py` | builds the first message from the results files; writes `FIRST_MESSAGE.md` and `FIRST_MESSAGE.sources.json` |
| `FIRST_MESSAGE.md` | the draft first message for the pilot config (33,894 characters), for the owner to read and approve |
| `OUR_NOTES_not_sent.md` | our diagnoses and failed attempts; the driver never sends it |
| `mock_editor.py` | the scripted fake editor for `--mock` |
| `estimate_cost.py` | cost per turn from the price table and stated assumptions (no API call) |
| `config.py`, `configs/{mock,pilot,real}.json` | every setting, with defaults; a config file overrides the defaults; `--set key.sub=value` overrides both |
| `tests/test_sandbox.py`, `tests/test_mock_run.py`, `tests/test_worker.py` | the test scripts; they write `tests/sandbox_report.json`, `tests/mock_report.json` and `tests/worker_report.json` (the worker test loads the model on this Mac) |
| `cache/tiny_aya_global_L18-22.safetensors` | the weight slice the kernel may read (781 MB, original bf16, 40 tensors), extracted once from the Hugging Face cache, plus its manifest |
| `runs/<run_id>/` | one folder per run (layout below). `runs/mocktest-20261111-153331` is the last mock test; `runs/_sandbox_test` belongs to the sandbox test |

## How to run (from `build/`, with `.venv`)

```
.venv/bin/python wsp/tests/test_sandbox.py           # sandbox checks (about 15 s)
.venv/bin/python wsp/tests/test_mock_run.py          # end-to-end mock with two simulated crashes and resumes (about 5 s)
.venv/bin/python wsp/driver.py --mock                # one mock session (fake editor, stub passes, no network)
.venv/bin/python wsp/driver.py --dry-run             # print the pilot config, the preflight list and the first message; exit
.venv/bin/python wsp/build_first_message.py          # rebuild FIRST_MESSAGE.md after the results change
.venv/bin/python wsp/estimate_cost.py                # cost per turn at high and max under stated assumptions
```

**Pilot**, once everything in "Missing before the pilot" is done and the owner has approved the first message:
```
export ANTHROPIC_API_KEY=...          # read by the SDK; never printed or written by the harness
.venv/bin/python wsp/driver.py --config wsp/configs/pilot.json --set remote.queue_dir=<folder the GPU worker polls>
```
**Real run**: the same with `--config wsp/configs/real.json` (effort `max`, `max_tokens` 128000, `dollar_cap` 50.0, `pause_at_turn` 10: the driver stops cleanly after turn 10 and prints how to resume, `--resume <run_id> --set pause_at_turn=null`). **Resume** after a crash or Ctrl-C: `.venv/bin/python wsp/driver.py --resume <run_id>`. A recorded reply is reused, never asked for (and paid for) again. Its block runs again on the fresh kernel, and the editor is told that the kernel restarted. The driver refuses to start a non-mock run while the preflight list (shown by `--dry-run`) is not empty.

## How a session works

- **Turn.** One API call returns a reply. The reply must hold a plain-English explanation and then exactly one fenced block: ```` ```python ```` runs in the kernel; ```` ```forward ```` is one pass (JSON). A reply with no block is taken as the final answer only after the harness has asked for one. Before that it gets a nudge, at most `max_nudges` times.
- **Explain every step, from turn 1.** If fewer than `min_explanation_words` words precede the block, the block is sent back unrun and no budget is used. This happens at most `max_unexplained_in_a_row` times in a row; the next unexplained block runs, with a note.
- **Format errors.** Two blocks, a block of another kind, an unclosed fence, or a reply cut at `max_tokens`: nothing is run and no budget is used. After `max_consecutive_format_errors` in a row the run stops.
- **Edits.** `save_edit` checks an edit in the kernel and writes it to `WORK/edits/pending/`. After each Python run the driver checks every pending file again, outside the sandbox, against the original weights, and only then accepts it. The authoritative edit state lives in `runs/<id>/edits/current/`, which the kernel cannot write; the kernel sees a mirror in `WORK/edits/accepted/`. The rules: only `model.layers.22.mlp.{gate,up,down}_proj.weight`; only rows (gate, up) or columns (down) of the allowed units, so at s = 0 nothing else differs from the original model; at most `edit.max_units` distinct units; a knob part (multiplied by s) only on `edit.knob_tensors`; finite values; norm caps.
- **Forward passes.** A read is forward only; a behave pass generates and judges. Each has its own budget. A request that fails validation, fails on the worker, or repeats an earlier request exactly with the same edit is not counted. The editor names prompts by set id and index. No prompt or reply text exists in the harness.
- **Notices.** A kernel restart (crash, time limit, memory limit, or a restart of the harness) is announced in the next message. A content-filter refusal (`stop_reason == "refusal"`) is logged in `refused/` and answered with a plain note. After `max_consecutive_refusals` in a row the run stops.
- **Handoff.** Every `handoff_every` turns, or once one request's input passes `handoff_max_input_tokens`, a new conversation starts. It opens with the same first message, then the counters, the accepted edit, the files in WORK, the tail of `WORK/notes.md`, and the last result. The kernel and its variables survive. The history is append-only within a conversation (Fable 5.2 rejects edited histories that carry thinking blocks): `system` is fixed, and notices are appended, never inserted.
- **End.** When both pass budgets or all Python runs are used, the harness asks for the final answer; Python stays available if runs are left. The run also ends at `max_turns`, at the dollar cap, after repeated refusals or format errors, or when the editor gives up after `max_nudges` nudges. The accepted edit at the end is the submission.

## The dollar cap

Before every call the driver computes a conservative upper bound on its cost. Every input token is priced at the highest input rate (a 1-hour cache write, $16 per million): the last request's input, plus the last reply, plus the new text at 2 characters per token. All `max_tokens` of output are priced at $40 per million. If spent + bound > `dollar_cap`, the driver stops without calling. If spent + 2 × bound > `dollar_cap`, this call becomes the last one: the message asks for the final answer, so the run never ends without one. A call that fails without a usage report is charged its full bound. Real costs come from each reply's `usage`, including cache reads and cache writes split by TTL. In practice the bound is large at `max_tokens` 64000: output alone is $2.56, and the bound reached about $4.8 per call in a simulation with the cost assumptions below. So a $15 pilot asks for its final answer once about $5.6 to $6.5 is spent: after 13 turns at 8,000 output tokens per turn (8 to 21 turns for 16,000 to 4,000 tokens). Lower `max_tokens` or raise the cap to get more turns.

## Every config value (`config.py`; defaults, then what the pilot and real configs change)

| key | default | meaning |
| --- | --- | --- |
| `run_id` | auto `<mode>-<timestamp>` | run folder name |
| `mode` | pilot | label: mock, pilot or real |
| `model` | claude-fable-5-2 | editor model id |
| `effort` | high (real: max) | `output_config.effort` |
| `max_tokens` | 64000 (mock 8000, real 128000) | cap on thinking + reply per call; calls are streamed |
| `thinking_display` | summarized | thinking summaries are logged in `reply.json` |
| `cache_ttl` | 1h | prompt-cache TTL for the system prompt, the first message and the conversation tail |
| `fallbacks` | null | server-side refusal fallback; off, because a fallback model would silently replace the editor |
| `api_timeout_s` | 900 | SDK timeout per call |
| `api_retries` | 2 | driver retries of a failed call (each re-checked against the cap) |
| `dollar_cap` | 15.0 (mock 6.0, real 50.0) | hard cap in USD |
| `prices_usd_per_mtok` | input 8, output 40, cache write 5m 10, cache write 1h 16, cache read 0.20 | Claude Fable 5.2 first-party prices; source: the `claude-api` skill bundled with Claude Code 2.1.322 (model table cached 2026-11-03) |
| `chars_per_token_floor` | 2.0 | new text is assumed to cost at least len / 2 tokens in the bound |
| `max_turns` | 400 (mock 60) | API replies, counting format errors and refusals |
| `pause_at_turn` | null (real 10) | stop cleanly once this turn's block has run; resume with `--resume <run_id> --set pause_at_turn=null` (or a later turn). It is the only key `--set` may change on a resume; the change is written to the run's `config.json` and logged |
| `handoff_every` | 40 (mock 5) | turns per conversation |
| `handoff_max_input_tokens` | 400000 | start a new conversation once one request's input passes this |
| `notes_tail_chars` | 8000 | notebook tail carried into a handoff |
| `min_explanation_words` | 20 | words before the block |
| `max_unexplained_in_a_row` | 2 | unexplained blocks sent back unrun in a row |
| `max_consecutive_format_errors` | 5 | then stop |
| `max_consecutive_refusals` | 3 | then stop |
| `max_nudges` | 3 | replies with no block while budget is left |
| `allow_early_finish` | false | if true, a reply with no block ends the run at once |
| `python_runs` | 300 (mock 30) | Python runs |
| `python_timeout_s` | 900 (mock 60) | wall clock per run; then SIGINT; after `python_grace_s` (15) SIGKILL and an announced restart |
| `kernel_memory_gb` | 10.0 (mock 6.0) | memory watchdog: physical footprint or RSS, whichever is larger, sampled every `kernel_watchdog_interval_s` (0.25) |
| `output_max_chars` | 10000 | Python output sent back (the first 60% and the last 40% are kept) |
| `work_max_gb` | 20.0 | warn when WORK grows beyond this |
| `python_executable` | .venv/bin/python | the kernel's interpreter |
| `backend` | stub (pilot/real: remote) | stub = fake numbers, mock only |
| `remote.queue_dir`, `remote.poll_s`, `remote.timeout_s` | unset, 10, 3600 | the GPU worker's file queue |
| `reads`, `behaves` | 40, 12 (mock 6, 3) | pass budgets |
| `read_max_prompts`, `read_max_hidden_layers` | 2048, 6 | read limits |
| `behave_max_prompts_per_arm`, `behave_max_arms`, `behave_max_new_tokens` | 1024, 3, 128 | behave limits |
| `weights.hf_dir`, `weights.read_layers`, `weights.slice_file` | HF cache dir, 18 to 22, `wsp/cache/...` | what the kernel may read |
| `edit.fold_units` | 6666, 1896 | the knob's units |
| `edit.census_file` | `results/round3/block3/marker.json` | dormant-unit census of layers[22] |
| `edit.exclude_units` | 1785, 4466 | the private lax pair, never offered |
| `edit.extra_units_max` | 8 | further census units allowed: 9173, 6376, 8037, 1101, 10378, 3315, 8098, 1829 |
| `edit.max_units` | 10 | distinct units that may differ from the original |
| `edit.knob_tensors` | gate_proj | where a knob part (times s) may be; gate only matches the current GGUF export |
| `edit.max_row_norm`, `edit.max_knob_row_norm` | gate 400, up 50, down 120; knob 400 | sanity caps; AyaDial's rows are about 20 (gate), 0.2 (up), 5.7 (down) and 26 (knob) |
| `start_knob` | ayadial_improved (mock: ayadial) | the editor's starting point and "current" |
| `sources.ayadial_adapters`, `.ayadial_improved_vec`, `.refusal_direction` | `runs_private/round4/adapters/fold_cap`, `cache_r5/vec_v3g.npz`, `results/day1/directions.npz` | inputs of the case builder |
| `scoreboard.*` | scales 0, 1, 1.25, 2; headline 1; budget 5; harmless rise vs current 2 (proposed); KL bar 1e-3; monotonic step 3 | the fixed scoreboard described in the first message |

## What is logged (`runs/<run_id>/`)

- `config.json`: the merged config, frozen at the start.
- `state.json`: rewritten after every step; it is what `--resume` reads.
- `driver.log`: one line per turn: kind, counters, dollars spent, edit hash.
- `first_message.md`: exactly what was sent, with its sha256 in the state.
- `case/`: everything the kernel may read.
- `sandbox.sb`: the profile in force.
- `turns/tNNN/`:
  - `message_sent.md`;
  - `reply.json` (all content blocks, including thinking summaries, plus `stop_reason`, `stop_details`, usage and wall time);
  - `reply_text.md`;
  - `block.txt`;
  - `tool_result.md`;
  - `usage.json` (this call's cost, spend so far, and the bound used for the cap check).
- `conv/cNN.jsonl`: each conversation exactly as sent, append-only.
- `tool_calls.jsonl`: every Python run (code, output, seconds, restart) and every pass (normalised request, result, edit hash).
- `requests/fwd_<k>.json`, `fwd_<k>.done.json`, `fwd_<k>.failed.json`: requests and summaries.
- `edits/current/`, `edits/history/`, `edits/log.jsonl`: every accepted or refused edit, with the reason.
- `explanations.md`: the prose before each block.
- `refused/`: content-filter refusals.
- `final_answer.md`, `summary.json`.
- `work/`: the editor's WORK folder, with its `notes.md`, saved files and pass arrays.

What is never logged: the API key, prompt or reply text of any dataset (none exists in the harness), and absolute paths in messages to the API (kernel output is rewritten to WORK, CASE, `<python>`, `<build>` and `~`).

## The sandbox (`kernel.py`, `build_profile`; tested by `tests/test_sandbox.py`, 16 of 16 checks pass)

`/usr/bin/sandbox-exec` with `(deny default)` and Apple's `system.sb` imported. On top of that the profile allows:
- exec of the Python interpreter only;
- reads of the Python runtime, the weight slice, the run's case folder and the run's work folder;
- writes only in the work folder;
- `stat()` only on those paths and their ancestors.

It denies:
- network;
- fork and exec of any other program;
- NVRAM;
- reading other processes' information, as described below.

The kernel starts with a scrubbed environment of nine variables (no API key). The tests show these are blocked: TCP, DNS and unix-socket connections; reading or listing `runs_private/` and `data/`; reading or even stat-ing `~/.zshrc`; writes to `wsp/`, `/tmp`, home and the case folder; exec of `/bin/ps`; fork. They also check the time limit (a busy loop is interrupted and the variables survive; a loop that ignores SIGINT is killed and the restart is announced) and the memory limit.

Three findings the owner should know:
1. **`sandbox-exec` alone does not stop untrusted code from reading another process's environment.** The `kern.procargs2` sysctl returns any same-user process's arguments and environment. It went through `(deny default)`, a blanket `(deny sysctl-read)`, name-filtered sysctl denies and `(deny process-info*)` each on its own. It is blocked only when `sysctl-read` and `process-info*` are both denied; the profile does that and then allows back self-info and a short list of named sysctls Python needs. The test shows the leak with that rule removed, and no leak with it: a scan of every pid from 1 to 99999 finds no readable process. If a future macOS changes this, the test will catch it.
2. **The memory limit is a watchdog, not a hard limit.** macOS refuses to lower `RLIMIT_AS` and `RLIMIT_DATA`. RSS also misses compressed pages: a 3 GB array showed 1.2 GB of RSS and 3.0 GB of physical footprint. So the watchdog reads the footprint every 0.25 s. In the test it killed a 3 GB allocation at 2.2 GB against a 1.5 GB limit: a fast allocation overshoots by up to a sampling interval's worth.
3. **What stays open.** `system.sb` keeps a few Mach services reachable (logging, preferences, directory lookups); none gives network or file access outside the profile. The kernel can read the whole venv and the interpreter's `lib` folder.

## Cost (estimates, not measurements; `estimate_cost.py`)

Pricing: Claude Fable 5.2 at $8 / $40 per million input / output tokens, cache writes $10 (5 minutes) or $16 (1 hour), cache reads $0.20. Source: the `claude-api` skill bundled with Claude Code (model table cached 2026-11-03, section "Migrating to Claude Fable 5.2").

Assumptions:
- the first message is about 9,700 tokens (characters / 3.5);
- a tool result is 2,500 tokens;
- thinking blocks stay in the history and are re-read from the cache every turn;
- every call writes its new tail with the 1-hour TTL, and the cache never misses;
- the editor writes 8,000 output tokens per turn at `high` and 24,000 at `max`. These are guesses. The only reference is the teammate's Astra run at xhigh: about 6,500 output tokens per turn (HARNESS-NOTES section 3.2).

| effort | output tokens per turn (assumed) | USD per turn, mean over 120 turns | range tried |
| --- | --- | --- | --- |
| high | 8,000 | 0.53 | 0.29 (4,000 tokens) to 0.97 (16,000 tokens) |
| max | 24,000 | 1.41 | 0.97 (16,000 tokens) to 2.27 (40,000 tokens) |

Output dominates: $0.32 of the $0.53 at `high`. One cache miss on a 200,000-token history costs $3.20 at the 1-hour write price. That is why the TTL is 1 hour: the gaps between calls include GPU passes and Python runs of up to 15 minutes. The pilot's `usage.json` files replace every assumption here.

## Missing before the pilot

1. **AyaDial-Improved's vectors.** `cache_r5/vec_v3g.npz` (m, q, w, theta0 of round 5's v3g) is not on this Mac. Neither are round 5's adapter files (`runs_private/round5/adapters/strict_install*`, `strict_knob*`) nor round 6's `cache_r6/vec_v4.npz`. They should be on the Drive mirror (`lexsi-knob/round5/cache/`, `lexsi-knob/round5/runs_private/adapters/`, `lexsi-knob/round6/cache/`). Copy the first one to `build/cache_r5/vec_v3g.npz`, or set `sources.ayadial_improved_vec`. AyaDial (round 4) is rebuilt exactly from `runs_private/round4/adapters/fold_cap`: the two units' rows agree with the construction to 2e-8, and the recovered r has cosine 1.0 with day 1's r.
2. **The GPU worker** is written (`worker.py`) and tested on this Mac only (MPS, tiny jobs; `tests/worker_report.json`). It has not run on Colab. `WORKER-COLAB.md` lists what the runtime needs and the commands. Still needed: a queue folder that both the Mac and Colab see (one per run), and a first start on Colab, whose start-up check covers all 14 formats.
3. `pip install anthropic` into `.venv` (not installed), and `ANTHROPIC_API_KEY` set in the shell that runs the driver.
4. **The owner reads and approves `FIRST_MESSAGE.md`** and the scoreboard rule in it (see "Decisions to confirm").
5. **The real API path is untested.** `AnthropicEditor` follows the skill (streaming, `get_final_message`, effort, adaptive thinking with summaries, cache breakpoints, refusal handling), but no call was made. The first pilot turns should be watched.

## Missing before the real run

1. **A sealed scoreboard set.** Exam B is spent. A third set must be built and hashed before the run, and there is little unused published data left.
2. **The scoring script.** It scores the final edit at s = 0, 1, 1.25 and 2 against Base, Control and AyaDial-Improved, with the KL check at s = 0, the monotonicity check, and the quality checks (chrF, XQuAD). It is not written.
3. **Settings from the pilot.** Set the real dollar cap from the pilot's measured cost per turn, and decide whether `max` is worth it over `high` (the skill says gains above `high` are uneven).
4. **The refusal rate.** Look at the pilot's refused turns. The task is about harm and refusal, and Fable 5.2's classifiers may decline.
5. **Export.** An accepted edit must still pass the hook/weights parity bar and the GGUF export (`knob_tensors` = gate only keeps the knob a gate_proj LoRA).

## Decisions taken while building, for the owner to confirm

1. **The first message gives facts only** (owner's change of 11 Nov). It has no diagnosis, no "we believe", no suggested direction, no example edit and no ranking of weaknesses. Every development group is shown. Our diagnoses and failed attempts are in `OUR_NOTES_not_sent.md`.
2. **The scoreboard text** (first message, "HOW YOUR EDIT WILL BE SCORED") is my draft. It is built from round 5's "holds" rule, the build/CLAUDE.md parity and monotonicity rules, and round 6's quality checks. The limit "no harmless group more than 2 points above AyaDial-Improved" and the definition of "beats" are mine.
3. **Allowed units.** The two fold units plus the next 8 units of the round 3 census, skipping the lax pair 1785 and 4466 (private side): 9173, 6376, 8037, 1101, 10378, 3315, 8098, 1829. At most 10 distinct units.
4. **The knob part only on gate_proj**, so an accepted edit still exports as today's install + knob LoRA pair. Norm caps are 400 (gate), 50 (up), 120 (down) and 400 (knob): loose sanity limits, about 20 times AyaDial's gate rows.
5. **No refusal fallback.** The skill recommends opting into server-side `fallbacks`, but its targets (Opus 4.8, Opus 5) would silently replace the editor. Refusals are logged and answered with a note instead.
6. **What the kernel gets.** A weight slice holding only layers 18 to 22, so the model cannot be run in the kernel. No tokenizer, since no text exists. An `mlp22()` helper that computes layer 22's MLP on inputs returned by reads.
7. **Editor-written probe texts are not allowed.** Reads and behave passes name catalogue sets only. `"n"` takes a fixed shuffled order per set. Behave results are split by "ids already read" and "never read".
8. **Cache TTL of 1 hour** rather than the 5-minute TTL with keep-alive calls that the skill suggests for Fable 5.2. The 1-hour TTL is simpler, and gaps of up to 15 minutes are expected.
9. **The final answer must be plain text** (no JSON block), because a reply may hold only one block of kind python or forward. Ending with the knob unchanged is stated to be acceptable.
10. **The dollar-cap bound is the true worst case**: every input token priced as a 1-hour cache write, and full `max_tokens` of output. This costs pilot turns (see "The dollar cap").
11. **Handoff also on size** (400,000 input tokens), not only every 40 turns. At `max` effort the history may approach the 1M context within 40 turns if thinking blocks stay in the history.
