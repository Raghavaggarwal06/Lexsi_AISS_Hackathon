# The GPU worker (`wsp/worker.py`) on Colab

The worker serves the driver's forward passes through a queue folder that both machines see. The driver (on the Mac) writes `fwd_<k>.json`; the worker (on a Colab GPU) claims it, runs it, and writes the arrays or labels and `fwd_<k>.done.json`. The protocol is in the docstring of `passes.RemoteBackend`. Nothing the worker writes to the queue holds prompt or reply text. Replies it keeps go to a private folder outside the queue.

## Bootstrap

**What has to be on the runtime.** The 22 files below, under `/content/build/` with the same relative paths as in `build/` on the Mac. `python wsp/worker.py --preflight` checks every one of them, and `--print-bundle-list` prints the same list. Each was kept only after reading the code that uses it.

| file | why the worker needs it |
| --- | --- |
| `wsp/worker.py`, `wsp/edits.py`, `wsp/case.py`, `wsp/config.py` | the worker; the edit format and `state_hash`; the set catalogue (the same function the driver uses) and the knob construction |
| `wsp/refusal_openers.json` | token ids of the first-token refusal score (built on the Mac by `--build-openers`; ids only) |
| `scripts/r4lib.py`, `scripts/r5_robust.py`, `scripts/r6_guard.py` | read as source with `ast`, never imported: model and judge ids, `MAXLEN`, special tokens, the six developer preambles |
| `scripts/r5lib.py` | read as text by `case.fold_constants`: alpha, kappa, beta, gain, the knob units |
| `scripts/05_score.py`, `scripts/r5_degcheck.py` | the judge-output parser and `is_degenerate5` |
| `results/round5/tables/dev_main.json`, `results/round5/lab/v3g.json`, `results/round6/run/d_xd6.json`, `results/round6/robust/rb_v4.json` | the catalogue of the 71 sets, and the member sets of the composite groups |
| `results/round5/robust/suites.json` | the format pool: its 200 request ids, in order |
| `results/round5/labels/base_own.json` | base labels that pick the earlier turns of the multi-turn formats |
| `results/round6/robust/positions_v3g.json` | the start-up reference |
| `results/day1/directions.npz` | r, for the fire measure |
| `runs_private/round6/sets.jsonl` | **private.** Every prompt the catalogue names. |
| `runs_private/round4/gen/base_own.jsonl` | **private.** The base replies used as earlier turns. |
| `cache_r5/vec_v3g.npz` | AyaDial-Improved's m, q, w and theta0, for the start-up check |

**Checked and not needed:**
- `runs_private/round5/sets.jsonl`: round 6's file holds every round 5 row unchanged (compared set by set).
- `runs_private/round6/format_requests.jsonl`: the pool's ids, in the same order, are in `suites.json`.
- `runs_private/round4/adapters/fold_cap`: every job carries the starting knob, and the start check rebuilds v3g from `vec_v3g.npz`.
- The round 5 and 6 `rb_*.jsonl` suites: the worker renders the 14 formats itself, and all 200 requests fit in every format, as in those suites.
- `wsp/cache/` (the kernel's weight slice).
- `results/round3/block3/marker.json`: the client sends the allowed units in each job.

**Also needed:**
- A GPU runtime.
- A Colab secret `HF_TOKEN` from an account that accepted the licence of `CohereLabs/tiny-aya-global` (gated). `Qwen/Qwen3Guard-Gen-0.6B` is not gated.
- A `transformers` version with Cohere2, the `dtype=` argument and `logits_to_keep`. The Mac test used 5.22.2 with torch 2.15.0.
- Two Drive folders that the Mac also sees through Google Drive for desktop:
  - the queue, one folder per run, named after the run id;
  - a private folder outside it.

**Commands.** On the Mac, once per bundle, from `build/`:
```
.venv/bin/python wsp/worker.py --preflight
zip -q colab/wsp_worker_bundle_private.zip $(.venv/bin/python wsp/worker.py --print-bundle-list)
```
The zip holds prompt text. Its name matches `colab/*_private.zip` in `.gitignore`. The owner copies it by hand to the private Drive folder, for example `MyDrive/lexsi-knob/wsp/`.

On Colab, first cell:
```python
from google.colab import drive, userdata
import os
drive.mount("/content/drive")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
RUN = "pilot-01"                                   # the driver's --run-id
Q = f"/content/drive/MyDrive/lexsi-knob/wsp/queue/{RUN}"
PRIV = "/content/drive/MyDrive/lexsi-knob/wsp/private"
```
Second cell:
```
!pip install -q "transformers==5.22.2"
!mkdir -p /content/build && cd /content/build && unzip -qo /content/drive/MyDrive/lexsi-knob/wsp/wsp_worker_bundle_private.zip
!cd /content/build && python wsp/worker.py --preflight
!cd /content/build && nohup python wsp/worker.py --queue "{Q}" --private "{PRIV}" > /content/worker.out 2>&1 &
```
Third cell, repeat it until the log says `ready`. The start-up check runs about 2,800 forward passes, so this takes a few minutes:
```
!tail -n 25 /content/worker.out; cat "{Q}/worker_info.json" | head -c 1500
```
On the Mac, start the driver with the same run id and the same folder as Drive for desktop shows it:
```
.venv/bin/python wsp/driver.py --config wsp/configs/pilot.json --run-id pilot-01 \
  --set remote.queue_dir="$HOME/Library/CloudStorage/GoogleDrive-<account>/My Drive/lexsi-knob/wsp/queue/pilot-01"
```
To stop the worker, create the file `{Q}/STOP` (`!touch "{Q}/STOP"`), or kill its PID. A job the worker had claimed and not finished is run again the next time a worker starts on that queue.

## What the worker does

- **Start.**
  - Loads the model once: bf16 on CUDA, fp16 on MPS, bf16 on CPU.
  - Checks that layer 22's MLP in memory equals the checkpoint, then records its hash.
  - Runs the reference check: AyaDial-Improved, rebuilt by `case.py` from `vec_v3g.npz`, at s = 1, on the 200 format-pool requests. For each format it compares four numbers with `results/round6/robust/positions_v3g.json`:
    - the share of harmful prompts whose two units write more than 0.5 r at the reply-start token;
    - the same share for harmless prompts;
    - the mean z on harmful prompts;
    - the mean z on harmless prompts.
  - Tolerance: each fire share within 2 prompts, each mean z within 0.02, and every prompt must end with the reply-start token.
  - It checks all 14 formats on CUDA and `default` elsewhere (`--check-formats`). If the check fails, it writes `worker_info.json` and exits without serving.
  - It loads the judge, Qwen3Guard-Gen-0.6B, when the first behave pass arrives.
- **Sets and formats.**
  - Id k of a set is the k-th row of that set in `runs_private/round6/sets.jsonl`.
  - A composite group (for example `adversarial_harmful`, or the Aya harmless groups) is its member sets from `dev_main.json`, concatenated in that order.
  - The format pool follows `suites.json`.
  - Formats are rendered as `r5_robust.build` and `r6_guard.render_fmt` did, with the same seeds (`1000 + i`, `8000 + i`) and the same earlier-turn pools. For any other set, i is the id inside the set, and a request is never drawn as its own earlier turn.
  - Prompts are cut from the left at 1536 tokens, as `r4lib.enc` did. The cut is counted.
- **Weights.**
  - original: no change.
  - current: the job's starting knob at s.
  - edited: the driver's edit at s.
  - Each is written as install + s × knob (float32, then the model's dtype) into the rows or columns of its units.
  - The worker restores the weights after each arm and checks the restored rows exactly. After each job it checks the hash of all three tensors.
  - It recomputes `edits.state_hash` of both edits it received and refuses a job whose hashes differ from the driver's.
- **Read.**
  - Hooks:
    - `hidden_L` is the input of `layers[L]`;
    - `mlp22_in` is the input of `layers[22].mlp`;
    - `mlp22_units` is the input of `down_proj` at the requested units.
  - The arrays are float16, at the last K positions, with the marker last. `mlp22_unit_ids` is int64. `meta.json` lists rows, token counts and definitions.
  - The summary gives, per set: z (the starting knob's w · mlp22_in at the marker), the fire rate, and the first-token score. The format pool also gets a split by harmful and harmless.
  - fire = Σ a_j (d_j · r) / (r · r) at the marker, over the units of the weights used. original therefore has fire 0.
  - first_token = log P_R − log(1 − P_R) at the reply-start token, where P_R is the probability of a refusal-opener token. This is round 4's refusal score from `scripts/r4_wang.py`, per prompt. The rule gives one token, "I'm", which starts 54.1% of the base model's refused development replies.
- **Behave.**
  - Greedy generation with `r4lib.generate`'s settings, batch-major, so every arm sees exactly the same batches.
  - The judge as in `r4lib.judge`: (request, reply) pairs. `is_degenerate5` comes first, and a degenerate reply skips the judge.
  - Returns, by arm and set:
    - refusal, unsafe and degenerate %;
    - the counts read before and never read;
    - each rate on both halves.
  - For every pair of arms, on all prompts, on the never-read prompts and by set:
    - the unsafe and refusal differences in points, with 95% bootstrap intervals over prompts (2,000 draws);
    - flips in both directions.
  - `beh_<k>.jsonl` holds labels by set and id.
  - `<private>/<run_id>/fwd_<k>.replies.jsonl` holds the request, the reply and the raw judge output.
- **Batching.** Batches are filled to a token budget, set by `--read-tokens`, `--gen-tokens`, `--read-batch`, `--gen-batch` and `--judge-batch`. On out-of-memory a batch is split in half and retried. Defaults:
  - CUDA: 120,000 tokens and 256 prompts per read batch, 90,000 tokens and 160 prompts per generation batch, 64 per judge batch, as in rounds 5 and 6.
  - MPS: 8,192 / 32, 4,096 / 16, and 16.
- **Restarts.** Every output is written under a temporary name and then renamed. `done.json` comes last, with a manifest of file sizes, so the client waits until a synced folder has delivered everything. On start, a job claimed and not finished is run again, and its partial outputs are removed first.

## Measured on the Mac (MPS, fp16; `wsp/tests/test_worker.py`, 11 Nov 2026)

Source: `wsp/tests/worker_report.json` (16 of 16 checks pass) and `wsp/runs/_worker_test/worker_stdout.log`. During the test the Mac was swapping heavily: 13.8 of 14.3 GB of swap was in use, from other programs. These times measure that state, not the chip.

| what | measured |
| --- | --- |
| model load (fp16, MPS) | 14.9 s |
| start-up check: 200 prompts of about 390 tokens, forward to layer 22 | 181.1 s |
| read job, 16 prompts, every return field, full forward | 17.3 s to 119.4 s (five jobs: 61.2, 44.6, 17.3, 119.4; 62.1 s for 4 longer prompts) |
| behave job, 3 arms × 12 prompts × 24 new tokens, plus judge | 94.6 s in all: 67.9 s generating (0.47 s per decoding step for a batch of 6; 0.079 s per generated token), 7.1 s judge load, 11.2 s judge |

At this pace the Mac is not a usable worker. One full behave pass (1,024 prompts × 3 arms × 128 tokens) would take hours. A 2,048-prompt read would take about half an hour or more.

## Notes for the owner

- **Drive sync is the slow part.**
  - A read with 6 hidden layers × 16 positions × 2,048 prompts is about 800 MB of float16. Prefer `"positions": "marker"` and fewer layers.
  - Delete the `fwd_*` folders of finished runs from the queue.
- **One queue folder per run.** The client also refuses answers whose job key is not its own, so a stale answer under the same k is never read.
