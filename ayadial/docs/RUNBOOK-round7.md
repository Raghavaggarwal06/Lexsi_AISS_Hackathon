# Round 7 runbook: the SafeTune rows of the report, overnight on one Colab A100

What runs and why: `results/round7/journal.md`, J0. This file is only how to run it. Nothing here pushes, commits or uploads; the only copy outside the runtime is the private Drive folder `MyDrive/lexsi-knob/round7/`.

## 0. Before the session (once)

`colab/r7_bundle_private.zip` (built on the Mac by `python3 scripts/r7_bundle.py`) must be in `MyDrive/lexsi-knob/round7/`. It holds private prompt rows: keep it in that private folder only. Contents (paths relative to `/content/build`):

| path in the zip | what | why |
| --- | --- | --- |
| `scripts/common.py`, `05_score.py`, `r4lib.py`, `r5lib.py`, `r6lib.py`, `r4_scoreboard.py`, `r5_degcheck.py` | rounds 4 to 6's runtime code, unchanged | model loading, chat template, generation and batching, judge, labels, the scoreboard rule |
| `scripts/r5_report.py`, `r6_report.py` | rounds 5 and 6's table code, unchanged | the cluster bootstrap and clusters of the paired differences |
| `scripts/r7lib.py`, `r7gpu.py`, `r7_safetune.py`, `r7_boot.sh`, `r7_queue.sh`, `r7_pack.py`, `r7_bundle.py` | round 7 | stages, SafeTune wrappers and patches, boot, queue, pack |
| `scripts/pipeline/check_no_harmful_text.py`, `data/harmful.jsonl`, `data/xstest_unsafe.jsonl` | the leak check and its private sources | `r7_pack.py` refuses to pack results that contain prompt text |
| `results/round5/lab/v3g.json`, `frozen_declaration.json`, `sets_manifest.json`, `tables/frz_tight.json`, `labels/*.json` | round 5's fit list, test ids, summary table, label files (ids and 0/1 only) | calibration set; test id check; the report's rows; Control and Base for the paired differences |
| `results/round6/sets_manifest.json`, `exam_formats.json`, `tables/xb_tight.json`, `labels/*.json` | the same for round 6 | the same |
| `results/round7/calib.json`, `journal.md`, `RUNBOOK.md` | the calibration ids and checks made on the Mac; J0; this file | the runtime recomputes `calib.json` and refuses to run if it differs |
| `runs_private/round7/sets.jsonl` | round 6's `sets.jsonl`, unchanged (sha256 `27a89c7b8b8f…`, checked on every load) | every prompt of every suite and of the calibration set |
| `runs_private/round7/suites/xbf_<format>.jsonl` (8) | round 6's format rows with their rendered chats, from round 6's Drive mirror; ids match `exam_formats.json` | the open-problem blocks |
| `runs_private/round7/ref/base_wjx_r4.jsonl` | round 4's Base replies on the 1910 development rows | checks that this runtime reproduces the earlier rounds' Base |

## 1. Colab cells (fresh A100 runtime, Drive mounted, `HF_TOKEN` set)

**Cell 1, setup** (skip what is already done):
```python
from google.colab import drive
drive.mount('/content/drive')
import os
if not os.environ.get('HF_TOKEN'):   # gated model CohereLabs/tiny-aya-global
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
```

**Cell 2, boot** (about 10 minutes, most of it `pip`):
```
!mkdir -p /content/build && cd /content/build && unzip -qo /content/drive/MyDrive/lexsi-knob/round7/r7_bundle_private.zip scripts/r7_boot.sh && bash scripts/r7_boot.sh
```
It unpacks the bundle into `/content/build`, copies back anything an earlier round 7 session mirrored to Drive (never overwriting a local file), installs `safetune==0.1.13`, `transformers==5.22.1` (rounds 5 and 6's version) and `scikit-learn`, and runs the `calib` stage. Expected: a versions line ending in `A100-SXM4-40GB` (or 80GB), `sets.jsonl 27a89c7b8b8fc99c`, `overlap: 0 test rows by id, 0 by text, 0 shared clusters (of 12406 test rows)`, three `matches the declared hash`, `R7_CALIB DONE`, `R7_BOOT DONE`. If `pip` replaced torch or printed a resolver error, restart the runtime once (Runtime > Restart session) and run cell 2 again.

**Cell 3a, the smoke check, by hand** (about 5 minutes; watch it): it is the first run of the round 7 code on a GPU (16 calibration prompts: each SafeTune patch against SafeTune unpatched, every wrapper through a mini cell with the judge), and `pick`/`test` refuse to run unless it passed, so a failure here should be seen now, not in the morning.
```
!cd /content/build && python scripts/r7_safetune.py smoke 2>&1 | grep -v -E "it/s|s/it|Calibrate \[|Loading weights" | tail -n 8
```
Expected last lines: `P1+P2 extractor: {'identical': True, ...}`, `calibration patched vs unpatched: ...`, four mini cells, `smoke ok=True -> results/round7/smoke_cuda.json`, `R7_SMOKE DONE`. If it says `ok=False` or stops with a traceback, do not start the queue: send `results/round7/smoke_cuda.json` or the traceback back for a fix.

**Cell 3b, the queue** (starts the night; returns at once; its smoke line is skipped because 3a passed):
```
%%bash
cd /content/build
L=runs_private/round7/logs; mkdir -p $L
cat > queue7.txt <<EOF
python scripts/r7_safetune.py smoke > $L/smoke.txt 2>&1
python scripts/r7_safetune.py pick rd adasteer > $L/pick_rd_adasteer.txt 2>&1
python scripts/r7_safetune.py test A > $L/test_A.txt 2>&1
python scripts/r7_safetune.py report > $L/report_A.txt 2>&1
python scripts/r7_safetune.py pick cast > $L/pick_cast.txt 2>&1
python scripts/r7_safetune.py test B > $L/test_B.txt 2>&1
python scripts/r7_safetune.py report > $L/report_B.txt 2>&1
python scripts/r7_safetune.py test C > $L/test_C.txt 2>&1
python scripts/r7_safetune.py report > $L/report.txt 2>&1
python scripts/r7_pack.py > $L/pack.txt 2>&1
EOF
nohup bash scripts/r7_queue.sh > /dev/null 2>&1 &
echo started
```
The lines run one at a time, in this order (J0, section 4: RefusalDirection and AdaSteer on every block first, CAST on the two main sections next, the formats last). `report` runs after each test phase, so `safetune_tests.json` always holds every row whose cells are complete (a row with a missing cell is left out and stays "pending" in the report). Estimated hours: smoke 0.1, pick rd adasteer 1.5, test A 1.4, pick cast 0.9, test B 0.5, test C 0.1, report and pack 0.05: about 4.7 hours. If the CAST check finds that CAST does not fit the suites' batching, CAST runs in batches of 8 with its own Base: pick cast 3.2, test B 3.3, test C 0.8, about 10.5 hours in all; RefusalDirection and AdaSteer are complete after the first 3 hours either way.

**Cell 4, watch** (run it whenever you like; it changes nothing):
```
!cd /content/build && tail -n 12 queue7.log; echo; for f in runs_private/round7/logs/*.txt; do echo "== $f"; grep -v -E "it/s|s/it|Calibrate \[|Loading weights" $f | tail -n 3; done; echo; python scripts/r7_safetune.py status | tail -n 6; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```
Every finished cell prints one line (`<cell>: n=... seconds refused ... unsafe ...`) and is mirrored to Drive. `python scripts/r7_safetune.py <stage> --dry` lists any stage's cells with their estimated minutes, marking those already done.

## 2. What "done" looks like

- `queue7.log` ends with `END(0) python scripts/r7_pack.py ...`; every line before it shows `END(0)`.
- `results/round7/safetune_pick.json` has `methods.rd`, `methods.adasteer`, `methods.cast`, each with `setting_text`, its development numbers, the walk, and no `flag` (a flag means the weakest grid setting was over the 5-point budget; it is still reported).
- `results/round7/castcheck.json` has `decision` (alpha reaches the forward pass and changes replies or not; suite batching or batches of 8).
- `results/round7/safetune_tests.json` exists, with `settings` and rows under `first` and `second` for the three methods; its `meta.rows_written` gives 55 rows for each method on each test (every row of the two summary tables; the report's 12 blocks use 20 of them).
- `results/round7/leak_check_last.txt` ends with `0 hits`; Drive has `MyDrive/lexsi-knob/round7/r7_results_<HHMM>.zip`.
- Checks to read in the morning: `smoke_cuda.json` (`ok: true`), `safetune_pick.json` `base_check` (share of Base replies identical to round 4's on the 1910 development rows; round 5 found 1.0 after a runtime change), and `results/round7/run/cells.json` (`oom_splits` should be empty everywhere).

## 3. If the runtime is recycled, or a line fails

- **Recycled runtime:** run cells 1, 2 and 3 again, unchanged. Boot restores from `MyDrive/lexsi-knob/round7/` (labels, replies in chunks of 768, the judge cache, the calibration states, the pick and check files); every stage skips what is finished: `smoke` if `smoke_cuda.json` is ok, `pick` for every method already in `safetune_pick.json`, `test` for every cell whose label file covers all its rows, and a half-generated cell restarts at its last finished chunk of 768. The calibration is never recomputed once its state file exists, so the picks and the tests keep using the same fitted state.
- **`smoke` not ok:** `pick` and `test` refuse to run (the queue moves through them quickly). Read `smoke_cuda.json`: which comparison differs. Do not set `R7_SKIP_SMOKE=1` without reading it.
- **Out of memory in generation:** the batch is split in two and retried automatically; the split is logged and recorded in `run/cells.json`. Out of memory in the judge: the judge batch is halved and retried (round 5's loop).
- **A line ended with `END(1)`:** read its log in `runs_private/round7/logs/`; after a fix, append the same line to `queue7.txt` (the queue picks it up) or run it by hand. A `test` line needs its method's pick; a `report` line can be rerun at any time and only reads label files.
- **Short of time:** stop after `test A` for RefusalDirection and AdaSteer on every block, or after `test B` for CAST on the main sections; then run `report`. Missing rows stay "pending" in the report.

## 4. Afterwards, on the Mac

The Drive mirror appears under `~/Library/CloudStorage/GoogleDrive-…/My Drive/lexsi-knob/round7/results/`. Copy `safetune_tests.json`, `safetune_pick.json`, `castcheck.json`, `smoke_cuda.json`, `calib.json`, `run/` and `labels/` into `build/results/round7/` (or unzip the newest `r7_results_<HHMM>.zip`). `python3 scripts/r7_safetune.py report` can be rerun on the Mac from those label files. Then run the leak check, `python3 scripts/pipeline/check_no_harmful_text.py results/round7 scripts`, and rebuild the report with `python3 presentation/build_report_tex.py`.
