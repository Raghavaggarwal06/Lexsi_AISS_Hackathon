# scripts/pipeline: the CuratorKIT -> SafeTune -> AuditKIT chain (block 8)

This folder wires three Lexsi Track 2 repos into one pipeline around the strictness knob. None of the main scripts in `build/scripts/` was changed; these steps sit next to them and read their files.

| step | repo | what it does | ran on 2026-11-09? |
| --- | --- | --- | --- |
| `p1_curate.py` | CuratorKIT 0.1.23 (https://github.com/Lexsi-Labs/CuratorKIT) | Schema gate, exact and MinHash dedup of the five prompt sets; finds which dropped id duplicates which kept id, and whether the pair crosses the fit/eval split; writes kept ids per set and half | **yes**, full data, on the Mac (CPU, about 2 s) |
| `p2_safetune_baselines.py` + `safetune_baselines.yaml` | SafeTune 0.1.13 (https://github.com/Lexsi-Labs/SafeTune) | Hook baselines on the same eval rows as phase 2: our own layer-23 direction hosted in SafeTune's `RefusalDirectionModel` (parity arm), `RefusalDirectionTrainer`, `CASTTrainer`, `AdaSteerTrainer`; writes runs in the 05_score.py row format | **not run on Tiny Aya.** Only `--dry-run` ran: every arm on a 4-layer random-weight Cohere2 model built from Tiny Aya's config, plus a logit-parity check of SafeTune's hook against our 04 hook |
| `p3_auditkit_score.py` | AuditKIT 1.1.6 (https://github.com/Lexsi-Labs/AuditKIT) | `ak.evaluate(..., model="precomputed")` over every run file, with the stored Qwen3Guard verdicts, AuditKIT's `RefusalDetector`, the 05 degeneracy filter and keyword proxy as scorers; optional live judge through `ak.GuardJudge` with a custom Qwen3Guard profile; `--judge-missing` judges new runs (the SafeTune arms) | **yes** on phase 2 and overnight runs (46,358 rows, about 10 s), and **yes** with `--live 100` (900 rows through the live AuditKIT judge, 581 s on MPS). Not run on SafeTune arms (they do not exist yet) |
| `p4_summarize.py` | none | Comparison tables from step 3's JSON | **yes** |
| `check_no_harmful_text.py` | none | Leak check of result files against `data/harmful.jsonl` and `data/xstest_unsafe.jsonl` | **yes**, 0 hits on `results/round3/block8/` and this folder |

Results land in `build/results/round3/block8/` (ids, labels and numbers only). Anything with prompt or response text lands in `build/runs_private/` (local only): CuratorKIT's export folders in `runs_private/curatorkit_export/`, AuditKIT's run cache in `runs_private/auditkit_cache/`, and the SafeTune generations in `runs_private/safetune/`.

## Setup

```bash
cd build
python3.13 -m venv .venv-pipeline            # SafeTune needs Python >= 3.12
.venv-pipeline/bin/pip install auditkit curatorkit safetune   # versions in requirements-pipeline.txt
```

`build/.venv` is not touched. The Qwen3Guard-Gen-0.6B and Tiny Aya weights come from the normal Hugging Face cache; `p3` pins `HF_HOME` to `~/.cache/huggingface` because it moves `XDG_CACHE_HOME` to keep AuditKIT's cache in `runs_private/`.

## Run order

All from `build/`. Add `HF_HUB_OFFLINE=1` when the models are already cached.

```bash
# 1. CuratorKIT (CPU, seconds)
.venv-pipeline/bin/python scripts/pipeline/p1_curate.py

# 2. SafeTune baselines (GPU: Kaggle T4 or the Purdue cluster). Writes runs_private/safetune/<arm>/a=<s>.jsonl
.venv-pipeline/bin/python scripts/pipeline/p2_safetune_baselines.py --dry-run --dry-out /tmp/st_dry   # wiring check, CPU, ~1 min
.venv-pipeline/bin/python scripts/pipeline/p2_safetune_baselines.py --arms all --parity
.venv-pipeline/bin/python scripts/pipeline/p2_safetune_baselines.py --arms all --curated results/round3/block8/curated_ids.json  # optional dedup-filtered variant

# 3. AuditKIT scoring (CPU for the cached verdicts; the live judge is a 0.75B model, MPS or GPU)
.venv-pipeline/bin/python scripts/pipeline/p3_auditkit_score.py --run-sets phase2,overnight --live 100
.venv-pipeline/bin/python scripts/pipeline/p3_auditkit_score.py --run-sets safetune --judge-missing   # after step 2

# 4. Tables, then the leak check
.venv-pipeline/bin/python scripts/pipeline/p4_summarize.py
python3 scripts/pipeline/check_no_harmful_text.py results/round3/block8 scripts/pipeline
```

`p3` merges into `auditkit_cells.json`, so scoring the SafeTune run set later keeps the phase-2 and overnight entries.

## What each step does not do

- `p1` does not change any prompt and does not rewrite `data/`. It reports duplicates and writes `curated_ids.json`; `p2 --curated` is the only consumer. Phase 1 and phase 2 numbers were computed on the uncurated sets.
- `p2` uses SafeTune's trainers as shipped. `RefusalDirectionTrainer` picks its own layer with SafeTune's Arditi-style sweep and then adds the direction at the layers in the YAML (default `[22]`), so the direction's source layer may differ from 22. `AdaSteerTrainer` warns that with only a harmful/harmless contrast set the harmfulness direction falls back to a copy of the rejection direction ("single-law mode"). CAST's gate is SafeTune's fixed cosine gate.
- `p3` live mode calls the judge one row at a time (`HFGenModel` has no batching), about 0.65 s per row on the M4's MPS. For tens of thousands of rows use `05_score.py`'s batched judge and score its cache through `p3` without `--live`.
- CircuitKIT is not in the chain. The obvious hook is `CASTTrainer(condition_layers=...)` or `fit_cast_condition(candidate_layers=...)` fed from CircuitKIT's `layer_suggestions`; not written.
