# data/

One JSONL per prompt set, one row per prompt: `{"id", "prompt", "source", "set"}`.
Rebuilt by `scripts/01_fetch_data.py`.

| file | rows | here? |
| --- | --- | --- |
| `benign.jsonl` | 250 | yes |
| `xstest_safe.jsonl` | 250 | yes |
| `orbench_hard.jsonl` | 250 | yes |
| `splits.json` | — | yes |
| `harmful.jsonl` | 300 | **no** |
| `xstest_unsafe.jsonl` | 200 | **no** |

The two harmful sets are not published. Running `scripts/01_fetch_data.py` rebuilds them from
`walledai/HarmBench` and `JailbreakBench/JBB-Behaviors`, which are gated at the source, so anyone
reproducing this gets them under those datasets' own terms rather than from us. Any log or table we
publish carries the row id and source index only, never the prompt text.

`splits.json` holds the fit and eval halves per set, by id, seed 0. Directions and thresholds are
fitted on the fit half. AUROC, FPR and every curve in the report are read on the eval half only.
Do not mix them.
