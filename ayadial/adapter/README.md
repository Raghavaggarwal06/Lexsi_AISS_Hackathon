# The adapters

AyaDial ships as **four LoRA files**, two per direction of the dial. They are small: the whole set is
under 800 KB.

| id | file | bytes | sha256 (first 16) | what it is |
| --- | --- | --- | --- | --- |
| 0 | `strict_install.gguf` | 314048 | `7d36cfff86834896` | Tighten: gate, up and down rows of units 6666 and 1896, layer 22 |
| 1 | `strict_knob.gguf` | 104864 | `92407ec95389cb49` | Tighten dial: `α·w` on both gate rows. Its scale is `s` |
| 2 | `lax_install.gguf` | 314048 | `afe96fdb5bb141d1` | Loosen: units 1785 and 4466, down columns zeroed |
| 3 | `lax_knob.gguf` | 52640 | `efab732884d2145e` | Loosen dial: the two down columns, rank 1. Its scale is `t` |

Sizes and hashes are the ones recorded at export time in
[`../results/round5/gguf/export_both_sides.json`](../results/round5/gguf/export_both_sides.json).

## The binaries are not in this repo yet

They were exported inside a Colab session and were not copied back before it ended. Everything needed
to rebuild them byte-for-byte is here:

| | |
| --- | --- |
| `vec_v3g.npz` | the fitted vectors: `w`, `d`, `m`, `q`, all 2048-dim, plus `theta0`, `b`, `U`. **This is the one piece that cannot be recovered from the JSON** — `../results/round5/lab/v3g.json` holds only the scalars |
| `../results/round5/gguf/export_both_sides.json` | thresholds, unit indices, pinned llama.cpp commit, expected sizes and hashes |
| `../scripts/r5_gguf.py` | the export itself |

Rebuild them with

```bash
python scripts/r5_gguf.py prep                 # llama.cpp with CUDA, base model to GGUF
python scripts/r5_gguf.py test 0.3747 2.0      # theta_lax and t; writes the four files
```

then check each file against the table above before publishing it. The export is deterministic, so a
matching hash means you have the artifact the report measured.

Pinned for reproducibility: llama.cpp commit `7c2653acc749258dc62067da62d07024d971ccd2`, vector set
`v3g`, strict units `[6666, 1896]`, lax units `[1785, 4466]`.

## Serving

Load all four in order and give one user-facing number `s`. It maps onto the four scales as:

```
(1, max(s, 0), 1, max(-s, 0))
```

So `s = 1` is Tighten at the report's setting, `s = 0` is the base model untouched, and `s = -1`
engages Loosen. With `llama-server`, the scales go in the per-request `lora` field:

```json
{
  "prompt": "...",
  "lora": [
    {"id": 0, "scale": 1.0},
    {"id": 1, "scale": 1.0},
    {"id": 2, "scale": 1.0},
    {"id": 3, "scale": 0.0}
  ]
}
```

Requests with different LoRA scales are not batched together, which costs throughput if callers use
many different settings at once.

## Shipping only one direction

The Loosen pair (ids 2 and 3) can be withheld without touching the Tighten pair: export with the
strict units only and serve ids 0 and 1. The Tighten numbers in the report are unaffected.
