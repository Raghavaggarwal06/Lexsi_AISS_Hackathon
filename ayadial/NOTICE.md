# Notice

This repository mixes three kinds of material under three different sets of terms.

## 1. Code

The code in `scripts/` and `wsp/`, and the LaTeX sources in `report/`, carry whatever licence the team puts on this
repository. There is none at the repository root yet.

## 2. Adapter weights — CC-BY-NC-4.0, inherited

The AyaDial adapters are computed from the activations and weights of
[`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) and are written into
that model's own MLP units. They are a derivative work of it, so they carry the model's licence:

> **CC-BY-NC-4.0** — attribution required, **non-commercial use only**.

Attribution: Tiny Aya Global, Cohere Labs. Any redistribution of the adapters, merged weights or GGUF
files must keep this notice and the non-commercial restriction. Whatever licence the team chooses for the
code does **not** extend to them.

## 3. Prompt sets — the terms of each source

`data/` contains only the sets that may be redistributed, and only the fields needed to rerun the
evaluation. The harmful sets are not included here; `scripts/01_fetch_data.py` fetches them from the
original sources, under those sources' own terms and gating.

| set | source | licence / gating |
| --- | --- | --- |
| `benign.jsonl` | `JailbreakBench/JBB-Behaviors` (benign split), `tatsu-lab/alpaca` | MIT / CC-BY-NC-4.0 |
| `xstest_safe.jsonl` | `walledai/XSTest`, `label == "safe"` | CC-BY-4.0, auto-gated |
| `orbench_hard.jsonl` | `bench-llm/or-bench`, `or-bench-hard-1k` | CC-BY-4.0 |
| not shipped | `walledai/HarmBench`, `JailbreakBench/JBB-Behaviors` (harmful split), WildJailbreak | gated, withheld here by choice |

The judge model, `Qwen/Qwen3Guard-Gen-0.6B`, is Apache-2.0.

## Dual use

AyaDial ships both directions of the dial. The **Tighten** pair raises refusals on harmful requests;
the **Loosen** pair lowers refusals, which is what makes the dial a dial rather than a one-way switch
and is how the report measures over-refusal honestly. The Loosen pair can be withheld without touching
the Tighten pair, and the export script supports that: see `adapter/README.md`.
