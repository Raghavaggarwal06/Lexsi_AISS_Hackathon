"""Configuration for the weight-space-programming harness.

Every value has a default here. A run reads one JSON file from wsp/configs/ (mock.json, pilot.json, real.json) whose keys
override these defaults, then any `--set key.sub=value` from the command line. The merged config is written to
wsp/runs/<run_id>/config.json at the start of a run and never changes afterwards (resume reads it back).

Paths are relative to build/ unless absolute. `~` is expanded.
"""
import copy
import json
from pathlib import Path

WSP = Path(__file__).resolve().parent
BUILD = WSP.parent

DEFAULTS = {
    "run_id": None,                 # default: <mode>-<UTC timestamp>
    "mode": "pilot",                # label only: mock | pilot | real
    # --- editor (Anthropic API). Facts from the claude-api skill (model table cached 2026-11-03, Fable 5.2 section).
    "model": "claude-fable-5-2",
    "effort": "high",               # output_config.effort: low | medium | high | xhigh | max
    "max_tokens": 64000,            # hard cap on thinking + reply text per call; streamed
    "thinking_display": "summarized",  # "summarized" logs a readable summary of the thinking; "omitted" logs nothing
    "cache_ttl": "1h",              # "5m" or "1h" for the prompt cache (1h writes cost 2x input, 5m writes 1.25x)
    "fallbacks": None,              # None = no server-side refusal fallback (a fallback would silently change the editor model)
    "api_timeout_s": 900,           # read timeout for one streamed call
    "api_retries": 2,               # driver-level retries of a failed call (each retry is re-checked against the dollar cap)
    "dollar_cap": 15.0,             # hard cap in USD, computed from API usage; the driver stops BEFORE a call that could exceed it
    "prices_usd_per_mtok": {        # Claude Fable 5.2, Anthropic first-party API
        "input": 8.0, "output": 40.0, "cache_write_5m": 10.0, "cache_write_1h": 16.0, "cache_read": 0.20,
        "source": "claude-api skill (bundled with Claude Code 2.1.322): model table cached 2026-11-03 and 'Migrating to Claude Fable 5.2' section",
    },
    "chars_per_token_floor": 2.0,   # conservative: new text is assumed to cost len(text)/2 tokens when estimating the next call
    "max_turns": 400,               # API calls that produce a reply (format retries and refusals count too)
    "pause_at_turn": None,          # stop cleanly once this turn's block has run (resume: --resume <run_id> --set pause_at_turn=null)
    "handoff_every": 40,            # start a fresh conversation every N turns of the current conversation
    "handoff_max_input_tokens": 400000,  # ... or as soon as one request's input (thinking blocks are kept in history) passes this
    "notes_tail_chars": 8000,       # tail of WORK/notes.md carried into a handoff
    # --- reply rules
    "min_explanation_words": 20,    # plain-English words before the block; fewer = "unexplained"
    "max_unexplained_in_a_row": 2,  # an unexplained block is sent back unrun at most this many times in a row; the next one runs
    "max_consecutive_format_errors": 5,
    "max_consecutive_refusals": 3,
    "max_nudges": 3,                # replies without a block while budget is left; after this many the reply is taken as final
    # --- python kernel
    "python_runs": 300,
    "python_timeout_s": 900,
    "python_grace_s": 15,           # after the timeout: SIGINT, wait this long, then SIGKILL (and restart, announced)
    "kernel_memory_gb": 10.0,       # watchdog: RSS above this kills the kernel (announced); macOS ignores RLIMIT_AS / RLIMIT_DATA
    "kernel_watchdog_interval_s": 0.25,
    "output_max_chars": 10000,      # Python output sent to the editor is cut to this many characters
    "work_max_gb": 20.0,            # warn the editor and refuse further runs if WORK grows beyond this
    "python_executable": ".venv/bin/python",
    # --- forward passes
    "backend": "stub",              # stub (mock only) | remote
    "remote": {"queue_dir": None, "poll_s": 10, "timeout_s": 3600},
    "reads": 40,                    # forward-only read passes
    "behaves": 12,                  # generate + judge passes
    "read_max_prompts": 2048,
    "read_max_hidden_layers": 6,
    "behave_max_prompts_per_arm": 1024,
    "behave_max_arms": 3,
    "behave_max_new_tokens": 128,
    # --- what the editor may read and edit
    "weights": {
        "hf_dir": "~/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global",
        "read_layers": [18, 19, 20, 21, 22],
        "slice_file": "wsp/cache/tiny_aya_global_L18-22.safetensors",
    },
    "edit": {
        "layer": 22,
        "fold_units": [6666, 1896],
        "census_file": "results/round3/block3/marker.json",   # dormant-unit census of layers[22] (round 3, block 3)
        "exclude_units": [1785, 4466],  # the lax pair of rounds 5 and 6 (private side): never offered to the editor
        "extra_units_max": 8,           # how many further census units, in census order, are allowed on top of the fold units
        "max_units": 10,                # distinct units that may differ from the original model across gate, up and down
        "knob_tensors": ["gate_proj"],  # tensors whose knob part (multiplied by s) may be non-zero; gate only = the current GGUF export
        "max_row_norm": {"gate_proj": 400.0, "up_proj": 50.0, "down_proj": 120.0},
        "max_knob_row_norm": 400.0,
    },
    "start_knob": "ayadial_improved",   # ayadial_improved (round 5, v3g) | ayadial (round 4, v1)
    "sources": {
        "ayadial_adapters": "runs_private/round4/adapters/fold_cap",
        "ayadial_improved_vec": "cache_r5/vec_v3g.npz",            # m, q, w, theta0 of round 5's v3g; not on the Mac on 11 Nov
        "refusal_direction": "results/day1/directions.npz",        # r = r_raw[23]
    },
    # --- the fixed scoreboard (described in the first message; numbers from files are inserted by the builder)
    "scoreboard": {
        "scales": [0, 1, 1.25, 2],
        "headline_scale": 1,
        "budget_points": 5,                     # the 'holds' rule of round 5 (results/round5/summary.md, section 4)
        "monotonic_step_points": 3,             # build/CLAUDE.md phase 2 monotonicity rule
        "harmless_rise_vs_current_max": 2.0,   # PROPOSED: no harmless group may rise by more than this over AyaDial-Improved
        "kl_bar": 1e-3,                         # s = 0 against the original model, next-token KL (build/CLAUDE.md parity bar)
    },
    "allow_early_finish": False,
}


def deep_update(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_value(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def load(config_file=None, sets=()):
    cfg = copy.deepcopy(DEFAULTS)
    if config_file:
        deep_update(cfg, json.loads(Path(config_file).read_text()))
    for item in sets:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parse_value(val)
    validate(cfg)
    return cfg


def validate(cfg):
    assert cfg["effort"] in ("low", "medium", "high", "xhigh", "max"), cfg["effort"]
    assert cfg["cache_ttl"] in ("5m", "1h"), cfg["cache_ttl"]
    assert cfg["backend"] in ("stub", "remote"), cfg["backend"]
    assert cfg["start_knob"] in ("ayadial_improved", "ayadial"), cfg["start_knob"]
    assert cfg["thinking_display"] in ("summarized", "omitted"), cfg["thinking_display"]
    assert 0 < cfg["dollar_cap"], cfg["dollar_cap"]
    assert 1 <= cfg["max_unexplained_in_a_row"] <= 5
    assert set(cfg["edit"]["knob_tensors"]) <= {"gate_proj", "up_proj", "down_proj"}
    assert cfg["edit"]["max_units"] >= len(cfg["edit"]["fold_units"])
    pt = cfg.get("pause_at_turn")
    assert pt is None or (isinstance(pt, int) and not isinstance(pt, bool) and pt >= 1), f"pause_at_turn must be null or a turn >= 1: {pt!r}"
    return cfg


# keys that `--resume <run_id> --set key=value` may change in a run's frozen config (the change is written back and logged)
RESUME_SETTABLE = ("pause_at_turn",)


def path(p):
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else BUILD / p
