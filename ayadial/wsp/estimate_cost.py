"""Rough cost per turn of a real run, from the price table in config.py and stated assumptions. Nothing here calls the API.
  .venv/bin/python wsp/estimate_cost.py [--out-high 8000] [--out-max 24000] [--result 2500] [--turns 120]

Assumptions (each is a guess the pilot replaces with measured usage.json numbers):
  - the first message is len(FIRST_MESSAGE.md) / 3.5 tokens (dense tables tokenize worse than prose at 4 characters per token);
  - each tool result the editor reads is --result tokens on average (output is cut at 10,000 characters, about 2,900 tokens);
  - each reply costs --out-high / --out-max output tokens (thinking + text) at effort high / max. Reference points, not
    measurements of Fable 5.2: the teammate's Astra run at xhigh produced 90,576 output tokens in 14 turns, about 6,500 per turn
    (notes/weight-programming/HARNESS-NOTES.md, section 3.2); thinking at max is assumed to be 3 times that at high;
  - thinking blocks stay in the history (preserved thinking), so the history grows by output + result every turn and is re-read
    from the cache on every call; every call writes its new tail to the cache at the configured TTL and never misses;
  - a new conversation starts every handoff_every turns or when the input passes handoff_max_input_tokens, and re-writes the
    first message plus a handoff block of about 3,000 tokens.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def simulate(cfg, out_tokens, result_tokens, turns, first_tokens):
    pr = cfg["prices_usd_per_mtok"]
    w = pr["cache_write_1h"] if cfg["cache_ttl"] == "1h" else pr["cache_write_5m"]
    total, prefix, conv_turns, convs, max_input = 0.0, 0, 0, 1, 0
    for t in range(turns):
        if t == 0 or conv_turns >= cfg["handoff_every"] or prefix >= cfg["handoff_max_input_tokens"]:
            new, prefix, conv_turns = first_tokens + (3000 if t else 0), 0, 0
            convs += t > 0
        else:
            new = out_tokens + result_tokens
        cost = (prefix * pr["cache_read"] + new * w + out_tokens * pr["output"]) / 1e6
        total += cost
        prefix += new
        max_input = max(max_input, prefix)
        conv_turns += 1
    return {"usd_per_turn": round(total / turns, 3), "usd_total": round(total, 2), "turns": turns, "conversations": convs,
            "largest_input_tokens": max_input}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-high", type=int, default=8000)
    ap.add_argument("--out-max", type=int, default=24000)
    ap.add_argument("--result", type=int, default=2500)
    ap.add_argument("--turns", type=int, default=120)
    a = ap.parse_args()
    first = int(len((C.WSP / "FIRST_MESSAGE.md").read_text()) / 3.5)
    out = {"pricing": C.DEFAULTS["prices_usd_per_mtok"], "first_message_tokens_assumed": first, "result_tokens_assumed": a.result}
    for name, conf, o in (("high", "pilot.json", a.out_high), ("max", "real.json", a.out_max)):
        cfg = C.load(C.WSP / "configs" / conf)
        r = simulate(cfg, o, a.result, a.turns, first)
        r["output_tokens_per_turn_assumed"] = o
        r["worst_case_single_call_usd_at_10k_input"] = round((10000 * cfg["prices_usd_per_mtok"]["cache_write_1h"] + cfg["max_tokens"] * cfg["prices_usd_per_mtok"]["output"]) / 1e6, 2)
        out[name] = r
    out["one_cache_miss_on_a_200k_prefix_usd"] = round(200000 * C.DEFAULTS["prices_usd_per_mtok"]["cache_write_1h"] / 1e6, 2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
