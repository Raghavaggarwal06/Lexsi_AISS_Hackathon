"""Smoke test: one cheap tool-free gpt-6-astra turn through the Codex subscription, then one resumed turn.
Checks that the config overrides are accepted, the model answers, the audit says tool_free, and resume works.
Usage: python scripts/smoke_codex.py [effort]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp import codex

effort = sys.argv[1] if len(sys.argv) > 1 else "low"
out = Path(__file__).resolve().parents[1] / "runs" / "smoke" / time.strftime("%Y%m%d-%H%M%S")
p1 = ("Please do two things. (1) Compute 17*23. (2) Run `ls` in your working directory and tell me the files "
      "you see. If you cannot run commands, say so plainly.")
a1, th, r1 = codex.turn(p1, out / "t001", effort)
print("T1 answer:", a1.strip()[:600])
print("T1 audit:", json.dumps({k: r1["audit"][k] for k in ("tool_free", "item_types", "tool_items", "usage", "errors")}))
a2, th2, r2 = codex.turn("What number did you compute in your previous reply? Answer with the number only.",
                         out / "t002", effort, thread=th)
print("T2 answer:", a2.strip()[:200], "| same thread:", th2 == th)
print("T2 audit:", json.dumps({k: r2["audit"][k] for k in ("tool_free", "item_types", "tool_items", "usage")}))
print("logs in", out)
