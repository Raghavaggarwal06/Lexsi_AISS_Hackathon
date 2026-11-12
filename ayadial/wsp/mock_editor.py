"""A scripted fake editor for --mock runs: no network, no API key, no cost. It returns API-shaped replies (content blocks, stop_reason,
usage) so the driver's whole loop, including cost accounting and the dollar cap, runs exactly as with the real editor.

The script walks through every path the mock test must cover: a Python run, a reply without explanation (sent back), a valid read,
an invalid request (not counted), refused edits (the kernel's own check, and a forged file that only the driver's check catches), an
accepted edit, a behaviour pass, an exact repeat (refused), a reply with two blocks, a reply with no block (nudge), then plain Python
runs until the harness asks for the final answer, which it gives without a block.

Usage is simulated: output tokens are fixed per reply and input tokens follow the conversation length (chars / 4), split between cache
reads (the prefix seen before) and cache writes (the new part), so cost per turn grows as a real conversation's does.
"""
import json

EXPL = ("This step is part of the scripted mock run that tests the harness. The previous result is only used to check that the loop "
        "works, and none of these numbers are measurements. The block below exercises one path of the driver so that its log can be checked.")

GATE = "model.layers.22.mlp.gate_proj.weight"


def py(code, expl=EXPL):
    return f"{expl}\n\n```python\n{code.strip()}\n```\n"


def fwd(obj, expl=EXPL):
    return f"{expl}\n\n```forward\n{json.dumps(obj)}\n```\n"


SCRIPT = [
    ("python", py("""
print(KNOB['name'], KNOB['units'], round(float(KNOB['theta0']), 4), len(NAMES), ALLOWED_UNITS)
print(edited())
open('notes.md', 'w').write('# notes\\nmock run: starting knob loaded\\n')
""")),
    ("unexplained", "```python\nprint('this block has no explanation before it')\n```\n"),
    ("python", py("print('now with an explanation'); x_saved = 7")),
    ("read", fwd({"type": "read", "weights": "current", "scale": 1.0, "set": "adversarial_harmful", "n": 64, "format": "default",
                  "positions": "marker", "return": {"hidden": [22, 23], "mlp22_in": True, "mlp22_units": "knob", "first_token": True}})),
    ("invalid", fwd({"type": "read", "weights": "current", "set": "no_such_set", "n": 10})),
    ("refused_edits", py("""
G21 = load('model.layers.21.mlp.gate_proj.weight')
try:
    save_edit('model.layers.21.mlp.gate_proj.weight', G21 * 1.01)
except Exception as e:
    print('kernel check:', str(e)[:200])
G = load_original('""" + GATE + """').clone(); G[5] = G[5] * 2.0
try:
    save_edit('""" + GATE + """', G)
except Exception as e:
    print('kernel check:', str(e)[:200])
# forge a pending edit file directly, bypassing the kernel's check: only the driver's check can catch this one
import numpy as np, os
os.makedirs('edits/pending', exist_ok=True)
np.savez('edits/pending/99999_gate_proj.npz', name=np.array('""" + GATE + """'), units=np.array([5], dtype=np.int64),
         install=np.ones((1, 2048), np.float32), knob=np.zeros((1, 2048), np.float32))
print('forged file written')
""")),
    ("accepted_edit", py("""
G = load('""" + GATE + """')   # the starting knob installed at s = 0 ... plus its knob part at s = 1
G0 = load('""" + GATE + """', scale=0.0)
K = load_knob('""" + GATE + """')
G0[9173] = G0[6666] * 0.5          # an arbitrary change on an allowed unit, for the test only
K[9173] = K[6666] * 0.5
save_edit('""" + GATE + """', G0, knob=K)
print(edited())
open('notes.md', 'a').write('mock: edited unit 9173 gate row (test only)\\n')
""")),
    ("behave", fwd({"type": "behave", "arms": [{"weights": "current", "scale": 1}, {"weights": "edited", "scale": 1}],
                    "sets": ["adversarial_harmful", "adversarial_benign"], "n": 32, "max_new_tokens": 64})),
    ("repeat", fwd({"type": "behave", "arms": [{"weights": "current", "scale": 1}, {"weights": "edited", "scale": 1}],
                    "sets": ["adversarial_harmful", "adversarial_benign"], "n": 32, "max_new_tokens": 64})),
    ("python", py("import os; print(sorted(os.listdir('.'))); print(x_saved if 'x_saved' in globals() else 'variables gone')")),
    ("two_blocks", EXPL + "\n\n```python\nprint(1)\n```\n\nand\n\n```python\nprint(2)\n```\n"),
    ("no_block", "I have nothing to run at this moment; I am thinking about the results so far."),
]
FILLER = ("python", py("import numpy as np\nn_fill = globals().get('n_fill', 0) + 1\nprint('fill', n_fill, edited()['edit_hash'], float(np.linalg.norm(KNOB['w'])))"))
FINAL = ("final", "Final answer (mock). I changed the gate row of unit 9173 in layers[22].mlp as a test of the harness only. "
                  "There is no evidence for it: every number in this run came from the stub backend and is fake. I expect no effect.")


class MockEditor:
    model = "mock-editor"

    def __init__(self, out_tokens=5000):
        self.i = 0
        self.out_tokens = out_tokens
        self.seen_chars = {}

    def call(self, system, messages, cfg, conv_id, turn=None):
        """The script position is the turn number (turn t plays SCRIPT[t - 1]), so a resumed run continues where it stopped."""
        last = messages[-1]["content"]
        last = last if isinstance(last, str) else " ".join(b.get("text", "") for b in last if isinstance(b, dict))
        self.i = (turn or self.i + 1)
        if "[note from the experimenters] The session must end now" in last or "[note from the experimenters] All " in last:
            kind, text = FINAL
        elif self.i - 1 < len(SCRIPT):
            kind, text = SCRIPT[self.i - 1]
        else:
            kind, text = FILLER
        chars = len(system) + sum(len(json.dumps(m["content"])) for m in messages)
        prev = self.seen_chars.get(conv_id, 0)
        self.seen_chars[conv_id] = chars + len(text) + 4 * self.out_tokens
        new = max(chars - prev, 0)
        usage = {"input_tokens": 3, "cache_read_input_tokens": prev // 4, "cache_creation_input_tokens": new // 4,
                 "cache_creation": {"ephemeral_5m_input_tokens": 0 if cfg["cache_ttl"] == "1h" else new // 4,
                                    "ephemeral_1h_input_tokens": new // 4 if cfg["cache_ttl"] == "1h" else 0},
                 "output_tokens": self.out_tokens + len(text) // 4}
        content = [{"type": "thinking", "thinking": f"(mock thinking summary for step {self.i})", "signature": "mock"},
                   {"type": "text", "text": text}]
        return {"id": f"mock_{self.i}", "model": self.model, "content": content, "stop_reason": "end_turn", "stop_details": None,
                "usage": usage, "mock_kind": kind}
