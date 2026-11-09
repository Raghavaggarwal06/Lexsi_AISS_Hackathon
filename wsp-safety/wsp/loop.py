"""The conversation driver (the part of the paper's harness that lm_understand-logs does not release).

Protocol, reconstructed from the released turns (exp7_sycophancy/runs/*/pair*/turns/t*/prompt.txt, state.json):
  - turn 1 sends the task prompt; every later turn resumes the same Codex thread.
  - A reply holding a ```python block (or a block of another enabled tool, e.g. ```attention) is one RUN: the first
    such block is executed and the next message is
        "OUTPUT of run k of at most N:\n```\n<output>\n```"
    and after the last run: "\nThat was the last run. Now give your final answer, without a ```python block, ending
    with the JSON block."
  - A reply without such a block is the final answer. If runs < nudge.below and fewer than nudge.max nudges were sent,
    it is kept as early_answer_<n>.md and the model gets
        "You have used k of N runs. Before finishing, test at least one hypothesis you have not tested yet. If you
         still want to finish after that, give your final answer again."
  - A final answer that does not parse gets (up to max_format_retries times)
        "Your final answer could not be read (<reason>). Please reply again with your final answer, without a
         ```python block, ending with the JSON block in the requested format."
State is saved after every turn (state.json), so a killed driver resumes where it stopped.
"""
import json, re, time
from dataclasses import dataclass
from pathlib import Path

from . import codex

BLOCK_RE = r"```({langs})[ \t]*\n(.*?)```"
JSON_RE = re.compile(r"```json[ \t]*\n(.*?)```", re.S)


def last_json(text):
    """The last ```json block of a reply, parsed; None if missing or invalid."""
    blocks = JSON_RE.findall(text)
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except json.JSONDecodeError:
            continue
    return None


@dataclass
class Nudge:
    below: int = 0         # nudge a final answer given after fewer than `below` runs
    max: int = 0           # at most this many nudges per conversation


class Conversation:
    def __init__(self, out_dir, prompt, *, effort, max_runs, tools=None, validate=None, nudge=None,
                 model=codex.MODEL, max_format_retries=3, max_turns=None, log=None):
        """tools: {"python": fn(body, k) -> str, ...}; validate: fn(answer_text) -> (parsed, error_or_None)."""
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prompt, self.effort, self.max_runs, self.model = prompt, effort, max_runs, model
        self.tools = tools or {}
        self.validate = validate or (lambda a: (last_json(a), None if last_json(a) is not None else "no JSON block"))
        self.nudge = nudge or Nudge()
        self.max_format_retries = max_format_retries
        self.max_turns = max_turns or (max_runs + self.nudge.max + max_format_retries + 5)
        self.block_re = re.compile(BLOCK_RE.format(langs="|".join(map(re.escape, self.tools))), re.S) if self.tools else None
        self.log = Path(log) if log else self.dir / "loop.log"
        sp = self.dir / "state.json"
        self.state = json.loads(sp.read_text()) if sp.exists() else {
            "turn": 0, "runs": 0, "thread": None, "done": False, "format_retries": 0, "nudges": 0,
            "effort": effort, "model": model, "max_runs": max_runs, "tool_free": [], "cum_input": 0,
            "cum_output": 0, "next": prompt}
        if not sp.exists():
            (self.dir / "prompt.txt").write_text(prompt)

    def _say(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {self.dir.name} {msg}"
        with open(self.log, "a") as f:
            f.write(line + "\n")

    def _save(self):
        (self.dir / "state.json").write_text(json.dumps(self.state, indent=1))

    def _transcript(self):
        parts = [f"# {self.dir.name}\n\nmodel `{self.model}`, effort `{self.effort}`, runs {self.state['runs']} of "
                 f"{self.max_runs}\n"]
        for t in sorted((self.dir / "turns").glob("t*")):
            p, a = t / "prompt.txt", t / "answer.md"
            parts.append(f"\n## {t.name}: user\n\n{p.read_text() if p.exists() else ''}\n")
            parts.append(f"\n## {t.name}: model\n\n{a.read_text() if a.exists() else ''}\n")
        (self.dir / "transcript.md").write_text("".join(parts))

    def run(self):
        s = self.state
        while not s["done"] and s["turn"] < self.max_turns:
            s["turn"] += 1
            tdir = self.dir / "turns" / f"t{s['turn']:03d}"
            answer, thread, rec = codex.turn(s["next"], tdir, self.effort, s["thread"], self.model)
            s["thread"] = thread
            a = rec["audit"]
            s["tool_free"].append(a["tool_free"])
            u = a.get("usage") or {}
            s["cum_input"] += u.get("input_tokens", 0)
            s["cum_output"] += u.get("output_tokens", 0)
            if not a["tool_free"]:
                self._say(f"WARNING turn {s['turn']} not tool-free: {a['tool_items']}")
            m = self.block_re.search(answer) if self.block_re else None
            if m:
                lang, body = m.group(1), m.group(2)
                if s["runs"] >= self.max_runs:
                    s["next"] = (f"You have used all {self.max_runs} runs. Now give your final answer, without a "
                                 "```python block, ending with the JSON block.")
                else:
                    s["runs"] += 1
                    self._say(f"turn {s['turn']} run {s['runs']}/{self.max_runs} ({lang}, {rec['wall_s']} s)")
                    out = self.tools[lang](body, s["runs"])
                    nxt = f"OUTPUT of run {s['runs']} of at most {self.max_runs}:\n```\n{out.rstrip()}\n```"
                    if s["runs"] == self.max_runs:
                        nxt += ("\nThat was the last run. Now give your final answer, without a ```python block, "
                                "ending with the JSON block.")
                    s["next"] = nxt
            elif s["runs"] < min(self.nudge.below, self.max_runs) and s["nudges"] < self.nudge.max:
                s["nudges"] += 1
                (self.dir / f"early_answer_{s['nudges']}.md").write_text(answer)
                self._say(f"turn {s['turn']} early answer after {s['runs']} runs: nudge {s['nudges']}")
                s["next"] = (f"You have used {s['runs']} of {self.max_runs} runs. Before finishing, test at least one "
                             "hypothesis you have not tested yet. If you still want to finish after that, give your "
                             "final answer again.")
            else:
                parsed, err = self.validate(answer)
                if err and s["format_retries"] < self.max_format_retries:
                    s["format_retries"] += 1
                    self._say(f"turn {s['turn']} unreadable final answer ({err}): retry {s['format_retries']}")
                    s["next"] = (f"Your final answer could not be read ({err}). Please reply again with your final "
                                 "answer, without a ```python block, ending with the JSON block in the requested "
                                 "format.")
                else:
                    (self.dir / "final_answer.md").write_text(answer)
                    (self.dir / "prediction.json").write_text(json.dumps(
                        parsed if not err else {"error": err, "parsed": parsed}, indent=1))
                    s["done"] = True
                    s["next"] = None
                    self._say(f"turn {s['turn']} FINAL after {s['runs']} runs; error={err}")
            self._save()
            self._transcript()
        return s
