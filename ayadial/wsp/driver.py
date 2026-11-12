"""Driver for the weight-space-programming experiment: an LLM editor improves the strictness knob of Tiny Aya Global by editing
layer 22's MLP through a sandboxed Python kernel and a forward-pass tool. Run from build/ with the venv:

  .venv/bin/python wsp/driver.py --mock                       scripted fake editor + stub passes; no network, no cost
  .venv/bin/python wsp/driver.py --dry-run [--config F]       print the config and the first message, then exit
  .venv/bin/python wsp/driver.py --config wsp/configs/pilot.json      a real run (calls the Anthropic API: costs money)
  .venv/bin/python wsp/driver.py --resume <run_id>            continue a run after a crash or a stop
  options: --set key.sub=value (repeatable), --run-id NAME, --crash-at TURN:PHASE (testing; PHASE = after_reply | after_tool)

One turn = one API call that returns a reply. The reply must hold a plain-English explanation and exactly one fenced block:
```python (run in the kernel) or ```forward (one pass); a reply with no block is a final answer only once the harness has asked for
one. Everything is logged under wsp/runs/<run_id>/; state.json is rewritten after every step, so a crash loses at most one turn, and
a recorded reply is reused instead of asking (and paying) again.

Privacy: no prompt or reply text of any dataset exists in this harness (passes return numbers by set and id). The API key is read
by the SDK from ANTHROPIC_API_KEY and is never printed or written; the kernel starts with a scrubbed environment. Paths of this
machine are replaced by WORK, CASE and <python> before any kernel output is sent to the API.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

WSP = Path(__file__).resolve().parent
sys.path.insert(0, str(WSP))
import config as C  # noqa: E402
import edits as E  # noqa: E402
import passes as P  # noqa: E402

SYSTEM = ("You are the editor in a weight-editing experiment. The first user message describes the task, the tools and the rules. "
          "Every later user message comes from an automated harness: it reports the result of the single fenced block in your previous "
          "reply, or carries a note from the experimenters.")
FENCE = re.compile(r"^```([^\n`]*)\n(.*?)^```[ \t]*$", re.S | re.M)
END_NOTE = ("[note from the experimenters] The session must end now. Reply without any block: what you changed (tensors, units, "
            "values in words), the evidence (pass numbers and results), what you expect on the scoreboard and how sure you are. "
            "Your accepted edit at this moment is what gets scored.")


# == small helpers ==
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def atomic_write(path, text):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def jdump(obj):
    return json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


class Log:
    def __init__(self, path, echo=True):
        self.path, self.echo = Path(path), echo

    def __call__(self, *a):
        line = f"{now()} " + " ".join(str(x) for x in a)
        with open(self.path, "a") as f:
            f.write(line + "\n")
        if self.echo:
            print(line, flush=True)


# == cost ==
def call_cost(usage, cfg):
    """USD for one call from its usage block. Cache writes are split by TTL when the API reports the split."""
    pr = cfg["prices_usd_per_mtok"]
    u = usage or {}
    cc = u.get("cache_creation") or {}
    w5, w1 = cc.get("ephemeral_5m_input_tokens"), cc.get("ephemeral_1h_input_tokens")
    total_w = u.get("cache_creation_input_tokens") or 0
    if w5 is None and w1 is None:
        w5, w1 = (0, total_w) if cfg["cache_ttl"] == "1h" else (total_w, 0)
    usd = ((u.get("input_tokens") or 0) * pr["input"] + (u.get("output_tokens") or 0) * pr["output"]
           + (u.get("cache_read_input_tokens") or 0) * pr["cache_read"] + (w5 or 0) * pr["cache_write_5m"] + (w1 or 0) * pr["cache_write_1h"])
    return usd / 1e6


def worst_case_next(state, new_chars, cfg, fresh_conv_chars=None):
    """A conservative upper bound in USD for the next call: every input token priced at the highest input rate (a 1-hour cache
    write), the full max_tokens of output, and new text counted at chars_per_token_floor characters per token."""
    pr = cfg["prices_usd_per_mtok"]
    top_in = max(pr["input"], pr["cache_write_5m"], pr["cache_write_1h"], pr["cache_read"])
    if fresh_conv_chars is not None:
        tokens_in = fresh_conv_chars / cfg["chars_per_token_floor"]
    else:
        tokens_in = state.get("last_total_input", 0) + state.get("last_output", 0) + new_chars / cfg["chars_per_token_floor"]
    return (tokens_in * top_in + cfg["max_tokens"] * pr["output"]) / 1e6


# == editors ==
class AnthropicEditor:
    """The real editor. Not exercised by the mock test (the owner approves the first real call)."""

    def __init__(self, cfg):
        import anthropic  # pip install anthropic (not in .venv on 11 Nov 2026)
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=2, timeout=cfg["api_timeout_s"])  # key from ANTHROPIC_API_KEY, never logged
        self.cfg = cfg
        self.model = cfg["model"]

    def call(self, system, messages, cfg, conv_id, turn=None):
        ttl = cfg["cache_ttl"]
        params = dict(model=cfg["model"], max_tokens=cfg["max_tokens"],
                      system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral", "ttl": ttl}}],
                      messages=messages, thinking={"type": "adaptive", "display": cfg["thinking_display"]},
                      output_config={"effort": cfg["effort"]}, cache_control={"type": "ephemeral", "ttl": ttl})
        if cfg.get("fallbacks"):
            stream = self.client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks=cfg["fallbacks"], **params)
        else:
            stream = self.client.messages.stream(**params)
        with stream as s:
            msg = s.get_final_message()
        d = msg.to_dict()
        sd = d.get("stop_details")
        return {"id": d.get("id"), "model": d.get("model"), "content": d.get("content", []), "stop_reason": d.get("stop_reason"),
                "stop_details": sd, "usage": d.get("usage", {})}


def first_user_content(text, cfg):
    """The first user message, with a cache breakpoint so every later call reads it from the cache."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral", "ttl": cfg["cache_ttl"]}}]


# == reply parsing ==
def reply_text(content):
    return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")


def parse_reply(text):
    """(blocks [(lang, code)], prose before the first block)."""
    blocks = [(m.group(1).strip().lower(), m.group(2)) for m in FENCE.finditer(text)]
    opens = len(re.findall(r"^```", text, re.M))
    if opens % 2 == 1:  # an unclosed fence counts as a malformed block
        blocks.append(("unclosed", ""))
    first = FENCE.search(text)
    prose = text[: first.start()] if first else text
    return blocks, prose.strip()


def words(s):
    return len(re.findall(r"[A-Za-z][A-Za-z'\-]*", s))


# == the run ==
class Run:
    def __init__(self, cfg, run_dir, mock, log, crash_at=None):
        self.cfg, self.dir, self.mock, self.log = cfg, Path(run_dir), mock, log
        self.work, self.case = self.dir / "work", self.dir / "case"
        self.crash_at = crash_at
        self.state_path = self.dir / "state.json"
        self.kernel = None
        self.backend = P.make_backend(cfg, mock)
        self.editor = None

    # ---- setup
    def create(self):
        import build_first_message as B
        import case as K
        for d in ("turns", "conv", "requests", "refused", "edits/current", "edits/history"):
            (self.dir / d).mkdir(parents=True, exist_ok=True)
        (self.work / "edits" / "accepted").mkdir(parents=True, exist_ok=True)
        (self.work / "edits" / "pending").mkdir(parents=True, exist_ok=True)
        atomic_write(self.dir / "config.json", jdump(self.cfg))
        facts = B.collect_facts()
        info = K.build_case(self.cfg, self.dir, facts, log=self.log)
        msg = B.render(self.cfg, facts, info["knob"], info["rules"]["allowed_units"], info["rules"], len(info["sets"]), len(info["formats"]))
        atomic_write(self.dir / "first_message.md", msg)
        state = {"run_id": self.dir.name, "created": now(), "mode": self.cfg["mode"], "turn": 0, "conv": 1, "conv_turns": 0,
                 "python_runs": 0, "reads": 0, "behaves": 0, "requests": 0, "pass_k": 0, "spent_usd": 0.0, "costed_turns": [],
                 "uncertain_charges_usd": 0.0, "last_total_input": 0, "last_output": 0, "unexplained_streak": 0, "unexplained": 0,
                 "format_errors": 0, "format_streak": 0, "refusals": 0, "refusal_streak": 0, "nudges": 0, "handoffs": 0,
                 "kernel_restarts": 0, "notices": [], "phase": "send", "last_worst": None, "final_mode": False, "final_reason": None, "done": False, "stop_reason": None,
                 "next_message": None, "pending_result": None, "seen": [], "request_keys": [], "edit_hash": None,
                 "first_message_sha256": hashlib.sha256(msg.encode()).hexdigest()[:16], "case_sha256": info["case_sha256"],
                 "weights_slice": info["weights_manifest"]["source_snapshot"]}
        self.state = state
        self.save_edit_state(info["start_state"], first=True)
        self.state["next_message"] = None  # the first turn sends the first message
        self.save()
        self.log(f"run {self.dir.name} created: mode {self.cfg['mode']}, start knob {info['knob']['name']}, "
                 f"allowed units {info['rules']['allowed_units']}, first message {len(msg)} characters")

    def load(self):
        self.state = json.loads(self.state_path.read_text())

    def save(self):
        atomic_write(self.state_path, jdump(self.state))

    # ---- edit state (authoritative copy outside WORK; mirror inside WORK for the kernel)
    def read_edits(self, folder):
        st = {}
        for short, name in E.SHORT.items():
            p = folder / f"{short}.npz"
            st[name] = E.load_npz(p) if p.exists() else None
        return st

    def current_edit(self):
        return self.read_edits(self.dir / "edits" / "current")

    def start_edit(self):
        return self.read_edits(self.case / "start_edit")

    def save_edit_state(self, st, first=False):
        cur = self.dir / "edits" / "current"
        acc = self.work / "edits" / "accepted"
        for name, c in st.items():
            short = E.EDITABLE[name]
            for folder in (cur, acc):
                p = folder / f"{short}.npz"
                if c is None or len(c["units"]) == 0:
                    if p.exists():
                        p.unlink()
                else:
                    E.save_npz(p, c)
        h = E.state_hash(st)
        per, union = E.state_units(st)
        atomic_write(acc / "state.json", json.dumps({"edit_hash": h, "units_per_tensor": per, "distinct_units": union}))
        self.state["edit_hash"] = h
        return h

    def originals(self):
        if not hasattr(self, "_orig"):
            import case as K
            self._orig = K.Originals(json.loads((self.case / "kernel_config.json").read_text())["weights_file"])
        return self._orig

    def process_pending_edits(self):
        """The check that counts: every file the kernel left in WORK/edits/pending is validated here, outside the sandbox."""
        rules = json.loads((self.case / "kernel_config.json").read_text())["rules"]
        pend = self.work / "edits" / "pending"
        files = sorted(pend.glob("*")) if pend.exists() else []
        if not files:
            return []
        st, start, lines = self.current_edit(), self.start_edit(), []
        for p in files:
            rec = {"turn": self.state["turn"], "file": p.name}
            try:
                if p.suffix == ".json":
                    d = json.loads(p.read_text()[:10000])
                    name = d.get("name")
                    if name not in E.EDITABLE:
                        raise E.EditRefused(f"drop_edit of {name!r}: not editable")
                    st[name] = start.get(name)
                    rec.update(action="drop", tensor=E.EDITABLE[name], accepted=True)
                    lines.append(f"[harness] drop_edit({E.EDITABLE[name]}) accepted: back to the starting knob.")
                elif p.suffix == ".npz":
                    c = E.load_npz(p)
                    if c["name"] not in E.EDITABLE:
                        raise E.EditRefused(f"{c['name']!r} is not editable")
                    if c["install"].shape != (len(c["units"]), E.HIDDEN) or c["knob"].shape != c["install"].shape:
                        raise E.EditRefused("malformed edit file")
                    if len(c["units"]) and (c["units"].min() < 0 or c["units"].max() >= E.INTER):
                        raise E.EditRefused("unit ids out of range")
                    c = E.canonical(c, E.take_units(c["name"], self.originals()(c["name"]), c["units"]))
                    others = [u for n2, o in st.items() if n2 != c["name"] and o is not None for u in o["units"].tolist()]
                    errs = E.validate(c, rules, others)
                    if errs:
                        raise E.EditRefused("; ".join(errs))
                    st[c["name"]] = c
                    E.save_npz(self.dir / "edits" / "history" / f"t{self.state['turn']:03d}_{p.stem}.npz", c)
                    rec.update(action="save", tensor=E.EDITABLE[c["name"]], units=c["units"].tolist(), accepted=True)
                    lines.append(f"[harness] save_edit({E.EDITABLE[c['name']]}) accepted: units {c['units'].tolist()}.")
                else:
                    raise E.EditRefused("unknown file in edits/pending")
            except Exception as e:  # noqa: BLE001  (anything wrong with an untrusted file is a refusal)
                rec.update(accepted=False, reason=str(e)[:500])
                lines.append(f"[harness] {p.name}: REFUSED, nothing changed: {str(e)[:400]}")
            with open(self.dir / "edits" / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            p.unlink()
        h = self.save_edit_state(st)
        per, union = E.state_units(st)
        lines.append(f"[harness] your accepted edit now changes units {union} (per tensor {per}); edit hash {h}.")
        return lines

    # ---- kernel
    def ensure_kernel(self, announce_reason=None):
        from kernel import Kernel
        if self.kernel is None:
            kc = json.loads((self.case / "kernel_config.json").read_text())
            self.kernel = Kernel(python=C.path(self.cfg["python_executable"]), run_dir=self.dir, work=self.work, case=self.case,
                                 weights_file=kc["weights_file"], timeout_s=self.cfg["python_timeout_s"], grace_s=self.cfg["python_grace_s"],
                                 memory_gb=self.cfg["kernel_memory_gb"], max_chars=self.cfg["output_max_chars"],
                                 watchdog_s=self.cfg["kernel_watchdog_interval_s"], log=self.log)
            self.kernel.start()
            if announce_reason:
                self.state["kernel_restarts"] += 1
                self.state["notices"].append(f"[note from the experimenters: {announce_reason}, so your kernel restarted too and its "
                                             f"variables are gone; files in WORK and your accepted edit are kept]")

    def redact(self, text):
        pairs = sorted({str(self.work.resolve()): "WORK", str(self.work): "WORK", str(self.case.resolve()): "CASE", str(self.case): "CASE",
                        str(C.path(self.cfg["python_executable"]).parent.parent): "<python>",
                        str(Path(os.path.realpath(C.path(self.cfg["python_executable"]))).parent.parent): "<python>",
                        str(C.BUILD): "<build>", str(Path.home()): "~"}.items(), key=lambda kv: -len(kv[0]))
        for a, b in pairs:
            text = text.replace(a, b)
        return text

    # ---- tools
    def counters(self):
        c = self.cfg
        s = self.state
        return (f"Python runs left: {c['python_runs'] - s['python_runs']}. Reads left: {c['reads'] - s['reads']}. "
                f"Behaviour passes left: {c['behaves'] - s['behaves']}.")

    def run_python(self, code):
        s = self.state
        if s["python_runs"] >= self.cfg["python_runs"]:
            return "No Python runs are left, so the block was not run.", "python_refused"
        s["python_runs"] += 1
        k = s["python_runs"]
        t0 = time.time()
        res = self.kernel.run(code, f"run{k}")
        out = self.redact(res.get("output") or "")
        head = f"OUTPUT of Python run {k} of at most {self.cfg['python_runs']} ({res.get('seconds')} s):"
        parts = []
        if res.get("restarted"):
            s["kernel_restarts"] += 1
            parts.append(f"[note from the experimenters: during this run the kernel was stopped ({res['restarted']}) and restarted; "
                         f"its variables are gone; files in WORK and your accepted edit are kept]")
        parts.append(head + "\n```\n" + out + ("\n" if out and not out.endswith("\n") else "") + "```")
        parts += self.process_pending_edits()
        try:
            size = sum(p.stat().st_size for p in self.work.rglob("*") if p.is_file())
            if size > self.cfg["work_max_gb"] * 1024 ** 3:
                parts.append(f"[harness] WORK holds {size / 1024 ** 3:.1f} GB, above the {self.cfg['work_max_gb']:g} GB limit: delete files you do not need.")
        except Exception:
            pass
        with open(self.dir / "tool_calls.jsonl", "a") as f:
            f.write(json.dumps({"turn": s["turn"], "kind": "python", "run": k, "code": code, "output": out, "ok": res.get("ok"),
                                "seconds": round(time.time() - t0, 2), "restarted": res.get("restarted"), "edit_hash": s["edit_hash"]}) + "\n")
        return "\n".join(parts), "python"

    def run_forward(self, body):
        s, cfg = self.state, self.cfg
        s["requests"] += 1
        try:
            req = json.loads(body)
        except json.JSONDecodeError as e:
            return f"The forward block is not valid JSON ({e.msg} at line {e.lineno}); it was not run and not counted.", "forward_invalid"
        sets = json.loads((self.case / "sets.json").read_text())
        formats = json.loads((self.case / "formats.json").read_text())
        rules = json.loads((self.case / "kernel_config.json").read_text())["rules"]
        try:
            norm, picks = P.validate_request(req, sets, formats, cfg, allowed_units=rules["allowed_units"])
        except P.RequestError as e:
            return f"The forward request was refused and not counted: {e}", "forward_invalid"
        kind = norm["type"]
        budget_key, budget = ("reads", cfg["reads"]) if kind == "read" else ("behaves", cfg["behaves"])
        if s[budget_key] >= budget:
            return f"No {'read' if kind == 'read' else 'behaviour'} passes are left; the request was not run.", "forward_refused"
        edit = self.current_edit()
        key = P.req_hash({"req": norm, "edit": s["edit_hash"] if P.uses_edit(norm) else None})
        if key in s["request_keys"]:
            return ("The forward request repeats an earlier one exactly (same request and same edit), so it was not run and "
                    "not counted."), "forward_repeat"
        k = s["pass_k"] + 1
        done_path = self.dir / "requests" / f"fwd_{k}.done.json"
        atomic_write(self.dir / "requests" / f"fwd_{k}.json", jdump({"turn": s["turn"], "request": req, "normalised": norm,
                                                                       "edit_hash": s["edit_hash"], "key": key}))
        reused = False
        if done_path.exists() and json.loads(done_path.read_text()).get("turn") == s["turn"]:
            res = json.loads(done_path.read_text())["result"]
            reused = True
        else:
            seen = {tuple(x) for x in s["seen"]}
            try:
                res = self.backend.run(k, norm, picks, sets, edit, self.start_edit(), self.work, seen)
            except Exception as e:  # noqa: BLE001  (a failed pass is not counted)
                atomic_write(self.dir / "requests" / f"fwd_{k}.failed.json", jdump({"turn": s["turn"], "error": str(e)[:2000]}))
                return f"The forward pass failed and was not counted: {type(e).__name__}: {str(e)[:300]}", "forward_failed"
            atomic_write(done_path, jdump({"turn": s["turn"], "result": res}))
        s["pass_k"] = k
        s[budget_key] += 1
        s["request_keys"].append(key)
        if kind == "read":
            s["seen"] = sorted({tuple(x) for x in s["seen"]} | {(st, i) for st, ids in picks for i in ids})
        with open(self.dir / "tool_calls.jsonl", "a") as f:
            f.write(json.dumps({"turn": s["turn"], "kind": kind, "pass": k, "request": norm, "edit_hash": s["edit_hash"],
                                "reused": reused, "result": res}) + "\n")
        per, union = E.state_units(edit)
        used = f"Edit used in this pass: units {union}, edit hash {s['edit_hash']}." if P.uses_edit(norm) else "This pass did not use your edit."
        what = "read" if kind == "read" else "behaviour"
        return (f"Forward pass {k} done ({what}; {'read' if kind == 'read' else 'behaviour'} passes used: {s[budget_key]} of {budget}). "
                f"{used} Summary:\n```json\n{json.dumps(res, indent=1)}\n```"), kind

    # ---- conversation
    def conv_path(self):
        return self.dir / "conv" / f"c{self.state['conv']:02d}.jsonl"

    def conv_messages(self):
        p = self.conv_path()
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

    def conv_append(self, msg):
        with open(self.conv_path(), "a") as f:
            f.write(json.dumps(msg) + "\n")

    def handoff_text(self, last_result):
        s, cfg = self.state, self.cfg
        notes = ""
        np_ = self.work / "notes.md"
        if np_.exists():
            notes = np_.read_text(errors="replace")[-cfg["notes_tail_chars"]:]
        files = sorted(p.name for p in self.work.iterdir() if not p.name.startswith("."))[:80]
        per, union = E.state_units(self.current_edit())
        return (f"\n\nTHIS IS A RESTARTED CONVERSATION (to keep it short): {s['turn']} replies happened before. "
                f"Read passes used: {s['reads']} of {cfg['reads']}; behaviour passes used: {s['behaves']} of {cfg['behaves']}; "
                f"Python runs used: {s['python_runs']} of {cfg['python_runs']}. Your kernel is still running with its variables. "
                f"Your accepted edit changes units {union} (per tensor {per}); edit hash {s['edit_hash']}. Files in WORK: {files}.\n\n"
                f"YOUR NOTEBOOK (WORK/notes.md, last {cfg['notes_tail_chars']} characters):\n----- begin notes -----\n{notes}\n----- end notes -----\n\n"
                f"Continue from here. Last result:\n{last_result or '(none)'}")

    # ---- the loop
    def stop(self, reason):
        self.state["done"] = True
        self.state["stop_reason"] = reason
        self.state["paused"] = None
        self.save()
        self.log(f"STOP: {reason}. Spent ${self.state['spent_usd']:.4f} of ${self.cfg['dollar_cap']:.2f}.")

    def pause(self, pt):
        """pause_at_turn: a clean stop between turns (the run is not done; the next message is saved and sent on resume)."""
        s = self.state
        msg = (f"PAUSED after turn {s['turn']} (pause_at_turn = {pt}). Spent ${s['spent_usd']:.4f} of ${self.cfg['dollar_cap']:.2f}. "
               f"Read the run in wsp/runs/{self.dir.name}/, then resume with: .venv/bin/python wsp/driver.py --resume {self.dir.name} "
               f"--set pause_at_turn=null   (or --set pause_at_turn=<a later turn>)")
        s["paused"] = msg
        self.save()
        self.log(msg)

    def loop(self):
        """Phases: 'send' (build the next message and call), 'sending' (the message is in the conversation, the reply is not
        recorded yet), 'replied' (the reply is recorded, its block not yet handled). A resumed run picks up at the phase it stopped in:
        a recorded reply is reused, never asked again, and its block is run again on the fresh kernel."""
        s = self.state
        while not s["done"]:
            if s["phase"] == "send":
                pt = self.cfg.get("pause_at_turn")
                if pt is not None and s["turn"] >= pt:
                    self.pause(pt)
                    return
                if not self.send():
                    return
            if s["phase"] == "sending":
                if not self.receive():
                    return
            if s["phase"] == "replied":
                self.act()

    def send(self):
        s, cfg = self.state, self.cfg
        if s["turn"] >= cfg["max_turns"]:
            self.stop(f"max_turns ({cfg['max_turns']}) reached")
            return False
        new_conv = s["conv_turns"] == 0
        if not new_conv and (s["conv_turns"] >= cfg["handoff_every"] or s["last_total_input"] >= cfg["handoff_max_input_tokens"]):
            s["conv"] += 1
            s["conv_turns"] = 0
            s["handoffs"] += 1
            new_conv = True
            self.log(f"t{s['turn'] + 1:03d}: new conversation c{s['conv']:02d} (handoff {s['handoffs']})")
        body = s["next_message"] or ""
        if s["notices"]:
            body = "\n".join(s["notices"]) + ("\n\n" + body if body else "")
        # the dollar cap: stop before a call that could exceed it, and keep one call in reserve for the final answer
        if new_conv:
            text = (self.dir / "first_message.md").read_text() + (self.handoff_text(body) if s["turn"] > 0 else "")
            worst = worst_case_next(s, 0, cfg, fresh_conv_chars=len(SYSTEM) + len(text))
        else:
            text = body
            worst = worst_case_next(s, len(body), cfg)
        if s["spent_usd"] + worst > cfg["dollar_cap"]:
            self.stop(f"dollar cap: the next call could cost up to ${worst:.2f} and ${s['spent_usd']:.2f} of ${cfg['dollar_cap']:.2f} is spent")
            return False
        if not s["final_mode"] and s["spent_usd"] + 2 * worst > cfg["dollar_cap"]:
            s["final_mode"], s["final_reason"] = True, "dollar cap"
            text = text + ("\n\n" if text else "") + END_NOTE
            self.log(f"t{s['turn'] + 1:03d}: final mode (dollar cap reserve: the next call could cost up to ${worst:.2f}; spent ${s['spent_usd']:.2f})")
        s["turn"] += 1
        t = s["turn"]
        tdir = self.dir / "turns" / f"t{t:03d}"
        tdir.mkdir(parents=True, exist_ok=True)
        if new_conv:
            if self.conv_path().exists():
                self.conv_path().unlink()
            self.conv_append({"role": "user", "content": first_user_content(text, cfg)})
        else:
            self.conv_append({"role": "user", "content": text})
        atomic_write(tdir / "message_sent.md", text)
        s["notices"], s["next_message"], s["phase"], s["last_worst"] = [], None, "sending", worst
        self.save()
        return True

    def receive(self):
        s, cfg = self.state, self.cfg
        t = s["turn"]
        tdir = self.dir / "turns" / f"t{t:03d}"
        rec = tdir / "reply.json"
        if rec.exists():
            reply = json.loads(rec.read_text())
            self.log(f"t{t:03d}: reusing the recorded answer")
        else:
            if s["notices"]:  # a restart while this turn's message was out: say so in an appended message (history stays append-only)
                note = "\n".join(s["notices"])
                self.conv_append({"role": "user", "content": note})
                with open(tdir / "message_sent.md", "a") as f:
                    f.write("\n\n" + note)
                s["notices"] = []
                if s["spent_usd"] + worst_case_next(s, len(note), cfg) > cfg["dollar_cap"]:
                    self.stop("dollar cap reached before the resent call")
                    return False
            reply = self.call_editor(t, tdir)
            if reply is None:
                return False
        if t not in s["costed_turns"]:
            cost = call_cost(reply.get("usage"), cfg)
            s["spent_usd"] += cost
            s["costed_turns"].append(t)
            atomic_write(tdir / "usage.json", jdump({"usage": reply.get("usage"), "cost_usd": cost, "spent_usd": s["spent_usd"],
                                                     "worst_case_estimate_usd": s.get("last_worst"), "conv": s["conv"], "model": reply.get("model")}))
        u = reply.get("usage") or {}
        s["last_total_input"] = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
        s["last_output"] = u.get("output_tokens") or 0
        s["conv_turns"] += 1
        s["phase"] = "replied"
        self.save()
        if self.crash_at == (t, "after_reply"):
            self.log(f"t{t:03d}: simulated crash after the reply was recorded (--crash-at)")
            os._exit(3)
        return True

    def act(self):
        s, cfg = self.state, self.cfg
        t = s["turn"]
        tdir = self.dir / "turns" / f"t{t:03d}"
        reply = json.loads((tdir / "reply.json").read_text())
        result, kind = self.handle(reply, tdir, t)
        atomic_write(tdir / "tool_result.md", result or "")
        self.log(f"t{t:03d}: {kind} | runs {s['python_runs']} reads {s['reads']} behaves {s['behaves']} | ${s['spent_usd']:.4f} | edit {s['edit_hash']}")
        if s["done"]:
            return
        if self.crash_at == (t, "after_tool"):
            self.log(f"t{t:03d}: simulated crash after the tool ran, before the state was saved (--crash-at)")
            os._exit(3)
        if not s["final_mode"]:
            passes_done = s["reads"] >= cfg["reads"] and s["behaves"] >= cfg["behaves"]
            if passes_done or s["python_runs"] >= cfg["python_runs"]:
                s["final_mode"], s["final_reason"] = True, "budgets used"
                tail = ("All forward passes are used. Make your final choice of edits (Python is still available), then reply "
                        "without any block." if s["python_runs"] < cfg["python_runs"] else "All Python runs are used. Reply without any block.")
                result = (result or "") + f"\n\n[note from the experimenters] {tail}"
        s["next_message"] = (result or "") + "\n\n(" + self.counters() + ")"
        s["phase"] = "send"
        self.save()

    def call_editor(self, t, tdir):
        s, cfg = self.state, self.cfg
        messages = self.conv_messages()
        for attempt in range(cfg["api_retries"] + 1):
            t0 = time.time()
            try:
                reply = self.editor.call(SYSTEM, messages, cfg, s["conv"], turn=t)
            except Exception as e:  # noqa: BLE001
                # a call that failed without usage may still have been billed for streamed output: charge the worst case
                worst = worst_case_next(s, 0, cfg)
                s["spent_usd"] += worst
                s["uncertain_charges_usd"] += worst
                self.save()
                self.log(f"t{t:03d}: API call failed ({type(e).__name__}: {str(e)[:200]}); charged the worst case ${worst:.2f}")
                status = getattr(e, "status_code", None)
                if (status is not None and 400 <= status < 500 and status not in (408, 409, 429)) or attempt == cfg["api_retries"]:
                    self.stop(f"API error: {type(e).__name__} (status {status})")
                    return None
                if s["spent_usd"] + worst_case_next(s, 0, cfg) > cfg["dollar_cap"]:
                    self.stop("dollar cap reached while retrying a failed call")
                    return None
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            reply["wall_s"] = round(time.time() - t0, 1)
            atomic_write(tdir / "reply.json", jdump(reply))
            return reply
        return None

    def handle(self, reply, tdir, t):
        s, cfg = self.state, self.cfg
        content = reply.get("content") or []
        atomic_write(tdir / "reply_text.md", reply_text(content))
        sr = reply.get("stop_reason")
        if sr == "refusal":
            s["refusals"] += 1
            s["refusal_streak"] += 1
            cat = (reply.get("stop_details") or {}).get("category")
            shutil.copy2(tdir / "reply.json", self.dir / "refused" / f"t{t:03d}.json")
            if s["refusal_streak"] > cfg["max_consecutive_refusals"]:
                self.stop(f"{s['refusal_streak']} refusals in a row (last category {cat})")
                return "", "refusal"
            # the refused turn is not appended to the conversation; the next user message follows the previous one
            return (f"[note from the experimenters] Your previous reply was stopped by the provider's safety filter (category: {cat}) "
                    f"before it reached us, so nothing was run. Please continue with the task, and keep to numbers and ids."), "refusal"
        s["refusal_streak"] = 0
        msgs = self.conv_messages()
        if not (msgs and msgs[-1]["role"] == "assistant" and msgs[-1]["content"] == content):
            self.conv_append({"role": "assistant", "content": content})
        text = reply_text(content)
        blocks, prose = parse_reply(text)
        with open(self.dir / "explanations.md", "a") as f:
            f.write(f"\n## t{t:03d}\n\n{prose if prose else '(no explanation)'}\n")
        if sr == "max_tokens":
            return self.format_error(f"Your reply was cut at the output limit ({cfg['max_tokens']} tokens) before it ended, so nothing was "
                                     f"run. Please send a shorter reply with one block.")
        if not blocks:
            if s["final_mode"]:
                atomic_write(self.dir / "final_answer.md", text)
                self.stop(f"final answer received ({s['final_reason']})")
                return "", "final"
            s["nudges"] += 1
            if s["nudges"] > cfg["max_nudges"] or cfg["allow_early_finish"]:
                atomic_write(self.dir / "final_answer.md", text)
                self.stop("the editor finished early (no block)")
                return "", "final"
            return f"Your reply had no block. {self.counters()} The session continues until they are used.", "no_block"
        if len(blocks) > 1 or blocks[0][0] not in ("python", "forward"):
            what = f"{len(blocks)} blocks" if len(blocks) > 1 else f"a block of kind {blocks[0][0]!r}"
            return self.format_error(f"Your reply had {what}; it must have exactly one ```python or ```forward block. Nothing was run.")
        s["format_streak"] = 0
        if s["final_mode"] and s["final_reason"] == "dollar cap":
            atomic_write(self.dir / "final_answer.md", text)
            self.stop("dollar cap: the reply to the end-of-session note carried a block, which was not run")
            return "", "final"
        lang, code = blocks[0]
        atomic_write(tdir / "block.txt", f"{lang}\n{code}")
        if words(prose) < cfg["min_explanation_words"]:
            if s["unexplained_streak"] < cfg["max_unexplained_in_a_row"]:
                s["unexplained_streak"] += 1
                s["unexplained"] += 1
                return ("Your reply had no explanation, so its block was NOT run (no Python run or forward pass was used). EXPLAIN "
                        "EVERY STEP: before the block, write 2 to 6 plain sentences for the experimenters on what the last result "
                        "showed, what this block does and why, and what outcome would change your plan. Send the block again, "
                        "preceded by your explanation."), "unexplained"
            note = "[harness] This block had no explanation; it was run anyway. Please explain every step.\n"
        else:
            note = ""
        s["unexplained_streak"] = 0
        if s["final_mode"] and lang == "forward":
            return "The forward passes are used; this block was not run. Reply without any block when you are done.", "forward_refused"
        if lang == "python":
            out, kind = self.run_python(code)
        else:
            out, kind = self.run_forward(code)
        return note + out, kind

    def format_error(self, msg):
        s = self.state
        s["format_errors"] += 1
        s["format_streak"] += 1
        if s["format_streak"] > self.cfg["max_consecutive_format_errors"]:
            self.stop(f"{s['format_streak']} format errors in a row")
            return "", "format_error"
        return msg, "format_error"


# == main ==
def preflight(cfg, mock):
    """What a real run needs that may be missing. Returns a list of problems (empty = ready)."""
    probs = []
    if not mock:
        if cfg["backend"] == "stub":
            probs.append("backend 'stub' gives fake numbers and runs only with --mock; set backend to 'remote'")
        if cfg["backend"] == "remote" and not cfg["remote"].get("queue_dir"):
            probs.append("remote.queue_dir is not set (the folder the GPU worker polls)")
        try:
            import anthropic  # noqa: F401
        except ImportError:
            probs.append("the anthropic package is not installed in this Python (pip install anthropic)")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            probs.append("ANTHROPIC_API_KEY is not set")
    if cfg["start_knob"] == "ayadial_improved" and not C.path(cfg["sources"]["ayadial_improved_vec"]).exists():
        probs.append(f"AyaDial-Improved's vector file is missing: {cfg['sources']['ayadial_improved_vec']} (copy cache_r5/vec_v3g.npz "
                     f"from the Drive mirror lexsi-knob/round5/cache/)")
    if not Path("/usr/bin/sandbox-exec").exists():
        probs.append("/usr/bin/sandbox-exec is missing (macOS only)")
    return probs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume")
    ap.add_argument("--run-id")
    ap.add_argument("--crash-at")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    crash = None
    if a.crash_at:
        tt, ph = a.crash_at.split(":")
        crash = (int(tt), ph)

    if a.resume:
        run_dir = WSP / "runs" / a.resume
        cfg = json.loads((run_dir / "config.json").read_text())
        changes = []
        for item in a.set:  # only a few keys may change after the start; the change is written back to config.json and logged
            key, _, val = item.partition("=")
            if key not in C.RESUME_SETTABLE:
                print(f"--set {key} is not allowed with --resume (the config is frozen at the start); allowed: {list(C.RESUME_SETTABLE)}")
                return 2
            new = C.parse_value(val)
            changes.append((key, cfg.get(key), new))
            cfg[key] = new
        if changes:
            C.validate(cfg)
            atomic_write(run_dir / "config.json", jdump(cfg))
        mock = cfg["mode"] == "mock"
        log = Log(run_dir / "driver.log", echo=not a.quiet)
        for key, old, new in changes:
            log(f"config changed on resume: {key} {old!r} -> {new!r}")
        run = Run(cfg, run_dir, mock, log, crash)
        run.load()
        if run.state["done"]:
            print(f"run {a.resume} has already ended: {run.state['stop_reason']}")
            return 0
        run.state["paused"] = None
        log(f"resume of {a.resume} at turn {run.state['turn']} (phase {run.state['phase']})")
        if run.state["phase"] == "replied":
            log(f"t{run.state['turn']:03d}: reusing the recorded answer (its block runs again on the fresh kernel)")
        reason = "the harness was restarted"
    else:
        cfg_file = a.config or (WSP / "configs" / ("mock.json" if a.mock else "pilot.json"))
        cfg = C.load(cfg_file, a.set)
        mock = a.mock
        if mock:
            cfg["mode"] = "mock"
            if cfg["backend"] != "stub":
                print("--mock uses the stub backend")
                cfg["backend"] = "stub"
        if a.dry_run:
            import build_first_message as B
            import case as K
            facts = B.collect_facts()
            sets, formats = K.catalogue()
            msg = B.render(cfg, facts, B.knob_meta_without_case(cfg), K.allowed_units(cfg), K.rules(cfg), len(sets), len(formats))
            print("=== CONFIG ===")
            print(jdump(cfg))
            print("=== PREFLIGHT ===")
            for p_ in preflight(cfg, mock) or ["ready"]:
                print("-", p_)
            print(f"=== FIRST MESSAGE ({len(msg)} characters) ===")
            print(msg)
            return 0
        probs = preflight(cfg, mock)
        if probs:
            print("not starting:\n- " + "\n- ".join(probs))
            return 2
        run_id = a.run_id or cfg.get("run_id") or f"{cfg['mode']}-{time.strftime('%Y%m%d-%H%M%S')}"
        cfg["run_id"] = run_id
        run_dir = WSP / "runs" / run_id
        if run_dir.exists():
            print(f"{run_dir} exists; use --resume {run_id} or another --run-id")
            return 2
        run_dir.mkdir(parents=True)
        log = Log(run_dir / "driver.log", echo=not a.quiet)
        run = Run(cfg, run_dir, mock, log, crash)
        run.create()
        reason = None
    if mock:
        from mock_editor import MockEditor
        run.editor = MockEditor()
    else:
        run.editor = AnthropicEditor(cfg)
    try:
        pt = cfg.get("pause_at_turn")
        if not (pt is not None and run.state["phase"] == "send" and run.state["turn"] >= pt):  # no kernel for a run that pauses at once
            run.ensure_kernel(announce_reason=reason)
        run.save()
        run.loop()
    finally:
        if run.kernel is not None:
            run.kernel.stop()
    s = run.state
    summary = {k: s[k] for k in ("run_id", "turn", "conv", "handoffs", "python_runs", "reads", "behaves", "spent_usd",
                                 "uncertain_charges_usd", "unexplained", "format_errors", "refusals", "nudges", "kernel_restarts",
                                 "stop_reason", "edit_hash")}
    summary["paused"] = s.get("paused")
    atomic_write(run_dir / "summary.json", jdump(summary))
    print(jdump(summary))
    if s.get("paused"):
        print(s["paused"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
