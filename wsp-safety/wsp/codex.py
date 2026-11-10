"""One model turn through the Codex CLI on the user's ChatGPT subscription (no API key, no per-token billing).

Reproduces the paper's call (lm_understand-logs, every turn's run.json):
    codex exec [resume] --strict-config --skip-git-repo-check --json -m gpt-6-astra
        -c model_reasoning_effort="<effort>" -c model_context_window=872000 -c model_auto_compact_token_limit=1000000
        -c model_instructions_file="<instructions.md>" -o <turn>/answer.md -c model_reasoning_summary="detailed"
        [<thread>] -
with the prompt on stdin. The paper used a dedicated CODEX_HOME holding harness/codex_config.toml; we keep the user's
CODEX_HOME (so the ChatGPT login and its token refresh stay in one place) and pass every key of
harness/codex_config.toml as a -c override instead. Nothing in ~/.codex is edited.

Every turn is audited from its event stream: it is tool-free when the only items are the model's messages, its
reasoning summaries and Codex's own error notices (e.g. "code mode is disabled").
"""
import json, os, re, shutil, subprocess, time, tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "harness" / "codex_config.toml"
INSTRUCTIONS = ROOT / "harness" / "instructions.md"
CODEX = shutil.which("codex") or str(Path.home() / ".local/bin/codex")
MODEL = "gpt-6-astra"
ALLOWED_ITEMS = {"reasoning", "agent_message", "error"}
LIMIT_RE = re.compile(r"usage limit|rate limit|too many requests|\b429\b|try again (at|in)|quota", re.I)


def _toml(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return json.dumps(str(v))


def config_overrides(path=CONFIG, skip=("model", "model_reasoning_effort")):
    """harness/codex_config.toml flattened to -c key=value pairs."""
    cfg = tomllib.loads(Path(path).read_text())
    out = []

    def walk(d, pre):
        for k, v in d.items():
            key = f"{pre}{k}"
            if isinstance(v, dict):
                walk(v, key + ".")
            elif key not in skip:
                out.extend(["-c", f"{key}={_toml(v)}"])
    walk(cfg, "")
    return out


def build_cmd(answer_path, effort, thread=None, model=MODEL, instructions=INSTRUCTIONS, extra=()):
    cmd = [CODEX, "exec"] + (["resume"] if thread else [])
    cmd += ["--strict-config", "--skip-git-repo-check", "--json", "-m", model]
    cmd += config_overrides()
    cmd += ["-c", f'model_reasoning_effort="{effort}"',
            "-c", "model_context_window=872000", "-c", "model_auto_compact_token_limit=1000000",
            "-c", f'model_instructions_file="{instructions}"',
            "-o", str(answer_path), "-c", 'model_reasoning_summary="detailed"', *extra]
    if thread:
        cmd.append(thread)
    cmd.append("-")
    return cmd


def audit(events):
    ev_types, item_types, tool_items, errors = {}, {}, {}, []
    usage, thread, completed, failed = None, None, False, None
    for e in events:
        t = e.get("type", "?")
        ev_types[t] = ev_types.get(t, 0) + 1
        if t == "thread.started":
            thread = e.get("thread_id")
        elif t == "turn.completed":
            completed, usage = True, e.get("usage")
        elif t == "turn.failed":
            failed = (e.get("error") or {}).get("message") or json.dumps(e)[:500]
        elif t == "error":
            errors.append(e.get("message") or json.dumps(e)[:500])
        if t.startswith("item."):
            it = e.get("item", {})
            ty = it.get("type", "?")
            if t == "item.completed":
                item_types[ty] = item_types.get(ty, 0) + 1
                if ty == "error":
                    errors.append(it.get("message", ""))
            if ty not in ALLOWED_ITEMS:
                tool_items[ty] = tool_items.get(ty, 0) + 1
    return {"event_types": ev_types, "item_types": item_types, "tool_items": tool_items,
            "tool_free": not tool_items and completed, "usage": usage, "thread_id": thread,
            "errors": errors, "failed": failed, "turn_completed": completed}


SESSIONS = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "sessions"
USAGE_MAX = float(os.environ.get("WSP_USAGE_MAX", 95))   # pause when the 5 h or weekly window is this % used
_credits_at_start = None


class CreditsSpent(RuntimeError):
    pass


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find(v, key)
            if r is not None:
                return r
    return None


def usage_status(n_files=3):
    """The newest rate_limits record Codex wrote to its session logs: 5 h window (primary), weekly window
    (secondary), credits. Account-wide, so any recent Codex session (ours or the user's) reports it."""
    files = sorted(SESSIONS.glob("*/*/*/rollout-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:n_files]
    for f in files:
        for line in reversed(f.read_text(errors="replace").splitlines()):
            if '"rate_limits"' not in line:
                continue
            try:
                rl = _find(json.loads(line), "rate_limits")
            except json.JSONDecodeError:
                continue
            if isinstance(rl, dict) and rl.get("primary"):
                return rl
    return None


def guard_usage(log=print):
    """Before a turn: stop if the credit balance has dropped since the driver started (credits are being spent);
    sleep until the window resets if the 5 h or weekly window is at USAGE_MAX % or more. Uses included plan usage only."""
    global _credits_at_start
    while True:
        rl = usage_status()
        if rl is None:
            return None
        bal = (rl.get("credits") or {}).get("balance")
        bal = float(bal) if bal is not None else None
        if bal is not None:
            if _credits_at_start is None:
                _credits_at_start = bal
            elif bal < _credits_at_start - 1e-6:
                raise CreditsSpent(f"credit balance fell from {_credits_at_start} to {bal}: stopping")
        # a window whose reset time has passed has reset: the logged percentage is stale (no call since), skip it
        waits = [(w["used_percent"], w["resets_at"], name) for name in ("primary", "secondary")
                 if (w := rl.get(name)) and w.get("used_percent", 0) >= USAGE_MAX
                 and w.get("resets_at", 0) > time.time()]
        if not waits:
            return rl
        until = max(r for _, r, _ in waits) + 60
        log(f"usage {[(n, p) for p, _, n in waits]} >= {USAGE_MAX}%: sleeping until "
            f"{time.strftime('%a %H:%M', time.localtime(until))}")
        time.sleep(max(60, until - time.time()))


def turn(prompt, turn_dir, effort, thread=None, model=MODEL, timeout=6 * 3600, max_attempts=4,
         limit_wait=900, max_limit_waits=48, cwd=None):
    """Send one message; return (answer_text, thread_id, run_record). Writes prompt.txt, events.jsonl, answer.md,
    run.json and stderr.log into turn_dir. Pauses near the plan's usage limits and refuses to spend credits."""
    turn_dir = Path(turn_dir)
    turn_dir.mkdir(parents=True, exist_ok=True)
    usage_before = guard_usage()
    (turn_dir / "prompt.txt").write_text(prompt)
    answer_path = turn_dir / "answer.md"
    cwd = Path(cwd or turn_dir / "cwd")      # an empty folder: nothing for the model to see even if a tool leaked
    cwd.mkdir(exist_ok=True)
    cmd = build_cmd(answer_path, effort, thread, model)
    waits, attempt, rec = [], 0, None
    while True:
        attempt += 1
        answer_path.unlink(missing_ok=True)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=cwd)
            rc, stdout, stderr = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as ex:
            rc, stdout, stderr = -9, (ex.stdout or b"").decode() if isinstance(ex.stdout, bytes) else (ex.stdout or ""), "timeout"
        events = []
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        (turn_dir / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
        (turn_dir / "stderr.log").write_text(stderr)
        a = audit(events)
        # The reply is EVERY non-empty agent_message of the turn, joined in order. The model can split one reply
        # over several messages (e.g. explanation + code block, then "Please run the Python block above"), and -o
        # keeps only the last one (kept as answer_last.md).
        last = answer_path.read_text() if answer_path.exists() else ""
        msgs = [e["item"].get("text", "") for e in events
                if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "agent_message"]
        msgs = [m for m in msgs if m.strip()]
        answer = "\n\n".join(msgs) if msgs else last
        a["n_agent_messages"] = len(msgs)
        (turn_dir / "answer_last.md").write_text(last)
        answer_path.write_text(answer)
        ok = rc == 0 and a["turn_completed"] and answer.strip() != ""
        blob = " ".join([stderr[-3000:], a["failed"] or "", *a["errors"]])
        limit = (not ok) and bool(LIMIT_RE.search(blob))
        rec = {"ok": ok, "limit": limit, "rc": rc, "wall_s": round(time.time() - t0, 1), "audit": a,
               "stderr_tail": stderr[-2000:], "cmd": cmd, "attempt": attempt, "waits": waits,
               "thread_in": thread, "end": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "usage_before": usage_before}
        (turn_dir / "run.json").write_text(json.dumps(rec, indent=1))
        if ok:
            return answer, a["thread_id"] or thread, rec
        if limit and len(waits) < max_limit_waits:
            waits.append({"t": time.time(), "s": limit_wait, "why": blob[-300:]})
            time.sleep(limit_wait)
            attempt -= 1
            continue
        if attempt >= max_attempts:
            raise RuntimeError(f"codex turn failed after {attempt} attempts: rc={rc} {blob[-800:]}")
        time.sleep(30 * attempt)
