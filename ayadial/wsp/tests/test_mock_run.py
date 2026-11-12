"""End-to-end test of the driver with the scripted fake editor and the stub backend. No network, no API key, no cost.
Run from build/:  .venv/bin/python wsp/tests/test_mock_run.py

It runs one mock session in three pieces, the way a crash would cut it:
  1. driver --mock --crash-at 7:after_reply    the process dies right after turn 7's reply is recorded (exit code 3)
  2. driver --resume --crash-at 13:after_tool   resumes, reuses turn 7's recorded reply, dies again after turn 13's block ran
  3. driver --resume                            resumes again and runs until the dollar cap ends the session
The driver's environment holds a FAKE ANTHROPIC_API_KEY throughout; the test then checks that it appears in no file of the run.
Writes wsp/tests/mock_report.json and prints one PASS / FAIL line per check. Exit code 1 if any check fails.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WSP = Path(__file__).resolve().parents[1]
BUILD = WSP.parent
PY = str(BUILD / ".venv/bin/python")
FAKE = "FAKE-KEY-FOR-MOCK-TEST-77d0aa"
sys.path.insert(0, str(WSP))


def drive(*args):
    env = dict(os.environ)
    env["ANTHROPIC_API_KEY"] = FAKE
    return subprocess.run([PY, str(WSP / "driver.py"), *args, "--quiet"], cwd=str(BUILD), env=env, capture_output=True, text=True, timeout=900)


def main():
    rid = "mocktest-" + time.strftime("%Y%m%d-%H%M%S")
    run = WSP / "runs" / rid
    results = []

    def check(name, ok, detail=""):
        results.append({"check": name, "pass": bool(ok), "detail": str(detail)[:400]})
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {str(detail)[:200]}" if detail else ""))

    r1 = drive("--mock", "--run-id", rid, "--crash-at", "7:after_reply")
    st = json.loads((run / "state.json").read_text())
    check("piece 1 stops with the simulated crash after turn 7's reply was recorded", r1.returncode == 3 and st["turn"] == 7 and st["phase"] == "replied",
          f"exit {r1.returncode}, turn {st['turn']}, phase {st['phase']}")
    r2 = drive("--resume", rid, "--crash-at", "13:after_tool")
    st = json.loads((run / "state.json").read_text())
    check("piece 2 resumes and stops with the simulated crash after turn 13's block ran", r2.returncode == 3 and st["turn"] == 13,
          f"exit {r2.returncode}, turn {st['turn']}, phase {st['phase']}")
    r3 = drive("--resume", rid)
    st = json.loads((run / "state.json").read_text())
    check("piece 3 resumes and the session ends", r3.returncode == 0 and st["done"], f"exit {r3.returncode}: {r3.stderr[-300:]}")
    cfg = json.loads((run / "config.json").read_text())
    log = (run / "driver.log").read_text()
    kinds = {}
    for line in log.splitlines():
        parts = line.split(" ", 2)
        if len(parts) == 3 and parts[2][:1] == "t" and ": " in parts[2]:
            t, rest = parts[2].split(": ", 1)
            kinds.setdefault(rest.split(" |")[0], []).append(t)
    sent = {p.parent.name: p.read_text() for p in sorted((run / "turns").glob("t*/message_sent.md"))}
    res = {p.parent.name: p.read_text() for p in sorted((run / "turns").glob("t*/tool_result.md"))}
    tool = [json.loads(l) for l in (run / "tool_calls.jsonl").read_text().splitlines()]
    elog = [json.loads(l) for l in (run / "edits" / "log.jsonl").read_text().splitlines()]

    check("a Python turn ran in the sandboxed kernel", any(t["kind"] == "python" and t["ok"] for t in tool), kinds.get("python"))
    check("a reply without explanation was sent back unrun", "unexplained" in kinds and "NOT run" in res.get(f"t{int(kinds['unexplained'][0][1:]):03d}", ""), kinds.get("unexplained"))
    check("a valid read pass ran and was counted (stub numbers, marked fake)", st["reads"] == 1 and any(t["kind"] == "read" and t["result"].get("stub") for t in tool))
    check("an invalid forward request was refused and not counted", "forward_invalid" in kinds)
    check("a behaviour pass ran and was counted", st["behaves"] == 1 and any(t["kind"] == "behave" for t in tool))
    check("an exact repeat of a pass was refused and not counted", "forward_repeat" in kinds)
    check("save_edit refused by the kernel's own check (layer 21; unit 5)", "kernel check: save_edit refused" in res.get("t006", ""))
    check("a forged edit file that bypassed the kernel was refused by the driver's check", any(not e["accepted"] and "allowed units" in e.get("reason", "") for e in elog), elog[:1])
    check("an allowed save_edit was accepted (unit 9173 added) and the edit hash changed",
          any(e["accepted"] and 9173 in e.get("units", []) for e in elog) and st["edit_hash"] != json.loads((run / "case" / "knob.json").read_text())["start_edit_hash"],
          st["edit_hash"])
    check("a reply with two blocks was a format error", "format_error" in kinds)
    check("a reply with no block got a nudge, not the end", "no_block" in kinds)
    hand = [k for k, v in sent.items() if "THIS IS A RESTARTED CONVERSATION" in v]
    check("handoffs happened: new conversations open with the task, counters, the accepted edit and the notebook tail",
          st["handoffs"] >= 2 and hand and "YOUR NOTEBOOK" in sent[hand[0]] and "edit hash" in sent[hand[0]], f"handoffs {st['handoffs']} at {hand}")
    import re
    surv = [f"t{t['turn']:03d}: fill {m.group(1)}" for t in tool if t["kind"] == "python" and f"t{t['turn']:03d}" in hand
            for m in [re.search(r"fill (\d+)", t.get("output", ""))] if m and int(m.group(1)) >= 2]
    check("the kernel and its variables survived a handoff (a counter kept counting across a new conversation)", surv, surv)
    check("resume reused the recorded answer of turn 7 instead of calling again", "t007: reusing the recorded answer" in log)
    restart_msgs = [k for k, v in sent.items() if "the harness was restarted, so your kernel restarted too" in v]
    check("both resumes announced the kernel restart to the editor", len(restart_msgs) == 2, restart_msgs)
    check("turn 13's block ran again after the second crash (state was not saved after it)", len(kinds.get("python", [])) and log.count("t013: python") == 2)
    costs = {p.parent.name: json.loads(p.read_text())["cost_usd"] for p in (run / "turns").glob("t*/usage.json")}
    check("every turn was charged exactly once (sum of per-turn costs equals the total)",
          abs(sum(costs.values()) + st["uncertain_charges_usd"] - st["spent_usd"]) < 1e-9 and len(costs) == st["turn"],
          f"{len(costs)} charged turns of {st['turn']}; ${st['spent_usd']:.4f}")
    check("the dollar cap ended the session (final answer asked for with one call in reserve; spend stayed under the cap)",
          "dollar cap" in (st["stop_reason"] or "") and st["spent_usd"] <= cfg["dollar_cap"] and (run / "final_answer.md").exists(),
          f"{st['stop_reason']}; ${st['spent_usd']:.2f} of ${cfg['dollar_cap']:.2f}")
    leaks = [str(p.relative_to(run)) for p in run.rglob("*") if p.is_file() and p.stat().st_size < 50_000_000 and FAKE.encode() in p.read_bytes()]
    check("the API key (a fake one in the driver's environment) appears in no file of the run", not leaks, leaks)
    home = str(Path.home())
    leaked_paths = [k for k, v in sent.items() if home in v or str(BUILD) in v]
    check("no absolute path of this machine reached a message sent to the editor", not leaked_paths, leaked_paths)
    import passes as P
    try:
        P.StubBackend(mock=False)
        ok = False
    except RuntimeError:
        ok = True
    check("StubBackend refuses to run outside --mock", ok)
    r = subprocess.run([PY, str(WSP / "driver.py"), "--config", str(WSP / "configs" / "pilot.json"), "--set", "backend=stub"], cwd=str(BUILD),
                       capture_output=True, text=True, timeout=120, env={k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"})
    check("a non-mock run with the stub backend does not start", r.returncode == 2 and "only with --mock" in r.stdout, r.stdout.strip()[-200:])
    # pause_at_turn: a clean stop after turn 4, a resume that pauses again at once, a refused --set, then a resume to the end
    rid2 = rid.replace("mocktest-", "mockpause-")
    run2 = WSP / "runs" / rid2
    p1 = drive("--mock", "--run-id", rid2, "--set", "pause_at_turn=4")
    st2 = json.loads((run2 / "state.json").read_text())
    hint = f"--resume {rid2} --set pause_at_turn=null"
    check("pause_at_turn=4 stops cleanly after turn 4 and says how to resume",
          p1.returncode == 0 and st2["turn"] == 4 and st2["phase"] == "send" and not st2["done"] and hint in p1.stdout
          and st2.get("paused") and hint in (run2 / "driver.log").read_text(), f"exit {p1.returncode}, turn {st2['turn']}, phase {st2['phase']}")
    p2 = drive("--resume", rid2)
    st2b = json.loads((run2 / "state.json").read_text())
    check("a resume without changing pause_at_turn pauses again before any call (no new turn, no kernel restart)",
          p2.returncode == 0 and st2b["turn"] == 4 and st2b["kernel_restarts"] == st2["kernel_restarts"] and hint in p2.stdout,
          f"exit {p2.returncode}, turn {st2b['turn']}")
    p3 = drive("--resume", rid2, "--set", "dollar_cap=100")
    check("--resume refuses a --set of a frozen key", p3.returncode == 2 and "not allowed with --resume" in p3.stdout, p3.stdout.strip()[-200:])
    p4 = drive("--resume", rid2, "--set", "pause_at_turn=null")
    st2c = json.loads((run2 / "state.json").read_text())
    cfg2 = json.loads((run2 / "config.json").read_text())
    check("--resume --set pause_at_turn=null continues past turn 4 to the end; the change is written to config.json and logged",
          p4.returncode == 0 and st2c["done"] and st2c["turn"] > 4 and cfg2["pause_at_turn"] is None and not st2c.get("paused")
          and "config changed on resume: pause_at_turn 4 -> None" in (run2 / "driver.log").read_text(),
          f"exit {p4.returncode}, turn {st2c['turn']}, stop {st2c['stop_reason']}")
    rep = {"run_id": rid, "results": results, "passed": sum(x["pass"] for x in results), "total": len(results),
           "summary": json.loads((run / "summary.json").read_text())}
    (WSP / "tests" / "mock_report.json").write_text(json.dumps(rep, indent=1))
    print(f"{rep['passed']} of {rep['total']} mock-run checks passed (run {rid})")
    return 0 if rep["passed"] == rep["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
