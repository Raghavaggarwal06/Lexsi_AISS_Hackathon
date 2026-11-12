"""Sandbox tests for the editor's kernel. Run from build/:  .venv/bin/python wsp/tests/test_sandbox.py

The script re-runs itself as a child whose environment holds a FAKE ANTHROPIC_API_KEY, so the kernel's parent process really has a
key in its environment, as the driver will. It then starts the real sandboxed kernel (case folder built with start_knob=ayadial) and
runs probes inside it. Every probe that must fail has a positive control that must succeed, so a "blocked" result cannot come from
an unrelated error. Writes wsp/tests/sandbox_report.json and prints one PASS / FAIL line per check. Exit code 1 if any check fails.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WSP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WSP))
FAKE = "FAKE-KEY-FOR-SANDBOX-TEST-5b1c9e"


def probe(kern, code, i=[0]):
    i[0] += 1
    return kern.run(code, f"probe{i[0]}")


def main():
    import config as C
    import case as K
    from kernel import Kernel
    cfg = C.load(WSP / "configs" / "mock.json")
    run_dir = WSP / "runs" / "_sandbox_test"
    work = run_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    info = K.build_case(cfg, run_dir, facts={}, log=lambda *a: None)
    build = C.BUILD
    home = Path.home()
    kern = Kernel(python=C.path(cfg["python_executable"]), run_dir=run_dir, work=work, case=Path(info["case"]),
                  weights_file=info["weights_file"], timeout_s=4, grace_s=3, memory_gb=1.5, max_chars=4000)
    kern.start()
    results = []

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    def blocked(out):
        return ("PermissionError" in out or "Operation not permitted" in out or "BLOCKED" in out) and "LEAKED" not in out

    # (a) network
    r = probe(kern, """
import socket
res = []
for host, port in [("1.1.1.1", 443), ("127.0.0.1", 22)]:
    try:
        s = socket.create_connection((host, port), timeout=3); s.close(); res.append(f"CONNECTED {host}:{port} LEAKED")
    except Exception as e:
        res.append(f"BLOCKED {host}:{port} {type(e).__name__}: {e}")
try:
    socket.getaddrinfo("example.com", 443); res.append("DNS LEAKED")
except Exception as e:
    res.append(f"BLOCKED dns {type(e).__name__}")
try:
    u = socket.socket(socket.AF_UNIX); u.connect("/var/run/mDNSResponder"); res.append("UNIX LEAKED")
except Exception as e:
    res.append(f"BLOCKED unix {type(e).__name__}")
print(res)
""")
    out = r["output"]
    check("(a) network connection blocked (TCP to 1.1.1.1:443 and 127.0.0.1:22, DNS, unix socket to mDNSResponder)",
          out.count("BLOCKED") == 4 and "LEAKED" not in out, out.strip()[-400:])

    # positive control: reading the case folder and the weights works
    r = probe(kern, "print(len((CASE/'knob.json').read_text()) > 0, tuple(load('model.layers.22.mlp.gate_proj.weight').shape), len(NAMES))")
    check("control: kernel can read CASE and the weight slice", "True (11008, 2048) 40" in r["output"], r["output"].strip())

    # (b) runs_private/ and data/
    targets = [build / "runs_private/round4/adapters/fold_cap/install/adapter_config.json", build / "data/README.md",
               build / "runs_private/round6/sets.jsonl"]
    for t in targets:
        r = probe(kern, f"""
try:
    print("LEAKED", len(open({str(t)!r}).read()))
except Exception as e:
    print("BLOCKED", type(e).__name__)
try:
    import os; print("LISTED LEAKED", len(os.listdir({str(t.parent)!r})))
except Exception as e:
    print("BLOCKED listdir", type(e).__name__)
""")
        check(f"(b) read blocked: {t.relative_to(build)} (and listing its folder)", blocked(r["output"]) and t.exists(), r["output"].strip())

    # (c) ~/.zshrc
    z = home / ".zshrc"
    r = probe(kern, f"""
try:
    print("LEAKED", len(open({str(z)!r}).read()))
except Exception as e:
    print("BLOCKED read", type(e).__name__)
import os
try:
    os.stat({str(z)!r}); print("STAT visible (metadata only)")
except Exception as e:
    print("BLOCKED stat", type(e).__name__)
""")
    check("(c) read of ~/.zshrc blocked", "BLOCKED read" in r["output"] and z.exists(), r["output"].strip())
    check("(c') stat of ~/.zshrc blocked too (no metadata outside the allowed paths)", "BLOCKED stat" in r["output"], r["output"].strip())

    # (d) writes
    outside = [WSP / "SANDBOX_WRITE_TEST.txt", Path("/tmp/wsp_sandbox_write_test.txt"), home / "wsp_sandbox_write_test.txt",
               Path(info["case"]) / "knob.json"]
    code = "res=[]\n"
    for p in outside:
        code += f"""
try:
    open({str(p)!r}, 'a').write('x'); res.append('WROTE LEAKED {p.name}')
except Exception as e:
    res.append('BLOCKED ' + type(e).__name__)
"""
    code += "open(WORK/'inside.txt','w').write('ok'); res.append('inside ok'); print(res)"
    r = probe(kern, code)
    leaked = [p for p in outside[:3] if p.exists()]
    for p in leaked:
        p.unlink()
    check("(d) writes outside WORK blocked (wsp/, /tmp, home, CASE) while WORK is writable",
          r["output"].count("BLOCKED") == 4 and "inside ok" in r["output"] and not leaked, r["output"].strip())

    # (e) environment and other processes' environment
    r = probe(kern, f"""
import os, ctypes
print("env key present:", "ANTHROPIC_API_KEY" in os.environ, "| env keys:", sorted(os.environ))
libc = ctypes.CDLL(None, use_errno=True)
def procargs(pid):
    mib = (ctypes.c_int*3)(1, 49, pid); size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0: return None
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0: return None
    return buf.raw[:size.value]
me = procargs(os.getpid())
print("own procargs readable (control):", me is not None)
par = procargs(os.getppid())
print("parent procargs:", "LEAKED" if par and {FAKE!r}.encode() in par else ("BLOCKED" if par is None else "readable, no key"))
hits = 0; readable = 0
for pid in range(1, 100000):
    b = procargs(pid)
    if b is not None and pid != os.getpid():
        readable += 1
        if {FAKE!r}.encode() in b: hits += 1
print("scan of pids 1..99999: other processes readable", readable, "| with the key", hits)
import subprocess
for cmd in (["/bin/ps", "eww"], ["/usr/bin/env"]):
    try:
        subprocess.run(cmd, capture_output=True, timeout=5); print("EXEC LEAKED", cmd[0])
    except Exception as e:
        print("BLOCKED exec", cmd[0], type(e).__name__)
try:
    pid = os.fork()
    if pid == 0: os._exit(0)
    print("FORK LEAKED")
except Exception as e:
    print("BLOCKED fork", type(e).__name__)
""")
    o = r["output"]
    check("(e) ANTHROPIC_API_KEY not in the kernel's environment", "env key present: False" in o, o.splitlines()[0] if o else o)
    check("(e') the parent's environment (which holds a key) is not readable via sysctl kern.procargs2; no process is",
          "own procargs readable (control): True" in o and "parent procargs: BLOCKED" in o and "readable 0 | with the key 0" in o,
          " | ".join(l for l in o.splitlines()[1:4]))
    check("(e'') no exec of other programs and no fork", o.count("BLOCKED exec") == 2 and "BLOCKED fork" in o and "LEAKED" not in o,
          " | ".join(l for l in o.splitlines() if "exec" in l or "fork" in l))

    # limits: wall clock (interruptible), wall clock (ignores SIGINT), memory
    r = probe(kern, "x = 41\nwhile True:\n    pass")
    r2 = probe(kern, "x + 1")
    check("time limit: a busy loop is interrupted after the limit and the kernel keeps its variables",
          r["restarted"] is None and "interrupted" in r["output"] and "42" in r2["output"], f"{r['output'].strip()[-120:]} | next run: {r2['output'].strip()}")
    r = probe(kern, "import signal\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nwhile True:\n    pass")
    r2 = probe(kern, "print('x' in globals())")
    check("time limit: a loop that ignores SIGINT is killed after the grace period, the kernel restarts and says why",
          bool(r["restarted"]) and "time limit" in r["restarted"] and "False" in r2["output"], f"restarted: {r['restarted']} | next run sees old variables: {r2['output'].strip()}")
    r = probe(kern, "import numpy as np\nbig = np.ones(int(3e9) // 8)\nprint('allocated')")
    check("memory limit: a 3 GB allocation with a 1.5 GB limit kills the kernel, which restarts and says why",
          bool(r["restarted"]) and "memory limit" in r["restarted"], f"restarted: {r['restarted']}")
    r = probe(kern, "print(len(NAMES))")
    check("control: kernel works after the restarts", "40" in r["output"], r["output"].strip())
    kern.stop()

    # sensitivity control: the same parent-environment probe under the profile WITHOUT the sysctl-read / process-info deny must leak,
    # which shows (1) the probe can detect the leak and (2) sandbox-exec's defaults (system.sb) do not block it.
    weak = run_dir / "sandbox_weak.sb"
    weak.write_text("\n".join(l for l in kern.profile_text().splitlines() if not l.startswith("(deny sysctl-read process-info*)")
                              and not l.startswith("(allow process-info* (target self))")) + "\n")
    pcode = ("import os, ctypes\nlibc = ctypes.CDLL(None, use_errno=True)\nmib = (ctypes.c_int*3)(1, 49, os.getppid())\n"
             "size = ctypes.c_size_t(0)\nlibc.sysctl(mib, 3, None, ctypes.byref(size), None, 0)\nbuf = ctypes.create_string_buffer(size.value)\n"
             "r = libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0)\nprint('LEAKED' if r == 0 and " + repr(FAKE.encode()) + " in buf.raw else 'BLOCKED')\n")
    from kernel import scrubbed_env
    pr = subprocess.run(["/usr/bin/sandbox-exec", "-f", str(weak), str(C.path(cfg["python_executable"])), "-I", "-c", pcode],
                        capture_output=True, text=True, cwd=str(work), env=scrubbed_env(work), timeout=60)
    check("sensitivity control: without the sysctl-read + process-info deny, the same probe DOES read the parent's key",
          "LEAKED" in pr.stdout, (pr.stdout.strip() + " " + pr.stderr.strip()[-200:]).strip())
    rep = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "profile": str(run_dir / "sandbox.sb"), "results": results,
           "passed": sum(r["pass"] for r in results), "total": len(results)}
    (WSP / "tests" / "sandbox_report.json").write_text(json.dumps(rep, indent=1))
    print(f"{rep['passed']} of {rep['total']} sandbox checks passed")
    return 0 if rep["passed"] == rep["total"] else 1


if __name__ == "__main__":
    if os.environ.get("WSP_SANDBOX_CHILD") != "1":
        env = dict(os.environ)
        env["ANTHROPIC_API_KEY"] = FAKE   # the child (the kernel's parent) really holds a key, as the driver will
        env["WSP_SANDBOX_CHILD"] = "1"
        sys.exit(subprocess.run([sys.executable, __file__], env=env).returncode)
    sys.exit(main())
