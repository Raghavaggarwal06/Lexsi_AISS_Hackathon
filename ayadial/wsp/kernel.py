"""The editor's persistent Python kernel, and the driver-side client that runs it inside a macOS sandbox.

Two halves in one file:

  python kernel.py serve --work W --case C      (runs INSIDE the sandbox; the driver copies this file and edits.py to C/lib/)
      A notebook-like loop: it reads one JSON command per line on stdin, runs the code in one persistent namespace, and answers
      one JSON line on a private copy of the original stdout. Everything the code prints is captured and cut to a limit.

  Kernel(...)                                   (runs in the driver, outside the sandbox)
      Writes the sandbox profile, starts the server under /usr/bin/sandbox-exec with a scrubbed environment, enforces the
      wall-clock limit (SIGINT, then SIGKILL) and the memory limit (an RSS watchdog: macOS does not let a process lower
      RLIMIT_AS or RLIMIT_DATA), restarts the kernel when it dies and reports why, so the driver can announce it.

The sandbox profile (build_profile) denies everything by default and then allows: exec of the Python interpreter only;
reads of the Python runtime (the venv and the interpreter's lib folder), of the weight slice file, of the run's case folder
and of the run's work folder; writes only inside the work folder; no network; no reading of other processes' arguments or
environment (kern.procargs2 is blocked only when sysctl-read AND process-info are both denied, which tests/test_sandbox.py checks).
"""
import ast
import io
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

# ==
# Server side (inside the sandbox)
# ==


class _Capture(io.TextIOBase):
    """Keeps the first `head` and the last `tail` characters of everything written, and the total count."""

    def __init__(self, head, tail):
        self.head_n, self.tail_n = head, tail
        self.head, self.tail, self.total = [], "", 0
        self.head_len = 0

    def writable(self):
        return True

    def write(self, s):
        s = str(s)
        self.total += len(s)
        if self.head_len < self.head_n:
            take = s[: self.head_n - self.head_len]
            self.head.append(take)
            self.head_len += len(take)
            s = s[len(take):]
        if s:
            self.tail = (self.tail + s)[-self.tail_n:]
        return len(s)

    def text(self):
        head = "".join(self.head)
        cut = self.total - len(head) - len(self.tail)
        if cut > 0:
            return head + f"\n[... output cut: {cut} characters not shown ...]\n" + self.tail, True
        return head + self.tail, False


def _run_code(code, ns, max_chars):
    cap = _Capture(int(max_chars * 0.6), max_chars - int(max_chars * 0.6))
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = cap
    ok, t0 = True, time.time()
    try:
        tree = ast.parse(code, "<block>", "exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, "<block>", "exec"), ns)
        if last is not None:
            val = eval(compile(last, "<block>", "eval"), ns)
            if val is not None:
                print(repr(val))
    except KeyboardInterrupt:
        ok = False
        print("\n[the run was interrupted: it exceeded the time limit]")
    except SystemExit:
        print("[exit() is ignored: the kernel keeps running]")
    except BaseException:  # the editor sees its own traceback
        ok = False
        traceback.print_exc()
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    text, truncated = cap.text()
    return {"ok": ok, "output": text, "truncated": truncated, "seconds": round(time.time() - t0, 3)}


def _prelude(work, case):
    """Build the editor's namespace. Everything here is documented in the first message."""
    import numpy as np
    import torch
    from safetensors import safe_open

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import edits as E

    torch.set_grad_enabled(False)
    kc = json.loads((case / "kernel_config.json").read_text())
    weights_file = kc["weights_file"]
    rules = kc["rules"]
    names = sorted(kc["names"])
    knob_meta = json.loads((case / "knob.json").read_text())
    vec = dict(np.load(case / "knob_vectors.npz", allow_pickle=False))
    ayadial = dict(np.load(case / "ayadial_vectors.npz", allow_pickle=False)) if (case / "ayadial_vectors.npz").exists() else {}
    cache = {}

    def _orig(name):
        if name not in names:
            raise KeyError(f"{name!r} is not loadable here. NAMES lists the {len(names)} tensors you may read (layers 18 to 22).")
        if name not in cache:
            with safe_open(weights_file, framework="pt") as f:
                cache[name] = f.get_tensor(name).float().numpy()
            if len(cache) > 16:
                cache.pop(next(iter(cache)))
        return cache[name]

    def _read_state(folder):
        st = {}
        for short, name in E.SHORT.items():
            p = folder / f"{short}.npz"
            st[name] = E.load_npz(p) if p.exists() else None
        return st

    start_state = _read_state(case / "start_edit")
    pend_dir = work / "edits" / "pending"
    acc_dir = work / "edits" / "accepted"
    pend_dir.mkdir(parents=True, exist_ok=True)
    counter = [len(list(pend_dir.glob("*")))]

    def _state():
        st = _read_state(acc_dir) if (acc_dir / "state.json").exists() else dict(start_state)
        for p in sorted(pend_dir.glob("*")):
            if p.suffix == ".npz":
                c = E.load_npz(p)
                st[c["name"]] = c
            elif p.suffix == ".json":
                d = json.loads(p.read_text())
                st[d["name"]] = start_state.get(d["name"])
        return st

    def _t(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

    def load(name, scale=1.0):
        """The tensor as the editor's current edit makes it, at knob scale `scale` (float32 torch tensor)."""
        if name in E.EDITABLE:
            return _t(E.apply_full(name, _orig(name), _state().get(name), scale))
        return _t(_orig(name))

    def load_original(name):
        """The unedited Tiny Aya Global tensor."""
        return _t(_orig(name))

    def load_current(name, scale=1.0):
        """The tensor with the starting knob (KNOB['name']) installed at scale `scale`, ignoring your edits."""
        if name in E.EDITABLE:
            return _t(E.apply_full(name, _orig(name), start_state.get(name), scale))
        return _t(_orig(name))

    def load_knob(name):
        """The part of your edit that is multiplied by s (a full tensor, zero where there is none)."""
        if name not in E.EDITABLE:
            raise KeyError(f"only {sorted(E.EDITABLE)} carry an edit")
        W = np.zeros_like(_orig(name))
        c = _state().get(name)
        if c is not None and len(c["units"]):
            if E.unit_axis(name) == 1:
                W[:, c["units"]] = c["knob"].T
            else:
                W[c["units"]] = c["knob"]
        return _t(W)

    def _np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=np.float32)

    def save_edit(name, tensor, knob=None):
        """Set the full new value of one layer-22 MLP tensor at s = 0 (`tensor`) and, optionally, its knob part (`knob`,
        same shape, multiplied by s; None keeps the current knob part of this tensor; 0 removes it).
        Checked here at once and again by the harness after the run; the harness's check is the one that counts."""
        if name not in E.EDITABLE:
            raise E.EditRefused(f"save_edit refused: {name!r} is not editable; only {sorted(E.EDITABLE)} are")
        cur = _state().get(name)
        if knob is None:
            kfull = np.zeros_like(_orig(name))
            if cur is not None and len(cur["units"]):
                if E.unit_axis(name) == 1:
                    kfull[:, cur["units"]] = cur["knob"].T
                else:
                    kfull[cur["units"]] = cur["knob"]
        elif np.isscalar(knob) and float(knob) == 0.0:
            kfull = None
        else:
            kfull = _np(knob)
        c = E.compact_from_full(name, _np(tensor), kfull, _orig(name))
        c = E.canonical(c, E.take_units(name, _orig(name), c["units"]))
        st = _state()
        others = [u for n, o in st.items() if n != name and o is not None for u in o["units"].tolist()]
        errs = E.validate(c, rules, others)
        if errs:
            raise E.EditRefused("save_edit refused: " + "; ".join(errs))
        counter[0] += 1
        E.save_npz(pend_dir / f"{counter[0]:05d}_{E.EDITABLE[name]}.npz", c)
        per, union = E.state_units({**st, name: c})
        print(f"save_edit({E.EDITABLE[name]}): units {c['units'].tolist()} saved; distinct units now {union}. "
              f"Checked here; the harness confirms after this run.")
        return None

    def drop_edit(name):
        """Return one tensor to the starting knob (KNOB['name']); to remove the knob too, save the original rows."""
        if name not in E.EDITABLE:
            raise E.EditRefused(f"drop_edit: {name!r} is not editable")
        counter[0] += 1
        (pend_dir / f"{counter[0]:05d}_drop_{E.EDITABLE[name]}.json").write_text(json.dumps({"name": name}))
        print(f"drop_edit({E.EDITABLE[name]}): back to the starting knob; the harness confirms after this run.")

    def edited():
        """{tensor: units that differ from the original model} for your current edit, plus its hash."""
        st = _state()
        per, union = E.state_units(st)
        return {"units_per_tensor": per, "distinct_units": union, "edit_hash": E.state_hash(st),
                "knob_units": {E.EDITABLE[n]: [int(u) for u, k in zip(c["units"], c["knob"]) if (k != 0).any()]
                               for n, c in st.items() if c is not None}}

    def mlp22(x, weights="edited", scale=1.0, units=None):
        """Layer 22's MLP on LayerNorm'd inputs x [n, 2048] (as returned by read passes): returns (act [n, k], out [n, 2048]).
        weights: 'original' | 'current' (starting knob) | 'edited' (yours). units: list of unit ids, or None for all 11008."""
        x = _t(_np(x))
        f = {"original": lambda n: load_original(n), "current": lambda n: load_current(n, scale), "edited": lambda n: load(n, scale)}[weights]
        G, U, D = (f(f"model.layers.22.mlp.{t}.weight") for t in ("gate_proj", "up_proj", "down_proj"))
        if units is not None:
            idx = torch.as_tensor(list(units), dtype=torch.long)
            G, U, D = G[idx], U[idx], D[:, idx]
        act = torch.nn.functional.silu(x @ G.T) * (x @ U.T)
        return act, act @ D.T

    KNOB = dict(knob_meta)
    KNOB.update({k: v.astype(np.float32) for k, v in vec.items()})
    AYADIAL = {k: v.astype(np.float32) for k, v in ayadial.items()}
    ns = {"__name__": "__main__", "np": np, "torch": torch, "json": json, "Path": Path, "math": __import__("math"),
          "WORK": work, "CASE": case, "NAMES": names, "EDITABLE": sorted(E.EDITABLE), "ALLOWED_UNITS": sorted(rules["allowed_units"]),
          "KNOB": KNOB, "AYADIAL": AYADIAL, "load": load, "load_original": load_original, "load_current": load_current,
          "load_knob": load_knob, "save_edit": save_edit, "drop_edit": drop_edit, "edited": edited, "mlp22": mlp22}
    return ns


def serve(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--max-chars", type=int, default=10000)
    a = ap.parse_args(argv)
    work, case = Path(a.work), Path(a.case)
    os.chdir(work)
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    log = os.open(str(work / ".kernel_stdio.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(log, 1)
    os.dup2(log, 2)
    sys.stdout = io.TextIOWrapper(os.fdopen(1, "wb", buffering=0), write_through=True)
    sys.stderr = sys.stdout

    def say(obj):
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    try:
        ns = _prelude(work, case)
    except BaseException:
        say({"op": "ready", "ok": False, "error": traceback.format_exc()[-4000:]})
        return 1
    say({"op": "ready", "ok": True, "pid": os.getpid()})
    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            continue
        if not line:
            return 0
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            say({"op": "error", "error": "bad command"})
            continue
        if cmd.get("op") == "exec":
            res = _run_code(cmd["code"], ns, a.max_chars)
            res.update({"op": "result", "id": cmd.get("id")})
            say(res)
        elif cmd.get("op") == "ping":
            say({"op": "pong", "id": cmd.get("id")})
        elif cmd.get("op") == "quit":
            return 0


# ==
# Driver side (outside the sandbox)
# ==

SYSCTL_ALLOW = ["hw.", "machdep.cpu."]  # prefixes
SYSCTL_ALLOW_NAMES = ["kern.ostype", "kern.osrelease", "kern.version", "kern.hostname", "kern.osversion", "kern.osproductversion",
                      "kern.argmax", "kern.maxfilesperproc", "kern.usrstack64", "sysctl.proc_cputype", "sysctl.proc_native",
                      "kern.boottime", "kern.secure_kernel", "kern.osvariant_status", "vm.footprint_suspend", "vm.pagesize",
                      "kern.ngroups", "kern.hv_vmm_present"]


def _q(p):
    return '"' + str(p).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ancestors(paths):
    out = set()
    for p in paths:
        p = Path(p)
        for a in [p, *p.parents]:
            out.add(str(a))
    return sorted(out)


def symlink_chain(path):
    """Every path on the way from `path` to the real file (each link must be readable for exec to work)."""
    chain, p = [], Path(path)
    for _ in range(40):
        chain.append(str(p))
        if not p.is_symlink():
            break
        t = Path(os.readlink(p))
        p = t if t.is_absolute() else (p.parent / t)
        p = Path(os.path.normpath(p))
    return chain


def build_profile(python_real, runtime_dirs, read_files, read_dirs, rw_dir, python_links=()):
    """SBPL text. Later rules override earlier ones, so the denies that tighten system.sb come after its import."""
    read_files = list(read_files) + [l for l in python_links if l != python_real]
    allowed = [*runtime_dirs, *read_files, *read_dirs, rw_dir, python_real]
    L = ["(version 1)",
         "(deny default)",
         '(import "system.sb")',
         f"(allow process-exec (literal {_q(python_real)}))",
         "(allow file-read* " + " ".join([f"(subpath {_q(d)})" for d in runtime_dirs + read_dirs] +
                                         [f"(literal {_q(f)})" for f in read_files + [python_real]]) + ")",
         f"(allow file-read* file-write* (subpath {_q(rw_dir)}))",
         # path resolution needs stat() on every ancestor of an allowed path; nothing else outside is even stat-able
         "(allow file-read-metadata " + " ".join(f"(literal {_q(a)})" for a in _ancestors(allowed)) + ")",
         "(deny network*)",
         # system.sb allows every sysctl read; kern.procargs2 (another process's argv and environment) is only blocked when
         # sysctl-read and process-info are both denied, so deny both and allow back what Python needs.
         "(deny sysctl-read process-info*)",
         "(allow process-info* (target self))",
         "(allow sysctl-read " + " ".join([f"(sysctl-name-prefix {_q(p)})" for p in SYSCTL_ALLOW] +
                                         [f"(sysctl-name {_q(n)})" for n in SYSCTL_ALLOW_NAMES]) + ")",
         "(deny process-fork)",
         "(deny nvram*)"]
    return "\n".join(L) + "\n"


def scrubbed_env(work):
    """The kernel's whole environment. Nothing is inherited from the driver (so no API key)."""
    return {"PATH": "/usr/bin:/bin", "HOME": str(work), "TMPDIR": str(Path(work) / "tmp") + "/", "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "MPLCONFIGDIR": str(Path(work) / "tmp"),
            "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "8"}


class KernelDied(Exception):
    pass


def phys_footprint(pid):
    """ri_phys_footprint from proc_pid_rusage (RUSAGE_INFO_V0), in bytes, or None."""
    import ctypes

    class RU(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
            "phys_footprint", "proc_start_abstime", "proc_exit_abstime")]
    try:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        ru = RU()
        return ru.phys_footprint if lib.proc_pid_rusage(int(pid), 0, ctypes.byref(ru)) == 0 else None
    except Exception:
        return None


def rss_of(pid):
    try:
        import psutil
        return psutil.Process(pid).memory_info().rss
    except Exception:
        return None


class Kernel:
    """Driver-side handle on the sandboxed kernel."""

    def __init__(self, *, python, run_dir, work, case, weights_file, timeout_s, grace_s, memory_gb, max_chars,
                 watchdog_s=0.25, log=print):
        import threading
        self.python = Path(python)
        self.python_real = Path(os.path.realpath(self.python))
        self.run_dir, self.work, self.case = Path(run_dir), Path(work), Path(case)
        self.weights_file = Path(weights_file)
        self.timeout_s, self.grace_s, self.mem_bytes = timeout_s, grace_s, int(memory_gb * 1024 ** 3)
        self.max_chars, self.watchdog_s, self.log = max_chars, watchdog_s, log
        self.proc, self.q, self.kill_reason, self.starts = None, None, None, 0
        self._lock = threading.Lock()

    # -- lifecycle
    def runtime_dirs(self):
        venv = self.python.parent.parent
        base_lib = self.python_real.parent.parent / "lib"
        return [str(venv), str(base_lib)]

    def profile_text(self):
        return build_profile(str(self.python_real), self.runtime_dirs(), [str(self.weights_file)], [str(self.case)], str(self.work),
                             python_links=symlink_chain(self.python))

    def start(self):
        import queue
        import threading
        (self.work / "tmp").mkdir(parents=True, exist_ok=True)
        prof = self.run_dir / "sandbox.sb"
        prof.write_text(self.profile_text())
        cmd = ["/usr/bin/sandbox-exec", "-f", str(prof), str(self.python), "-I", str(self.case / "lib" / "kernel.py"), "serve",
               "--work", str(self.work), "--case", str(self.case), "--max-chars", str(self.max_chars)]
        errlog = open(self.run_dir / "kernel_stderr.log", "ab")
        self.proc = __import__("subprocess").Popen(cmd, stdin=-1, stdout=-1, stderr=errlog, cwd=str(self.work),
                                                    env=scrubbed_env(self.work), start_new_session=True)
        self.q = queue.Queue()
        self.kill_reason = None
        self.starts += 1
        threading.Thread(target=self._reader, args=(self.proc, self.q), daemon=True).start()
        threading.Thread(target=self._watchdog, args=(self.proc,), daemon=True).start()
        msg = self._get(180)
        if not msg or msg.get("op") != "ready" or not msg.get("ok"):
            self.kill("startup failed")
            raise KernelDied(f"kernel startup failed: {msg}")
        return msg

    @staticmethod
    def _reader(proc, q):
        while True:
            line = proc.stdout.readline(4 * 1024 * 1024)
            if not line:
                q.put(None)
                return
            try:
                q.put(json.loads(line))
            except json.JSONDecodeError:
                q.put({"op": "garbage"})

    def _watchdog(self, proc):
        """Kills the kernel when its memory use passes the limit. Uses the physical footprint (what Activity Monitor shows; it
        includes compressed pages, which RSS does not: a 3 GB array showed 1.2 GB of RSS and 3.0 GB of footprint), or RSS if larger."""
        while proc.poll() is None:
            used = max(phys_footprint(proc.pid) or 0, rss_of(proc.pid) or 0)
            if used > self.mem_bytes:
                self.kill_reason = f"memory limit: the kernel used {used / 1024 ** 3:.1f} GB, above the {self.mem_bytes / 1024 ** 3:.1f} GB limit"
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
                return
            time.sleep(self.watchdog_s)

    def _get(self, timeout):
        import queue
        end = time.time() + timeout
        while True:
            left = end - time.time()
            if left <= 0:
                return {"op": "timeout"}
            try:
                msg = self.q.get(timeout=left)
            except queue.Empty:
                return {"op": "timeout"}
            if msg is None:
                return None
            if msg.get("op") in ("result", "ready", "pong", "error"):
                return msg

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def kill(self, reason):
        if self.proc is not None and self.proc.poll() is None:
            self.kill_reason = self.kill_reason or reason
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except Exception:
                self.proc.kill()
            self.proc.wait(10)

    def stop(self):
        if self.alive():
            try:
                self.proc.stdin.write(b'{"op": "quit"}\n')
                self.proc.stdin.flush()
                self.proc.wait(5)
            except Exception:
                pass
        self.kill("stopped")

    # -- one run
    def run(self, code, run_id):
        """Returns {ok, output, truncated, seconds, restarted: None | reason}."""
        restarted = None
        if not self.alive():
            restarted = f"the kernel was not running ({self.kill_reason or 'stopped'})"
            self.start()
        try:
            self.proc.stdin.write((json.dumps({"op": "exec", "id": run_id, "code": code}) + "\n").encode())
            self.proc.stdin.flush()
        except BrokenPipeError:
            reason = self.kill_reason or "the kernel process ended"
            self.kill(reason)
            self.start()
            return {"ok": False, "output": "", "truncated": False, "seconds": 0, "restarted": reason}
        t0 = time.time()
        msg = self._get(self.timeout_s)
        if msg and msg.get("op") == "timeout":
            try:
                os.kill(self.proc.pid, signal.SIGINT)
            except Exception:
                pass
            msg = self._get(self.grace_s)
            if not msg or msg.get("op") == "timeout":
                reason = f"time limit: the run did not stop within {self.grace_s} s of the interrupt after {self.timeout_s} s"
                self.kill(reason)
                self.start()
                return {"ok": False, "output": "", "truncated": False, "seconds": round(time.time() - t0, 1), "restarted": reason}
        if msg is None:  # the process died during the run
            try:
                self.proc.wait(5)
            except Exception:
                pass
            reason = self.kill_reason or f"the kernel process ended with code {self.proc.returncode}"
            self.start()
            return {"ok": False, "output": "", "truncated": False, "seconds": round(time.time() - t0, 1), "restarted": reason}
        out = {k: msg.get(k) for k in ("ok", "output", "truncated", "seconds")}
        out["restarted"] = restarted
        return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        sys.exit(serve(sys.argv[2:]))
    print(__doc__)
