"""The model's Python sandbox: one fresh process per run, no network, only /case (read-only) and /work (read-write).

The paper used bubblewrap (lm_understand-logs/harness/sandbox.py). bwrap fails on Scholar nodes (pivot_root: Invalid
argument), so we build the same jail from unprivileged namespaces: `unshare -r -m -n -p -f` (user, mount, network and
pid namespaces), a tmpfs root holding read-only binds of /usr and the sandbox's Python, a fresh /proc, /dev/{null,zero,
random,urandom}, /case (ro) and /work (rw), then chroot into it with a cleared environment. /home, /scratch and /depot
are not visible; there is no network interface but loopback (down).
"""
import json, os, resource, shlex, subprocess, tempfile, time
from pathlib import Path

ENVS = Path("/scratch/scholar/pande1/wsp/envs")
VENVS = {"numpy": ENVS / "sandbox-np"}       # Exp-7-style: numpy 2.2.6 + stdlib only
PYROOT = ENVS / "python"                      # uv-managed CPython the venvs link to

_JAIL = r'''
set -e
R="$1"; VENV="$2"; PYROOT="$3"; CASE="$4"; WORK="$5"; shift 5
mount -t tmpfs -o mode=755 tmpfs "$R"
mkdir -p "$R/usr" "$R/proc" "$R/dev" "$R/tmp" "$R/work" "$R/case" "$R/etc" "$R$VENV" "$R$PYROOT"
mount --rbind /usr "$R/usr"
ln -s usr/lib "$R/lib"; ln -s usr/lib64 "$R/lib64"; ln -s usr/bin "$R/bin"; ln -s usr/sbin "$R/sbin"
mount -t proc proc "$R/proc"
for d in null zero random urandom; do touch "$R/dev/$d"; mount --bind "/dev/$d" "$R/dev/$d"; done
mount -t tmpfs -o mode=1777 tmpfs "$R/tmp"
ro() { mount --bind "$1" "$2"; mount -o remount,bind,ro,nosuid,nodev "$2" 2>/dev/null || mount -o remount,bind,ro "$2"; }
ro "$VENV" "$R$VENV"; ro "$PYROOT" "$R$PYROOT"
if [ -n "$CASE" ]; then ro "$CASE" "$R/case"; fi
mount --bind "$WORK" "$R/work"
exec chroot "$R" /usr/bin/env -i HOME=/tmp PATH=/usr/bin OMP_NUM_THREADS="$OMP" OPENBLAS_NUM_THREADS="$OMP" \
    MKL_NUM_THREADS="$OMP" CUDA_VISIBLE_DEVICES= /bin/sh -c 'cd /work && exec "$@"' sh "$@"
'''


class Sandbox:
    def __init__(self, work_dir, case_dir=None, venv="numpy", prelude="", timeout=600, mem=48 << 30,
                 max_out=15_000, threads=8, log=None):
        self.work = Path(work_dir).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        self.case = Path(case_dir).resolve() if case_dir else None
        self.venv = VENVS.get(venv, Path(venv))
        self.prelude, self.timeout, self.mem, self.max_out, self.threads = prelude, timeout, mem, max_out, threads
        self.log = Path(log) if log else None

    def run(self, code, meta=None):
        """Run prelude + code; return the printed output as the model will see it (cut at max_out)."""
        src = self.prelude + "\n" + code
        root = tempfile.mkdtemp(prefix="wsp-jail-")
        script_path = self.work / ".run.py"
        script_path.write_text(src)
        cmd = ["unshare", "-r", "-m", "-n", "-p", "-f", "--kill-child", "--propagation", "private",
               "/bin/sh", "-c", _JAIL, "jail", root, str(self.venv), str(PYROOT),
               str(self.case or ""), str(self.work),
               f"{self.venv}/bin/python", "-I", "/work/.run.py"]
        env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "OMP": str(self.threads)}
        t0 = time.time()
        rc = None
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, env=env,
                               preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (self.mem, self.mem)),
                               start_new_session=True)
            rc = p.returncode
            out = p.stdout + (("\n[stderr]\n" + p.stderr[-6000:]) if p.stderr.strip() else "")
        except subprocess.TimeoutExpired:
            out = f"[timed out after {self.timeout} s]"
        finally:
            script_path.unlink(missing_ok=True)
            try:
                os.rmdir(root)
            except OSError:
                pass
        full = out
        if len(out) > self.max_out:
            out = out[:self.max_out] + f"\n[output cut at {self.max_out:,} characters]"
        if self.log:
            with open(self.log, "a") as f:
                f.write(json.dumps({**(meta or {}), "t": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                    "s": round(time.time() - t0, 2), "rc": rc, "code": code,
                                    "out": full[:60_000]}) + "\n")
        return out or "(no output)"
