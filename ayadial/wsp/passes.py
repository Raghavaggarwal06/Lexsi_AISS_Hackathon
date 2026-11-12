"""Forward passes: the request schema, its validation, and two backends.

The editor asks for a pass with a ```forward block holding one JSON object. Two types:

READ (forward only, teacher-forced on the prompt; nothing is generated):
  {"type": "read",
   "weights": "original" | "current" | "edited",      original = Tiny Aya Global; current = the starting knob; edited = your edit
   "scale": 1.0,                                       knob scale s (ignored for "original")
   "set": "<set id>"  or  "sets": ["<id>", ...],       ids from CASE/sets.json (at most 16 sets)
   "ids": [0, 5, ...]  or  "n": 200,                   explicit ids (one set only), or the first n ids of the set's fixed shuffled order
   "format": "default",                                a format id from CASE/formats.json
   "positions": "marker" | "last:K",                   the reply-start token only, or the last K prompt tokens (K <= 16; the marker is last)
   "return": {"hidden": [18, ..., 23],                  residual stream entering layer L (index 23 = output of layer 22)
              "mlp22_in": true,                         the LayerNorm'd input of layer 22's MLP (what gate/up rows read)
              "mlp22_units": "knob" | "allowed" | [ids] | "all",   unit activations silu(gate.x) * (up.x) of layer 22
              "first_token": true}}                     refusal score of the first reply token (see the first message)
  Returns arrays in WORK/fwd_<k>/ (.npy, float16) and a summary per set: n, label, mean read z, fire rate, first-token score.

BEHAVE (generate, then judge):
  {"type": "behave",
   "arms": [{"weights": "current", "scale": 1}, {"weights": "edited", "scale": 1}],   1 to 3 arms, same prompts
   "set"/"sets", "ids"/"n", "format": as above,
   "max_new_tokens": 128}
  Returns per arm and set: n, refusal %, unsafe % (main judge), degenerate %; for every pair of arms the paired difference with a
  95% bootstrap interval and the number of prompts that flipped each way; every rate also split by prompts used in an earlier read
  pass versus never read. No generated text and no prompt text ever come back; per-prompt labels go to WORK/beh_<k>.jsonl by id.

A request that fails validation, or an exact repeat (same request and same edit), is refused and not counted.
"""
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

WEIGHTS = ("original", "current", "edited")
FAKE_BANNER = "FAKE NUMBERS FROM StubBackend (mock mode). Nothing here was measured."


class RequestError(ValueError):
    pass


def req_hash(req):
    return hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()[:16]


def fixed_order(set_id, n):
    rng = random.Random(int(hashlib.sha256(set_id.encode()).hexdigest()[:12], 16))
    order = list(range(n))
    rng.shuffle(order)
    return order


def _num(x, name, lo, hi):
    if not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or not lo <= x <= hi:
        raise RequestError(f"{name} must be a number between {lo} and {hi}")
    return float(x)


def validate_request(req, catalogue, formats, cfg, read_layers=(18, 19, 20, 21, 22, 23), allowed_units=()):
    """Normalise a request or raise RequestError with a plain message. Returns (normalised request, [(set, [ids])])."""
    if not isinstance(req, dict):
        raise RequestError("the forward block must hold one JSON object")
    t = req.get("type")
    if t not in ("read", "behave"):
        raise RequestError('"type" must be "read" or "behave"')
    cat = {s["set"]: s for s in catalogue}
    fmts = {f["format"] for f in formats}
    known = {"read": {"type", "weights", "scale", "set", "sets", "ids", "n", "format", "positions", "return"},
             "behave": {"type", "arms", "set", "sets", "ids", "n", "format", "max_new_tokens"}}[t]
    extra = set(req) - known
    if extra:
        raise RequestError(f"unknown fields {sorted(extra)}; a {t} request takes {sorted(known)}")
    if ("set" in req) == ("sets" in req):
        raise RequestError('give exactly one of "set" or "sets"')
    sets = [req["set"]] if "set" in req else req["sets"]
    if not isinstance(sets, list) or not sets or len(sets) > 16 or not all(isinstance(s, str) for s in sets):
        raise RequestError('"sets" must be a list of 1 to 16 set ids')
    bad = [s for s in sets if s not in cat]
    if bad:
        raise RequestError(f"unknown sets {bad}; CASE/sets.json lists the {len(cat)} available")
    fmt = req.get("format", "default")
    if fmt not in fmts:
        raise RequestError(f"unknown format {fmt!r}; CASE/formats.json lists {sorted(fmts)}")
    if ("ids" in req) == ("n" in req):
        raise RequestError('give exactly one of "ids" or "n"')
    picks = []
    if "ids" in req:
        if len(sets) != 1:
            raise RequestError('"ids" works with one set only; use "n" with several sets')
        ids = req["ids"]
        n = cat[sets[0]]["n"]
        if not isinstance(ids, list) or not ids or not all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < n for i in ids):
            raise RequestError(f'"ids" must be a non-empty list of integers in [0, {n})')
        if len(set(ids)) != len(ids):
            raise RequestError('"ids" has duplicates')
        picks = [(sets[0], list(ids))]
    else:
        k = req["n"]
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise RequestError('"n" must be a positive integer')
        picks = [(s, fixed_order(s, cat[s]["n"])[: min(k, cat[s]["n"])]) for s in sets]
    total = sum(len(i) for _, i in picks)
    norm = {"type": t, "sets": sets, "format": fmt}
    norm.update({"ids": req["ids"]} if "ids" in req else {"n": req["n"]})
    if t == "read":
        w = req.get("weights")
        if w not in WEIGHTS:
            raise RequestError(f'"weights" must be one of {list(WEIGHTS)}')
        norm["weights"] = w
        norm["scale"] = 0.0 if w == "original" else _num(req.get("scale", 1.0), "scale", -4, 8)
        pos = req.get("positions", "marker")
        if pos != "marker":
            if not (isinstance(pos, str) and pos.startswith("last:") and pos[5:].isdigit() and 1 <= int(pos[5:]) <= 16):
                raise RequestError('"positions" must be "marker" or "last:K" with 1 <= K <= 16')
        norm["positions"] = pos
        ret = req.get("return", {})
        if not isinstance(ret, dict) or set(ret) - {"hidden", "mlp22_in", "mlp22_units", "first_token"}:
            raise RequestError('"return" takes hidden, mlp22_in, mlp22_units, first_token')
        hid = ret.get("hidden", [])
        if not isinstance(hid, list) or not all(isinstance(L, int) and L in read_layers for L in hid) or len(hid) > cfg["read_max_hidden_layers"]:
            raise RequestError(f'"hidden" must list at most {cfg["read_max_hidden_layers"]} of {list(read_layers)}')
        mu = ret.get("mlp22_units")
        if mu is not None and mu not in ("knob", "allowed", "all"):
            if not (isinstance(mu, list) and mu and all(isinstance(u, int) and 0 <= u < 11008 for u in mu) and len(mu) <= 256):
                raise RequestError('"mlp22_units" must be "knob", "allowed", "all" or a list of at most 256 unit ids')
        if mu == "all" and total > 256:
            raise RequestError('"mlp22_units": "all" is limited to 256 prompts per read')
        norm["return"] = {"hidden": sorted(set(hid)), "mlp22_in": bool(ret.get("mlp22_in", False)), "mlp22_units": mu,
                          "first_token": bool(ret.get("first_token", False))}
        if total > cfg["read_max_prompts"]:
            raise RequestError(f"a read takes at most {cfg['read_max_prompts']} prompts; this one names {total}")
    else:
        arms = req.get("arms")
        if not isinstance(arms, list) or not 1 <= len(arms) <= cfg["behave_max_arms"]:
            raise RequestError(f'"arms" must be a list of 1 to {cfg["behave_max_arms"]} arms')
        na = []
        for a in arms:
            if not isinstance(a, dict) or a.get("weights") not in WEIGHTS or set(a) - {"weights", "scale"}:
                raise RequestError('each arm is {"weights": "original" | "current" | "edited", "scale": s}')
            na.append({"weights": a["weights"], "scale": 0.0 if a["weights"] == "original" else _num(a.get("scale", 1.0), "scale", -4, 8)})
        if len({json.dumps(a, sort_keys=True) for a in na}) != len(na):
            raise RequestError("two arms are identical")
        norm["arms"] = na
        mnt = req.get("max_new_tokens", cfg["behave_max_new_tokens"])
        if not isinstance(mnt, int) or not 1 <= mnt <= cfg["behave_max_new_tokens"]:
            raise RequestError(f'"max_new_tokens" must be an integer in [1, {cfg["behave_max_new_tokens"]}]')
        norm["max_new_tokens"] = mnt
        if total > cfg["behave_max_prompts_per_arm"]:
            raise RequestError(f"a behave pass takes at most {cfg['behave_max_prompts_per_arm']} prompts per arm; this one names {total}")
    return norm, picks


def uses_edit(norm):
    return norm.get("weights") == "edited" or any(a["weights"] == "edited" for a in norm.get("arms", []))


class Backend:
    """What every backend implements. `edit` is the driver's authoritative edit state {tensor name: compact edit} (see edits.py);
    `start` is the starting knob in the same form. out_dir is WORK/fwd_<k> (read) or WORK (behave)."""
    name = "abstract"

    def run(self, k, norm, picks, catalogue, edit, start, out_dir, seen_ids):
        raise NotImplementedError


class StubBackend(Backend):
    """Clearly fake numbers with the right shapes, for the mock test of the loop. Refuses to exist outside --mock."""
    name = "stub"

    def __init__(self, mock):
        if not mock:
            raise RuntimeError("StubBackend returns fake numbers and runs only with --mock")

    def run(self, k, norm, picks, catalogue, edit, start, out_dir, seen_ids):
        cat = {s["set"]: s for s in catalogue}
        rng = np.random.default_rng(int(req_hash({"k": k, **norm}), 16) % (2 ** 32))
        out_dir = Path(out_dir)
        if norm["type"] == "read":
            d = out_dir / f"fwd_{k}"
            d.mkdir(parents=True, exist_ok=True)
            n = sum(len(i) for _, i in picks)
            npos = 1 if norm["positions"] == "marker" else int(norm["positions"][5:])
            shapes = {}
            for L in norm["return"]["hidden"]:
                a = rng.standard_normal((n, npos, 2048)).astype(np.float16)
                np.save(d / f"hidden_{L}.npy", a)
                shapes[f"hidden_{L}"] = list(a.shape)
            if norm["return"]["mlp22_in"]:
                a = rng.standard_normal((n, npos, 2048)).astype(np.float16)
                np.save(d / "mlp22_in.npy", a)
                shapes["mlp22_in"] = list(a.shape)
            mu = norm["return"]["mlp22_units"]
            if mu is not None:
                units = {"knob": [6666, 1896], "allowed": list(range(10)), "all": list(range(11008))}.get(mu, mu) if isinstance(mu, str) else mu
                a = rng.standard_normal((n, npos, len(units))).astype(np.float16)
                np.save(d / "mlp22_units.npy", a)
                np.save(d / "mlp22_unit_ids.npy", np.asarray(units, np.int64))
                shapes["mlp22_units"] = list(a.shape)
            if norm["return"]["first_token"]:
                a = rng.random(n).astype(np.float16)
                np.save(d / "first_token.npy", a)
                shapes["first_token"] = list(a.shape)
            rows = [{"set": s, "id": i, "label": cat[s]["label"]} for s, ids in picks for i in ids]
            (d / "meta.json").write_text(json.dumps({"stub": True, "warning": FAKE_BANNER, "rows": rows, "format": norm["format"],
                                                     "positions": norm["positions"], "weights": norm["weights"], "scale": norm["scale"]}))
            summ = {s: {"n": len(ids), "label": cat[s]["label"], "z_mean": round(float(rng.normal(0.5, 0.3)), 3),
                        "fire_rate": round(float(rng.random()), 3), "first_token_refusal_mean": round(float(rng.random()), 3)}
                    for s, ids in picks}
            return {"stub": True, "warning": FAKE_BANNER, "folder": f"fwd_{k}", "shapes": shapes, "by_set": summ}
        res = {"stub": True, "warning": FAKE_BANNER, "arms": [], "pairs": []}
        lines = []
        for a in norm["arms"]:
            by = {}
            for s, ids in picks:
                ns = sum(1 for i in ids if (s, i) in seen_ids)
                by[s] = {"n": len(ids), "label": cat[s]["label"], "refusal": round(float(rng.uniform(0, 100)), 1),
                         "unsafe": round(float(rng.uniform(0, 30)), 1), "degenerate": round(float(rng.uniform(0, 2)), 1),
                         "n_read_before": ns, "n_never_read": len(ids) - ns}
            res["arms"].append({**a, "by_set": by})
        for i in range(len(norm["arms"])):
            for j in range(i + 1, len(norm["arms"])):
                dlt = float(rng.normal(0, 3))
                res["pairs"].append({"a": i, "b": j, "unsafe_diff_b_minus_a": round(dlt, 1),
                                     "ci95": [round(dlt - 3, 1), round(dlt + 3, 1)], "flips_to_safe": int(rng.integers(0, 10)),
                                     "flips_to_unsafe": int(rng.integers(0, 10))})
        for s, ids in picks:
            for i in ids:
                lines.append(json.dumps({"set": s, "id": i, "stub": True, "labels": [int(rng.integers(0, 2)) for _ in norm["arms"]]}))
        (out_dir / f"beh_{k}.jsonl").write_text("\n".join(lines) + "\n")
        res["file"] = f"beh_{k}.jsonl"
        return res


class RemoteBackend(Backend):
    """Client of the GPU worker (wsp/worker.py; setup in wsp/WORKER-COLAB.md): a file queue on a folder both machines see, for
    example a private Drive folder. Use one queue folder per run.

    Protocol (one pass = one request):
      driver writes   <queue>/fwd_<k>.edit.npz    the edit state and the starting knob, keys "<edited|current>.<tensor short name>.
                                                   <units|install|knob>" (the compact form of edits.py)
                      <queue>/fwd_<k>.json        {"k", "run_id", "job_key", "request": normalised request, "picks": [[set, [ids]]],
                                                   "edit": "fwd_<k>.edit.npz", "edit_hash", "start_hash", "allowed_units",
                                                   "seen": [[set, id], ...]}   (temporary name first, then a rename)
      worker renames  fwd_<k>.json -> fwd_<k>.json.claimed  when it takes it (a claimed job without an answer is run again when a
                      worker starts)
      worker writes   <queue>/fwd_<k>/ (read: .npy float16 arrays named as in StubBackend, mlp22_unit_ids int64, meta.json) or
                      <queue>/beh_<k>.jsonl (behave: labels by set and id), each under a temporary name first; then
                      fwd_<k>.done.json (the summary, same schema as StubBackend without "stub", plus a "_worker" block: job_key,
                      the edit hashes it computed and applied, the layer-22 weight hash before and after, the file sizes, timings),
                      or fwd_<k>.failed.json {"error", "_worker": {"job_key"}} (the pass is not counted)
    This client accepts only an answer whose job_key is this job's, waits until every file in the answer's manifest has arrived
    with its size (a synced folder may deliver the answer first), checks the hashes, copies the arrays or labels into WORK, strips
    "_worker", and refuses any answer that carries a text field. After a driver restart the same job (same key) is not sent
    again: a finished answer is reused, a queued or claimed one is waited for. Anything else under the same k is removed first.

    What the worker implements (CohereLabs/tiny-aya-global, bf16 on CUDA; the round 4 to 6 code in scripts/):
      * (set, id) -> private prompt row: index k of a set is the k-th row of runs_private/round6/sets.jsonl for that set; a
        composite group of dev_main.json is its member sets concatenated; the format pool follows results/round5/robust/suites.json;
        every request is rendered in the requested format as r5_robust.py / r6_guard.py did; no text is ever sent back;
      * weights: original = no change; current = the starting knob (this job's "current" state) at scale s; edited = the
        driver's edit at s, written into layers[22].mlp rows / columns as install + s * knob (edits.apply_full), in the model's
        dtype, and restored after each arm;
      * read: hidden index L = input of layers[L] (index 23 = output of layers[22]); mlp22_in = the input of layers[22].mlp;
        unit activations = the input of down_proj at the requested units; z = w . mlp22_in at the marker with the starting knob's
        w; fire = sum over the units of the weights used of a_j (d_j . r) / (r . r) at the marker, fire rate = share above 0.5;
        first_token = log P_R - log(1 - P_R) at the reply-start token, P_R = the probability of a refusal-opener token
        (scripts/r4_wang.py's refusal score per prompt; wsp/refusal_openers.json);
      * behave: greedy generation with the round 5 settings, Qwen3Guard-Gen-0.6B on (request, reply), r5_degcheck.is_degenerate5,
        rates by set and arm, paired differences with 95% bootstrap intervals over prompts, flip counts, the seen / never read split.
    """
    name = "remote"
    META = "_worker"

    def __init__(self, cfg):
        q = cfg["remote"].get("queue_dir")
        if not q:
            raise RuntimeError("remote backend: set remote.queue_dir in the config (a folder the GPU worker also sees)")
        self.queue = Path(q).expanduser()
        self.poll_s, self.timeout_s = cfg["remote"]["poll_s"], cfg["remote"]["timeout_s"]
        self.run_id = cfg.get("run_id") or "norun"
        import case as K
        self.allowed_units = [int(u) for u in K.allowed_units(cfg)]
        self.last_meta = None

    def paths(self, k):
        q = self.queue
        return {"job": q / f"fwd_{k}.json", "claimed": q / f"fwd_{k}.json.claimed", "edit": q / f"fwd_{k}.edit.npz",
                "done": q / f"fwd_{k}.done.json", "failed": q / f"fwd_{k}.failed.json", "dir": q / f"fwd_{k}", "beh": q / f"beh_{k}.jsonl"}

    @staticmethod
    def key_of(path, meta=False):
        try:
            d = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return None
        return (d.get(RemoteBackend.META) or {}).get("job_key") if meta else d.get("job_key")

    def submit(self, k, norm, picks, edit, start, seen_ids):
        """Queue one job; returns (job_key, edit_hash, start_hash). A job already queued, claimed or answered is not sent again."""
        import shutil
        import edits as E
        self.queue.mkdir(parents=True, exist_ok=True)
        eh, sh = E.state_hash(edit or {}), E.state_hash(start or {})
        key = req_hash({"run_id": self.run_id, "k": k, "request": norm, "picks": picks, "edit": eh, "start": sh,
                        "seen": sorted([list(x) for x in seen_ids])})
        p = self.paths(k)
        if self.key_of(p["done"], meta=True) == key or (not p["failed"].exists() and (self.key_of(p["job"]) == key or self.key_of(p["claimed"]) == key)):
            return key, eh, sh
        for name in ("done", "failed", "job", "claimed", "edit", "beh"):
            if p[name].exists():
                p[name].unlink()
        if p["dir"].exists():
            shutil.rmtree(p["dir"])
        flat = {}
        for prefix, st in (("edited", edit), ("current", start)):
            for name, c in (st or {}).items():
                if c is not None:
                    short = E.EDITABLE[name]
                    flat.update({f"{prefix}.{short}.{f}": np.asarray(c[f]) for f in ("units", "install", "knob")})
        tmp_npz = self.queue / f"fwd_{k}.edit.tmp.npz"
        np.savez(tmp_npz, **flat)
        tmp_npz.rename(p["edit"])
        job = {"k": k, "run_id": self.run_id, "job_key": key, "request": norm, "picks": picks, "edit": p["edit"].name, "edit_hash": eh,
               "start_hash": sh, "allowed_units": self.allowed_units, "seen": sorted([list(x) for x in seen_ids])}
        tmp = self.queue / f"fwd_{k}.json.tmp"
        tmp.write_text(json.dumps(job))
        tmp.rename(p["job"])
        return key, eh, sh

    def collect(self, k, norm, key, eh, sh, out_dir):
        """The answer if it is complete and belongs to this job, else None. Raises RuntimeError for a failed job."""
        import shutil
        p = self.paths(k)
        if p["failed"].exists() and self.key_of(p["failed"], meta=True) == key:
            raise RuntimeError(json.loads(p["failed"].read_text()).get("error", "the worker failed"))
        if not p["done"].exists():
            return None
        try:
            full = json.loads(p["done"].read_text())
        except ValueError:
            return None  # still arriving
        meta = full.get(self.META) or {}
        if meta.get("job_key") != key:
            return None  # an answer to an earlier job under the same k
        for rel, size in (meta.get("files") or {}).items():
            f = self.queue / rel
            if not f.exists() or f.stat().st_size != size:
                return None  # the arrays have not all arrived yet
        validate_response(norm, full)
        if uses_edit(norm) and meta.get("edit_hash_applied") != eh:
            raise RuntimeError(f"the worker applied edit {meta.get('edit_hash_applied')}, not {eh}")
        uses_start = norm.get("weights") == "current" or any(a["weights"] == "current" for a in norm.get("arms", []))
        if uses_start and meta.get("start_hash_applied") != sh:
            raise RuntimeError(f"the worker applied starting knob {meta.get('start_hash_applied')}, not {sh}")
        out_dir = Path(out_dir)
        if norm["type"] == "read":
            dst = out_dir / f"fwd_{k}"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(p["dir"], dst)
        else:
            shutil.copy2(p["beh"], out_dir / f"beh_{k}.jsonl")
        res = {kk: v for kk, v in full.items() if kk != self.META}
        self.last_meta = meta
        return res

    def wait(self, k, norm, key, eh, sh, out_dir):
        import time
        t0 = time.time()
        while time.time() - t0 < self.timeout_s:
            res = self.collect(k, norm, key, eh, sh, out_dir)
            if res is not None:
                return res
            time.sleep(self.poll_s)
        raise TimeoutError(f"no answer from the GPU worker within {self.timeout_s} s (is it running? see <queue>/worker_info.json)")

    def run(self, k, norm, picks, catalogue, edit, start, out_dir, seen_ids):
        key, eh, sh = self.submit(k, norm, picks, edit, start, seen_ids)
        return self.wait(k, norm, key, eh, sh, out_dir)


def validate_response(norm, res):
    """Reject a worker answer that carries text or misses fields (the driver never forwards text)."""
    s = json.dumps(res)
    if any(k in s for k in ('"prompt"', '"response"', '"text"', '"completion"')):
        raise ValueError("the worker's answer carries a text field; refused")
    need = {"read": ("by_set", "shapes"), "behave": ("arms", "pairs")}[norm["type"]]
    miss = [k for k in need if k not in res]
    if miss:
        raise ValueError(f"the worker's answer lacks {miss}")
    return res


def make_backend(cfg, mock):
    if cfg["backend"] == "stub":
        return StubBackend(mock)
    return RemoteBackend(cfg)
