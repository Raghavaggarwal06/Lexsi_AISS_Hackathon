"""Round 5 shared library (runs on the Colab runtime). Import this BEFORE any r4 module: it points r4lib at the round 5 folders.

Cells and suites. A cell (runs_private/round5/gen/<cell>.jsonl, results/round5/labels/<cell>.json) is a union over suites.
A suite is a fixed row list with fixed batching; every cell is compared with the base cell of the same suite (base_<suite>).
Suites 'wjx' and 'own' are last night's development rows with last night's batching, so last night's cells can be reused.
"""
import r4lib
from r4lib import *

OUT, RAW, CACHE = ROOT / "results/round5", ROOT / "runs_private/round5", ROOT / "cache_r5"
DRIVE = Path("/content/drive/MyDrive/lexsi-knob/round5")
R4OUT, R4RAW = ROOT / "results/round4", ROOT / "runs_private/round4"
for p in (OUT, RAW, CACHE, RAW / "gen", OUT / "labels"):
    p.mkdir(parents=True, exist_ok=True)
r4lib.OUT, r4lib.RAW, r4lib.CACHE, r4lib.DRIVE = OUT, RAW, CACHE, DRIVE
import r4_scoreboard as sb

LYR, ALPHA, KAPPA, BETA = 22, 20.0, 60.0, 4.0
STRICT_UNITS, LAX_UNITS = (6666, 1896), (1785, 4466)  # the four most dormant layers[22] units (round 3, block 3.3)
LAX_ALPHA, LAX_BETA = 40.0, 8.0
WJ_H, WJ_B = ["wj_adversarial_harmful", "dev_ho_h", "dev_in_h"], ["wj_adversarial_benign", "dev_ho_b", "dev_in_b"]
OWN = ["harmful", "harmful_jb", "xstest_safe", "orbench_hard", "benign"]
SUITE_ENV = {"wjx": (90000, 160), "own": (30000, 48), "frz5": (70000, 160), "sdev5": (30000, 48)}  # (token budget, max batch); anything else uses DEFAULT_ENV.
# sdev5 (SORRY-Bench development rows) ran out of GPU memory at the default budget: cipher styles have about one token per character, which the
# length heuristic (characters / 3) underestimates. Its base had not been written, so the whole suite runs at the small budget.
# frz5 runs with a smaller token budget: the dev5 base logged one allocator out-of-memory warning at 90000 on the longest prompts (it recovered), and a frozen cell must not die half way.
DEFAULT_ENV = (90000, 160)


def sync():
    subprocess.run(f"mkdir -p {DRIVE}/results {DRIVE}/runs_private {DRIVE}/scripts {DRIVE}/cache; cp -ru {OUT}/. {DRIVE}/results/; cp -ru {RAW}/. {DRIVE}/runs_private/; "
                   f"cp -u {ROOT}/scripts/r5*.py {DRIVE}/scripts/; cp -u {CACHE}/vec_*.npz {DRIVE}/cache/ 2>/dev/null; cp -ru {ROOT}/specs5 {DRIVE}/ 2>/dev/null", shell=True)


r4lib.sync = sync


from r5_degcheck import is_degenerate5  # script-aware degeneracy rule; identical to the old rule on Latin-script text (degenerate_rule_check.json)
score.is_degenerate = is_degenerate5

RAWMAP = {}  # request text -> fully rendered chat text, for suites whose rows carry their own format (robustness checks)
_render0 = r4lib.render
r4lib.render = lambda tok, p: RAWMAP.get(p) or _render0(tok, p)
render = r4lib.render


def rvec():
    return np.load(ROOT / "results/day1/directions.npz")["r_raw"][23].astype(np.float32)


def suite_rows(suite):
    allr = sets()
    if suite == "wjx":
        rows = [r for r in allr if r["set"] in WJ_H + WJ_B]
    elif suite == "own":
        rows = [r for r in allr if r["set"] in OWN]
    elif suite == "dev5":  # SORRY-Bench joined after this suite's base was generated, so its development rows are a suite of their own
        rows = [r for r in allr if (r["set"].endswith("_dev") or r["set"].endswith("_pool")) and not r["set"].startswith("sorry_")]
    elif suite == "sdev5":
        rows = [r for r in allr if r["set"].startswith("sorry_") and r["set"].endswith("_dev")]
    elif suite == "frz5":
        rows = [r for r in allr if r["set"].endswith("_frz")]
    else:  # a custom suite: runs_private/round5/suites/<suite>.jsonl
        rows = read_jsonl(RAW / "suites" / f"{suite}.jsonl")
        RAWMAP.update({r["prompt"]: r["raw"] for r in rows if r.get("raw")})  # one such suite per process
    if os.environ.get("R5_SMOKE"):
        seen, keep = {}, []
        for r in rows:
            if seen.setdefault(r["set"], 0) < int(os.environ["R5_SMOKE"]):
                seen[r["set"]] += 1; keep.append(r)
        rows = keep
    return rows


def cname(name):
    """Cell file name: smoke runs (R5_SMOKE) write to smoke_<name> so they never touch a real cell."""
    return ("smoke_" if os.environ.get("R5_SMOKE") else "") + name


def run(model, tok, suite, name, ctx=None, rows=None, max_new=128):
    """One cell on one suite, with the suite's batching. Resumable. Returns {set|id: [refused, unsafe, degenerate]} for the suite's rows."""
    rows = suite_rows(suite) if rows is None else rows
    tb, bs = SUITE_ENV.get(suite, DEFAULT_ENV)
    os.environ["R4_TOK"], os.environ["R4_BATCH"] = str(tb), str(bs)
    return run_cell(model, tok, cname(name), rows, ctx, max_new=int(os.environ.get("R5_MAXNEW", max_new)))


def base_name(suite):
    return f"base_{suite}"


# --- computed units
class Units:
    """Computed MLP units in layers[LYR].mlp. `units` is a list of dicts {n, gate, up, down} (float32: two x-space rows and one residual column).
    mode 'weights' writes the three slices in place, which is what the adapters do. mode 'hook' zeroes the down columns and adds the same
    function, computed in float32 from the MLP input, to the block output (the parity reference).
    fire = net push at the last prefill position, in units of r (positive toward refusal)."""
    def __init__(self, model, units, mode="weights", layer=LYR, r=None):
        self.mlp, self.block, self.mode = model.model.layers[layer].mlp, model.model.layers[layer], mode
        W = self.mlp.gate_proj.weight
        t32 = lambda v: torch.tensor(np.asarray(v, np.float32), device=W.device)
        r = rvec() if r is None else np.asarray(r, np.float32)
        self.units = [{"n": int(u["n"]), "gate": t32(u["gate"]), "up": t32(u["up"]), "down": t32(u["down"])} for u in units]
        assert len({u["n"] for u in self.units}) == len(self.units), "a unit is used twice"
        self.coef = [float(np.asarray(u["down"], np.float32) @ r / (r @ r)) for u in units]
        self.fire, self.handles, self.saved = [], [], []

    def _pre(self, _m, a):
        self.x = a[0]

    def _rec(self, _m, a):
        if a[0].shape[1] > 1:
            self.fire.append(sum(c * a[0][:, -1, u["n"]].float() for c, u in zip(self.coef, self.units)).cpu())

    def _add(self, _m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        x = self.x.float()
        add, fire = 0, 0
        for c, u in zip(self.coef, self.units):
            act = torch.nn.functional.silu(x @ u["gate"]) * (x @ u["up"])
            add = add + act.unsqueeze(-1) * u["down"]
            fire = fire + c * act[:, -1]
        if h.shape[1] > 1:
            self.fire.append(fire.cpu())
        pushed = h + add.to(h.dtype)
        return (pushed, *out[1:]) if isinstance(out, tuple) else pushed

    def __enter__(self):
        G, U, D = self.mlp.gate_proj.weight, self.mlp.up_proj.weight, self.mlp.down_proj.weight
        self.saved = [(G.data[u["n"]].clone(), U.data[u["n"]].clone(), D.data[:, u["n"]].clone()) for u in self.units]
        for u in self.units:
            n = u["n"]
            if self.mode == "weights":
                G.data[n], U.data[n], D.data[:, n] = u["gate"].to(G.dtype), u["up"].to(G.dtype), u["down"].to(G.dtype)
            else:
                D.data[:, n] = 0
        if self.mode == "weights":
            self.handles = [self.mlp.down_proj.register_forward_pre_hook(self._rec)]
        else:
            self.handles = [self.mlp.register_forward_pre_hook(self._pre), self.block.register_forward_hook(self._add)]
        return self

    def __exit__(self, *a):
        G, U, D = self.mlp.gate_proj.weight, self.mlp.up_proj.weight, self.mlp.down_proj.weight
        for u, (g, up, d) in zip(self.units, self.saved):
            G.data[u["n"]], U.data[u["n"]], D.data[:, u["n"]] = g, up, d
        for h in self.handles:
            h.remove()


def vec(name):
    """Load a vector file from cache_r5 (vec_v1.npz, vec_v2.npz, ...) as a dict of float32 arrays and floats."""
    P = np.load(CACHE / (name if name.endswith(".npz") else f"vec_{name}.npz"))
    return {k: (P[k].astype(np.float32) if P[k].ndim else float(P[k])) for k in P.files}


def strict_units(V, s, beta=BETA, gain=1.0, units=STRICT_UNITS, r=None):
    """The capped fold: push = gain * clamp(beta * (s*z - theta0), 0, 1) * r at the marker, z = w.x. The knob s sits behind the nonlinearity."""
    r = rvec() if r is None else r
    m, q, w, th = V["m"], V["q"], V["w"], V["theta0"]
    gA = -ALPHA * th * m - KAPPA * q + s * ALPHA * w
    gB = -ALPHA * (th + 1.0 / beta) * m - KAPPA * q + s * ALPHA * w
    d = gain * beta * r / ALPHA
    return [dict(n=units[0], gate=gA, up=m, down=d), dict(n=units[1], gate=gB, up=m, down=-d)]


def v1_units(V, s):
    """v1 exactly as shipped: the install adapter plus s times the knob adapter (rows rebuilt from the round 4 safetensors)."""
    return [dict(n=n, gate=V[f"gate0_{n}"] + s * V[f"knob_{n}"], up=V[f"up_{n}"], down=V[f"down_{n}"]) for n in STRICT_UNITS]


def lax_units(V, t, theta, beta=LAX_BETA, alpha=LAX_ALPHA, gain=1.0, units=LAX_UNITS, r=None):
    """The lax side: push = -t * gain * clamp(beta * (theta - z), 0, 1) * r at the marker. It fires only where the read is BELOW theta.
    The scale t multiplies the down columns, so it moves the size of the push and never the threshold."""
    r = rvec() if r is None else r
    m, q, w = V["m"], V["q"], V["w"]
    gC = alpha * (theta * m - w) - KAPPA * q
    gD = alpha * ((theta - 1.0 / beta) * m - w) - KAPPA * q
    d = t * gain * beta * r / alpha
    return [dict(n=units[0], gate=gC, up=m, down=-d), dict(n=units[1], gate=gD, up=m, down=d)]


def resolve(c):
    """A lax threshold may be given as '<vec>:<key>' (e.g. 'v1:h01'): it is read from results/round5/lax/exposure_<vec>.json, so that a cell can be queued
    before the number exists. Returns a copy of the spec with the number filled in."""
    c = dict(c)
    if isinstance(c.get("theta"), str):
        vn, key = c["theta"].split(":")
        c["theta_rule"] = c["theta"]
        c["theta"] = float(json.loads((OUT / "lax" / f"exposure_{vn}.json").read_text())["thresholds"][key])
    return c


def context(model, c):
    """Build the context manager for one cell spec (a dict). f: base | ungated | v1 | strict | lax | both | steer."""
    c = resolve(c)
    f = c["f"]
    if f == "base":
        return None
    if f == "ungated":
        return Steer(model, rvec(), c["scale"], f="ungated", pos=c.get("pos", "all"))
    mode = c.get("mode", "weights")
    if f == "v1":
        return Units(model, v1_units(vec("v1"), c["s"]), mode)
    if f == "strict":
        return Units(model, strict_units(vec(c["vec"]), c["s"], beta=c.get("beta", BETA), gain=c.get("gain", 1.0)), mode)
    lax_kw = {k: c[k] for k in ("beta", "alpha", "gain") if k in c}
    if f == "lax":
        return Units(model, lax_units(vec(c.get("vec", "v1")), c["t"], c["theta"], **lax_kw), mode)
    if f == "both":  # strict and lax installed together. svec 'v1x' = the shipped v1 rows; otherwise a vector file. lvec = the read of the lax side.
        sv = c.get("svec", "v1x")
        su = v1_units(vec("v1"), c["s"]) if sv == "v1x" else strict_units(vec(sv), c["s"])
        return Units(model, su + lax_units(vec(c.get("lvec", "v1")), c["t"], c["theta"], **lax_kw), mode)
    raise ValueError(f)


# --- forward-only capture at the marker
class _Stop(Exception):
    pass


@torch.no_grad()
def capture(model, tok, rows, tokens=120000, max_batch=256):
    """x = LayerNorm'd input of layers[LYR].mlp and h = hidden state entering layers[LYR], both at the last template token. float16 [n, 2048] each."""
    store = {}
    mlp, layers = model.model.layers[LYR].mlp, model.model.layers
    def stop(_m, _a):
        raise _Stop()
    hk = [mlp.register_forward_pre_hook(lambda _m, a: store.__setitem__("x", a[0][:, -1].float().cpu().numpy())),
          layers[LYR - 1].register_forward_hook(lambda _m, _i, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu().numpy())),
          layers[LYR + 1].register_forward_pre_hook(stop)]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]))
    X, H = np.zeros((len(rows), 2048), np.float16), np.zeros((len(rows), 2048), np.float16)
    s = 0
    try:
        while s < len(order):
            bs = int(max(8, min(max_batch, tokens // (120 + len(rows[order[min(s + max_batch, len(order)) - 1]]["prompt"]) // 2))))
            idx = order[s : s + bs]
            s += bs
            try:
                model.model(**enc(tok, [rows[i]["prompt"] for i in idx]), use_cache=False)
            except _Stop:
                pass
            X[idx], H[idx] = store["x"], store["h"]
    finally:
        for h in hk:
            h.remove()
    return X, H


def acts(model, tok, name, rows=None):
    """Cached capture for one set (or a custom row list saved under `name`)."""
    smoke = os.environ.get("R5_SMOKE")
    path = CACHE / f"{'smoke_' if smoke else ''}act_{name}.npz"
    if path.exists():
        P = np.load(path)
        return P["x"], P["h"], [str(i) for i in P["ids"]]
    rows = [r for r in sets() if r["set"] == name] if rows is None else rows
    if smoke:
        rows = rows[: 12 * int(smoke)]
    X, H = capture(model, tok, rows)
    np.savez(path, x=X, h=H, ids=np.array([r["id"] for r in rows]))
    return X, H, [r["id"] for r in rows]


def zread(V, X):
    """The read of a vector file on captured x: z = w.x at the marker (about 0 on harmless fit prompts, about 1 on harmful ones)."""
    return np.asarray(X, np.float32) @ V["w"]


def boot_ci(fn, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]
