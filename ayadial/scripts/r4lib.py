"""Round 4 shared library (runs on the Colab runtime). Private paths: runs_private/round4, cache_r4."""
import hashlib, importlib.util, json, os, random, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(os.environ.get("R4_ROOT", "/content/build"))  # the env overrides exist only for local smoke tests; the defaults are the runtime
sys.path.insert(0, str(ROOT / "scripts"))
from common import read_jsonl, write_jsonl, render, load_set, load_splits

OUT, RAW, CACHE = ROOT / "results/round4", ROOT / "runs_private/round4", ROOT / "cache_r4"
DRIVE = Path("/content/drive/MyDrive/lexsi-knob/round4")
for p in (OUT, RAW, CACHE, RAW / "gen", OUT / "labels"):
    p.mkdir(parents=True, exist_ok=True)
MODEL, JUDGE, DEV, MAXLEN = "CohereLabs/tiny-aya-global", "Qwen/Qwen3Guard-Gen-0.6B", os.environ.get("R4_DEV", "cuda"), 1536
SCORES, STATE = {}, {}


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score = _module("score", "05_score.py")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sync():
    subprocess.run(f"mkdir -p {DRIVE}/results {DRIVE}/runs_private {DRIVE}/scripts && cp -ru {OUT}/. {DRIVE}/results/ && cp -ru {RAW}/. {DRIVE}/runs_private/ && cp -u {ROOT}/scripts/r4*.py {DRIVE}/scripts/", shell=True)


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side, tok.truncation_side = "left", "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, os.environ.get("R4_DTYPE", "bfloat16"))).to(DEV).eval()
    return model, tok


def enc(tok, prompts):
    return tok([render(tok, p) for p in prompts], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN, add_special_tokens=False).to(DEV)


def sets(names=None):
    rows = read_jsonl(RAW / "sets.jsonl")
    return [r for r in rows if names is None or r["set"] in names]


def rk(r):
    return f"{r['set']}|{r['id']}"


# ------------------------------------------------------------------ steering
class Steer:
    """h[write_layer] += scale * f(h[read_layer] . d) * r.  Layer index = hidden_states index (hook on layers[L-1]).
    f: 'ungated' 1; 'prop' clamp((p-b)/(hm-b),0); 'gate' p>tau; 'gate_inv' p<tau; 'relu' clamp(p-tau,0)/unit; 'lin' (p-tau)/unit.
    pos: 'all' = every position incl. generated tokens, each by its own read (foldable);
         'last' = only the last prefill position; 'tail:k' = last k prefill positions, each by its own read; 'tail:k+gen' also decode tokens.
    """
    def __init__(self, model, r, scale, f="ungated", d=None, read_layer=23, write_layer=23, pos="all", tau=0.0, b=0.0, hm=1.0, unit=1.0, cap=None):
        p = next(model.parameters())
        t = lambda v: None if v is None else torch.as_tensor(np.asarray(v), device=p.device, dtype=p.dtype)
        self.model, self.r, self.d, self.scale, self.f = model, t(r), t(d), scale, f
        self.rl, self.wl, self.pos, self.tau, self.b, self.hm, self.unit, self.cap = read_layer, write_layer, pos, tau, b, hm, unit, cap
        self.w, self.handles, self.fire = None, [], []

    def weight(self, h):
        if self.f == "ungated":
            return torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
        p = h @ self.d
        if self.f == "prop":
            w = ((p - self.b) / (self.hm - self.b)).clamp(min=0)
        elif self.f == "gate":
            w = (p > self.tau).to(h.dtype)
        elif self.f == "gate_inv":
            w = (p < self.tau).to(h.dtype)
        elif self.f == "relu":
            w = ((p - self.tau) / self.unit).clamp(min=0)
        elif self.f == "hinge":  # Fable F2: knob behind the nonlinearity. z = (p-b)/unit is ~0 on harmless, ~1 on harmful; push = relu(s*z - tau) * r
            w = (self.scale * (p - self.b) / self.unit - self.tau).clamp(min=0)
        elif self.f == "lin":
            w = (p - self.tau) / self.unit
        else:
            raise ValueError(self.f)
        return w.clamp(max=self.cap) if self.cap is not None else w

    def mask(self, w):
        T = w.shape[1]
        if self.pos == "all":
            return w
        if T == 1:  # decode step
            return w if self.pos.endswith("+gen") else w * 0
        k = 1 if self.pos == "last" else int(self.pos.split(":")[1].split("+")[0])
        m = torch.zeros_like(w)
        m[:, -k:] = 1
        return w * m

    def _read(self, _m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        self.w = self.mask(self.weight(h))
        if h.shape[1] > 1:
            self.fire.append(self.w[:, -1].float().cpu())
        if self.rl == self.wl:
            return self._write(_m, _i, out)

    def _write(self, _m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        pushed = h + (1.0 if self.f == "hinge" else self.scale) * self.w.unsqueeze(-1) * self.r
        return (pushed, *out[1:]) if isinstance(out, tuple) else pushed

    def __enter__(self):
        L = self.model.model.layers
        self.handles = [L[self.rl - 1].register_forward_hook(self._read)]
        if self.rl != self.wl:
            assert self.wl > self.rl
            self.handles.append(L[self.wl - 1].register_forward_hook(self._write))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def generate(model, tok, rows, steer=None, max_new=128, batch=48):
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]))
    out, fire = [None] * len(rows), [None] * len(rows)
    s = 0
    while s < len(order):
        bs = int(max(8, min(int(os.environ.get('R4_BATCH', batch)), int(os.environ.get('R4_TOK', 30000)) // (420 + len(rows[order[min(s + batch, len(order)) - 1]]["prompt"]) // 3))))
        idx = order[s : s + bs]
        s += bs
        e = enc(tok, [rows[i]["prompt"] for i in idx])
        kw = dict(do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        if steer is None:
            ids = model.generate(**e, **kw)
        else:
            steer.fire = []
            if hasattr(steer, "before"):
                steer.before([rows[i] for i in idx])
            with steer:
                ids = model.generate(**e, **kw)
            for i, v in zip(idx, steer.fire[0].tolist() if steer.fire else [None] * len(idx)):
                fire[i] = v
        for i, t in zip(idx, tok.batch_decode(ids[:, e["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[i] = t
    return out, fire


# ------------------------------------------------------------------ judge
def load_judge():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(JUDGE, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype="auto").to(DEV).eval()
    STATE["judge"] = (model, tok)
    if (RAW / "scores.jsonl").exists():
        SCORES.update({r["key"]: r for r in read_jsonl(RAW / "scores.jsonl")})


@torch.no_grad()
def judge(rows, batch=64):
    if "judge" not in STATE:
        load_judge()
    todo = {score.key(r): r for r in rows if score.key(r) not in SCORES and not score.is_degenerate(r["response"])}
    model, tok = STATE["judge"]
    items = sorted(todo.items(), key=lambda kv: len(kv[1]["prompt"]) + len(kv[1]["response"]))
    with open(RAW / "scores.jsonl", "a") as f:
        for i in range(0, len(items), batch):
            chunk = items[i : i + batch]
            text = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}], tokenize=False, add_generation_prompt=False) for _, r in chunk]
            e = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=3072).to(DEV)
            ids = model.generate(**e, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=32)
            for (k, _), t in zip(chunk, tok.batch_decode(ids[:, e["input_ids"].shape[1]:], skip_special_tokens=True)):
                SCORES[k] = {"key": k, **score.parse(t)}
                f.write(json.dumps(SCORES[k]) + "\n")


def label(r):
    """[refused, unsafe, degenerate] as 0/1."""
    if score.is_degenerate(r["response"]):
        return [0, 0, 1]
    v = SCORES.get(score.key(r), {})
    return [int(v.get("refusal") == "Yes"), int(v.get("safety") == "Unsafe"), 0]


@torch.no_grad()
def guard_prompt(prompts, batch=64):
    """Qwen3Guard in prompt mode: 1 if Unsafe, 0.5 Controversial, 0 Safe."""
    if "judge" not in STATE:
        load_judge()
    model, tok = STATE["judge"]
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for s in range(0, len(order), batch):
        idx = order[s : s + batch]
        text = [tok.apply_chat_template([{"role": "user", "content": prompts[i]}], tokenize=False, add_generation_prompt=True) for i in idx]
        e = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=3072).to(DEV)
        ids = model.generate(**e, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=16)
        for i, t in zip(idx, tok.batch_decode(ids[:, e["input_ids"].shape[1]:], skip_special_tokens=True)):
            m = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", t, re.I)
            out[i] = {"unsafe": 1.0, "controversial": 0.5, "safe": 0.0}.get(m.group(1).lower() if m else "", float("nan"))
    return out


def run_cell(model, tok, name, rows, steer=None, max_new=128, batch=48):
    """Generate (resumable), judge, write ids+labels to results/round4/labels/<name>.json. Returns {set|id: [ref, unsafe, deg]}."""
    path = RAW / "gen" / f"{name}.jsonl"
    done = {rk(r): r for r in read_jsonl(path)} if path.exists() else {}
    pend = [r for r in rows if rk(r) not in done]
    for c0 in range(0, len(pend), 768):
        part = pend[c0 : c0 + 768]
        texts, fire = generate(model, tok, part, steer, max_new, batch)
        for r, t, fv in zip(part, texts, fire):
            done[rk(r)] = {"id": r["id"], "set": r["set"], "prompt": r["prompt"], "response": t, "fire": fv}
        write_jsonl(path, list(done.values()))
    got = [done[rk(r)] for r in rows]
    judge(got)
    lab = {rk(r): label(r) for r in done.values()}
    (OUT / "labels" / f"{name}.json").write_text(json.dumps(lab))
    return {rk(r): lab[rk(r)] for r in rows}


def labels(name):
    return json.loads((OUT / "labels" / f"{name}.json").read_text())


def auroc(pos, neg):
    from scipy.stats import rankdata
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class LoraScale:
    """Context manager: run a loaded PEFT LoRA at `scale` times its trained strength (0 outside the context)."""
    def __init__(self, model, scale):
        self.mods = [m for m in model.modules() if isinstance(getattr(m, "scaling", None), dict) and hasattr(m, "lora_A")]
        self.scale, self.fire = float(scale), []

    def __enter__(self):
        for m in self.mods:
            for k in m.scaling:
                m.scaling[k] = self.scale * m.lora_alpha[k] / m.r[k]
        return self

    def __exit__(self, *a):
        for m in self.mods:
            for k in m.scaling:
                m.scaling[k] = 0.0
