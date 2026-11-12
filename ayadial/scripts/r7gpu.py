"""Round 7 GPU side: rounds 4 to 6's runtime code (r4lib, r5lib, r6lib) pointed at the round 7 folders, plus SafeTune 0.1.13.
Import it before any r4/r5/r6 module in a round 7 process. It runs on the Colab A100; the smoke stage also runs on MPS.

What it reuses unchanged: r4lib.load (model and tokenizer), r4lib.enc and the chat template (r5lib's RAWMAP for rows that carry their own rendered
chat, as round 6 did for the formats), r4lib.generate (greedy, 128 new tokens, batches sorted by length with the suite's token budget) and
r4lib.run_cell (resumable per 768 rows, judge, label file), r4lib.label and r5lib's degeneracy rule. What it adds:
  - judge(): r4lib.judge with r5_judge2.py's halve-and-retry loop on out-of-memory (scripts with about one token per character overrun the batch).
  - OomSafe: a generation that runs out of memory is split in two halves (same padded tensors, rows kept in order) and retried; every split is logged.
  - SafeTune wrappers rebuilt from a saved calibration (runs_private/round7/calib/st_<arm>.pt), so that every process uses the same fitted state.
  - Four patches to SafeTune that change memory and time only, never a number (checked bit for bit by the smoke stage, journal J0):
      P1 the vector extractor keeps only the last position of each layer when it pools the last token (it kept every position of every layer on
         the CPU: about 780 GB for the 7555 calibration prompts at 36 layers);
      P2 during calibration the model computes logits for the last position only (the extractor, the CAST condition fit and the layer sweep read
         hidden states or the last position's logits; full logits are 16 x T x 262144 values per batch);
      P3 CAST's gate pass computes the last position's logits only (round 5's CAST asked for 37.6 GiB in one allocation at the default batch);
      P4 extract_refusal_direction is memoised within a process, so AdaSteer reuses RefusalDirection's identical call on the same prompts."""
import contextlib, dataclasses, functools, gc, hashlib, importlib, json, os, time
from pathlib import Path
import numpy as np
import torch
import r7lib as L7
import r6lib, r5lib, r4lib  # r6lib imports r5lib and r4lib; their folders are repointed below
from r4lib import STATE, SCORES, score, log, read_jsonl, write_jsonl, rk

for _m in (r4lib, r5lib, r6lib):
    _m.OUT, _m.RAW, _m.CACHE, _m.DRIVE = L7.OUT, L7.RAW, L7.CACHE, L7.DRIVE
    _m.sync = L7.sync
for _p in (L7.OUT / "labels", L7.OUT / "run", L7.RAW / "gen", L7.RAW / "calib", L7.CACHE):
    _p.mkdir(parents=True, exist_ok=True)
DEV, RAWMAP = r4lib.DEV, r5lib.RAWMAP
CUDA = torch.cuda.is_available() and str(DEV).startswith("cuda")
HOOKS0 = [None]


def free():
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()


def versions():
    import transformers
    try:
        import safetune
        sv = getattr(safetune, "__version__", "?")
    except Exception:
        sv = None
    return {"torch": torch.__version__, "transformers": transformers.__version__, "safetune": sv, "device": str(DEV),
            "gpu": torch.cuda.get_device_name(0) if CUDA else None}


# --- model, hooks
def n_hooks(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def assert_clean(model, where):
    n = n_hooks(model)
    assert n == HOOKS0[0], f"{where}: {n} forward hooks on the model, {HOOKS0[0]} expected (a SafeTune wrapper left hooks behind)"


def load():
    model, tok = r4lib.load()
    HOOKS0[0] = n_hooks(model)
    log(f"model on {DEV}; {HOOKS0[0]} hooks at load; {versions()}")
    return model, tok


# --- judge with halve-and-retry
@torch.no_grad()
def judge(rows, batch=64):
    """r4lib.judge (Qwen3Guard-Gen-0.6B, same template, parse and cache) with r5_judge2.py's loop: on out-of-memory, halve the batch and retry."""
    if "judge" not in STATE:
        r4lib.load_judge()
    todo = {score.key(r): r for r in rows if score.key(r) not in SCORES and not score.is_degenerate(r["response"])}
    model, tok = STATE["judge"]
    items = sorted(todo.items(), key=lambda kv: len(kv[1]["prompt"]) + len(kv[1]["response"]))
    with open(L7.RAW / "scores.jsonl", "a") as f:
        i = 0
        while i < len(items):
            bs = batch
            while True:
                chunk = items[i : i + bs]
                try:
                    text = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}], tokenize=False, add_generation_prompt=False) for _, r in chunk]
                    e = tok(text, return_tensors="pt", padding=True, truncation=True, max_length=3072).to(DEV)
                    ids = model.generate(**e, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=32)
                    break
                except torch.cuda.OutOfMemoryError:
                    e = ids = None
                    free()
                    if bs == 1:
                        raise
                    bs = max(1, bs // 2)
                    log(f"judge: out of memory, batch lowered to {bs}")
            for (k, _), t in zip(chunk, tok.batch_decode(ids[:, e["input_ids"].shape[1]:], skip_special_tokens=True)):
                SCORES[k] = {"key": k, **score.parse(t)}
                f.write(json.dumps(SCORES[k]) + "\n")
            i += len(chunk)


r4lib.judge = judge  # run_cell calls the module's judge


class OomSafe:
    """Wraps whatever generates (the model or a SafeTune wrapper). A batch that runs out of memory is split in two halves of the same padded
    tensors and retried (recursively); outputs are right-padded with the pad token, which decoding drops. Splits are counted and logged."""
    def __init__(self, inner):
        self.inner, self.splits = inner, []

    def generate(self, **kw):
        try:
            return self.inner.generate(**kw)
        except torch.cuda.OutOfMemoryError:
            pass
        free()
        ids, am = kw["input_ids"], kw.get("attention_mask")
        n = ids.shape[0]
        if n == 1:
            raise RuntimeError("out of memory on a single prompt")
        self.splits.append([int(n), int(ids.shape[1])])
        log(f"generation: out of memory on a batch of {n} x {ids.shape[1]} tokens; split in two")
        outs = []
        for sl in (slice(0, n // 2), slice(n // 2, n)):
            k2 = dict(kw, input_ids=ids[sl])
            if am is not None:
                k2["attention_mask"] = am[sl]
            outs.append(self.generate(**k2))
        width = max(o.shape[1] for o in outs)
        pad = kw.get("pad_token_id")
        return torch.cat([torch.nn.functional.pad(o, (0, width - o.shape[1]), value=pad) for o in outs], 0)


def run(model, tok, suite, name, rows=None):
    """One cell: generate (resumable), judge, write results/round7/labels/<name>.json ({set|id: [refused, unsafe, degenerate]}), mirror to Drive.
    The batching is the suite's (r7lib.env). A cell whose label file already covers every row is not run again."""
    rows = L7.suite_rows(suite) if rows is None else rows
    lp = L7.OUT / "labels" / f"{name}.json"
    if lp.exists():
        lab = L7.J(lp)
        if all(rk(r) in lab for r in rows):
            return {rk(r): lab[rk(r)] for r in rows}
    RAWMAP.clear()  # one suite at a time: the map is keyed by request text
    RAWMAP.update({r["prompt"]: r["raw"] for r in rows if r.get("raw")})
    tb, bs = L7.env(suite)
    os.environ["R4_TOK"], os.environ["R4_BATCH"] = str(tb), str(bs)
    t0 = time.time()
    proxy = OomSafe(model)
    lab = r4lib.run_cell(proxy, tok, name, rows, None, max_new=int(os.environ.get("R7_MAXNEW", 128)))
    sec = round(time.time() - t0)
    mp = L7.OUT / "run" / "cells.json"
    meta = L7.J(mp) if mp.exists() else {}
    meta[name] = {"suite": suite, "n": len(rows), "batching": [tb, bs], "max_new": int(os.environ.get("R7_MAXNEW", 128)), "seconds": sec, "oom_splits": proxy.splits,
                  "written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
    L7.write_json(mp, meta)
    a = np.array([lab[rk(r)] for r in rows], float)
    log(f"{name}: n={len(rows)} {sec}s refused {a[:, 0].mean() * 100:.1f} unsafe {a[:, 1].mean() * 100:.1f} degenerate {a[:, 2].mean() * 100:.1f}" + (f" splits {len(proxy.splits)}" if proxy.splits else ""))
    L7.sync()
    return lab


# --- SafeTune patches (memory and time only)
PATCHES = []  # (owner, attribute, original, replacement)
ERD = {"calls": 0, "hits": 0, "last": None}


def _set(owner, attr, new):
    PATCHES.append((owner, attr, getattr(owner, attr), new))
    setattr(owner, attr, new)


def patch_safetune():
    if PATCHES:
        return
    from safetune.core.runtime.inference import vector_extraction as VE
    from safetune.interventions.steer import cast as CM
    S_alias, S_real = importlib.import_module("safetune.steer"), importlib.import_module("safetune.interventions.steer")

    orig_make_hook = VE.SteeringVectorExtractor._make_hook

    def make_hook(self, layer_idx):  # P1
        if self.config.pool_method != "last_token":
            return orig_make_hook(self, layer_idx)

        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self._activations.setdefault(layer_idx, []).append(h[:, -1:, :].detach().cpu())  # _pool takes hidden[:, -1, :] of this
        return hook_fn
    _set(VE.SteeringVectorExtractor, "_make_hook", make_hook)

    orig_gate = CM.CASTModel._gate_fires

    def gate(self, input_ids, **kwargs):  # P3
        kwargs.setdefault("logits_to_keep", 1)
        return orig_gate(self, input_ids, **kwargs)
    _set(CM.CASTModel, "_gate_fires", gate)

    orig_pool_hidden = CM._pool_hidden

    def pool_hidden(h, pool):  # P5 (J3): with pool="last" the original returns a view, which keeps every prompt's full (seq, hidden) float32 tensor alive
        return orig_pool_hidden(h, pool).clone()  # for every candidate layer (170 GB of RAM on 7555 prompts); the copy has the same values
    _set(CM, "_pool_hidden", pool_hidden)

    orig_erd = S_real.extract_refusal_direction

    def erd(model, tokenizer, harmful_prompts, harmless_prompts, config=None):  # P4
        key = (id(model), hashlib.sha256(json.dumps([harmful_prompts, harmless_prompts, repr(config)]).encode()).hexdigest())
        ERD["calls"] += 1
        if ERD.get("key") == key:
            ERD["hits"] += 1
        else:
            ERD["key"], ERD["last"] = key, orig_erd(model, tokenizer, harmful_prompts, harmless_prompts, config)
        return ERD["last"]
    for owner in {id(S_alias): S_alias, id(S_real): S_real}.values():
        _set(owner, "extract_refusal_direction", erd)


@contextlib.contextmanager
def unpatched():
    """SafeTune exactly as installed (for the smoke stage's comparisons)."""
    saved = [(o, a, getattr(o, a)) for o, a, _, _ in PATCHES]
    for o, a, orig, _ in PATCHES:
        setattr(o, a, orig)
    try:
        yield
    finally:
        for o, a, cur in saved:
            setattr(o, a, cur)


@contextlib.contextmanager
def last_logits_only(model):
    """P2: while it is active, a forward pass that does not say otherwise computes the logits of the last position only (hidden states unchanged)."""
    orig = model.forward

    @functools.wraps(orig)
    def fwd(*a, **kw):
        kw.setdefault("logits_to_keep", 1)
        return orig(*a, **kw)
    model.forward = fwd
    try:
        yield
    finally:
        del model.forward


# --- calibration and wrappers
def state_path(arm):
    return L7.RAW / "calib" / f"st_{arm}.pt"


def load_state(arm):
    return torch.load(state_path(arm), weights_only=False)


def calibrate(model, tok, arm, C, force=False, save=True, p2=True):
    """SafeTune's own trainer on the calibration set (every row AyaDial's harm direction was fit on, r7lib.calibration), as round 6 called it
    (RefusalDirectionTrainer, AdaSteerTrainer, CASTTrainer, defaults otherwise). The fitted state is saved and reused by every later
    process; the file's sha256 is recorded in results/round7/safetune_pick.json. p2=False runs without the last-position-logits patch (smoke)."""
    if state_path(arm).exists() and not force:
        return load_state(arm)
    patch_safetune()
    from safetune.runner.steer import AdaSteerTrainer, CASTTrainer, RefusalDirectionTrainer
    hp, bp = [r["prompt"] for r in C["harmful"]["rows"]], [r["prompt"] for r in C["harmless"]["rows"]]
    assert_clean(model, f"before calibrating {arm}")
    t0 = time.time()
    log(f"calibrating {arm} on {len(hp)} harmful + {len(bp)} harmless prompts")
    with (last_logits_only(model) if p2 else contextlib.nullcontext()):
        if arm == "rd":
            c0 = ERD["calls"]
            w, _ = RefusalDirectionTrainer(model, tok, alpha=1.0).calibrate(harmful=hp, harmless=bp)
            idx = int(ERD["last"][1]) if ERD["calls"] > c0 else None  # the layer SafeTune's sweep selected (known when the memo saw the call)
            st = {"arm": "rd", "direction": w.direction.detach().clone().float().cpu(), "layer_selected": idx, "steer_layers": None}  # J2: None = every layer, SafeTune's default
        elif arm == "adasteer":
            w, _ = AdaSteerTrainer(model, tok, alpha=1.0).calibrate(harmful=hp, harmless=bp)
            st = {"arm": "adasteer", "rejection_direction": {int(k): np.asarray(v, np.float32) for k, v in w.rejection_direction.items()},
                  "rd_probe_layer": int(w.config.rd_probe_layer), "hd_probe_layer": int(w.config.hd_probe_layer), "target_layers": [int(x) for x in w.config.target_layers],
                  "rd_lr": w.coeff._rd_lr, "hd_lr": w.coeff._hd_lr}
            w.remove_hooks()
        else:
            w, _ = CASTTrainer(model, tok, alpha=1.0).calibrate(harmful=hp, harmless=bp)
            cond = w.condition
            st = {"arm": "cast", "steering_vectors": {int(k): v.detach().float().cpu() for k, v in w.steering_vectors.items()},
                  "condition": {f.name: getattr(cond, f.name) for f in dataclasses.fields(cond)}, "per_prompt_gate": bool(w.per_prompt_gate)}
            w.remove()
    st.update(seconds=round(time.time() - t0), n=[len(hp), len(bp)], erd_calls=ERD["calls"], erd_hits=ERD["hits"], versions=versions())
    assert_clean(model, f"after calibrating {arm}")
    if save:
        torch.save(st, state_path(arm))
        L7.sync()
    log(f"calibrated {arm} in {st['seconds']}s")
    free()
    return st


def build(model, arm, s, st):
    """The SafeTune wrapper for one setting, rebuilt from the saved calibration exactly as the trainer builds it (setting in place of alpha)."""
    patch_safetune()
    S = importlib.import_module("safetune.steer")
    if arm == "rd":
        w = S.RefusalDirectionModel(model, direction=st["direction"], mode="steer", strength=float(s), layers=st["steer_layers"])
        w.direction = st["direction"].clone()  # the trainer's unit vector bit for bit (the constructor divides by a norm of 1 +- 1e-7 again)
        return w
    if arm == "adasteer":  # AdaSteerTrainer: AdaSteerModel(rejection_direction, target_layers=None, base_multiplier=alpha, probes) + fit_adaptive
        w = S.AdaSteerModel(model, rejection_direction=st["rejection_direction"], target_layers=None, base_multiplier=float(s),
                            rd_probe_layer=st["rd_probe_layer"], hd_probe_layer=st["hd_probe_layer"])
        assert list(w.config.target_layers) == st["target_layers"]
        w.coeff._rd_lr, w.coeff._hd_lr, w.coeff.fitted = st["rd_lr"], st["hd_lr"], True  # the fitted logistic laws (fit_adaptive's state)
        return w
    from safetune.interventions.steer.cast import CASTCondition, CASTModel
    return CASTModel(model, steering_vectors=st["steering_vectors"], condition=CASTCondition(**st["condition"]), alpha=float(s), per_prompt_gate=st["per_prompt_gate"])


def cleanup(w, arm):
    (w.remove_hooks if arm == "adasteer" else w.remove)()


@contextlib.contextmanager
def method(model, arm, s, st):
    assert_clean(model, f"before {arm}@{s:g}")
    w = build(model, arm, s, st)
    try:
        yield w
    finally:
        cleanup(w, arm)
        assert_clean(model, f"after {arm}@{s:g}")


# --- batches exactly as r4lib.run_cell + r4lib.generate form them
def batches(rows, env_):
    tb, bmax = env_
    out = []
    for c0 in range(0, len(rows), 768):
        part = rows[c0 : c0 + 768]
        order = sorted(range(len(part)), key=lambda i: len(part[i]["prompt"]))
        s = 0
        while s < len(order):
            bs = int(max(8, min(bmax, tb // (420 + len(part[order[min(s + 48, len(order)) - 1]]["prompt"]) // 3))))
            out.append([part[i] for i in order[s : s + bs]])
            s += bs
    return out


# --- the CAST check (development prompts only)
def _layer_out(model, layer, e, install=None, remove=None):
    cap = {}
    if install:
        install()
    h = model.model.layers[layer].register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[:, -1].float().cpu()))
    try:
        with torch.no_grad():
            model(**e, logits_to_keep=1)
    finally:
        h.remove()
        if remove:
            remove()
    return cap["h"]


def alpha_reach(model, tok, st, prompts, alphas=(0.01, 0.1, 1.0)):
    """Does alpha reach the forward pass? At the first behaviour layer the steered output minus the plain output must be alpha * v (the layer's
    input is not steered). Reported per alpha: |delta| / (alpha |v|) and cos(delta, v), at the last prompt position."""
    w = build(model, "cast", 1.0, st)
    l0 = min(st["steering_vectors"])
    v = st["steering_vectors"][l0].float()
    RAWMAP.clear()
    e = r4lib.enc(tok, prompts)
    plain = _layer_out(model, l0, e)
    out = {"layer": int(l0), "vector_norm": float(v.norm()), "residual_norm": float(plain.norm(dim=-1).mean())}
    for a in alphas:
        w.alpha = float(a)
        h = _layer_out(model, l0, e, install=w._install_steering, remove=w._remove_steering)
        dlt = h - plain
        out[f"{a:g}"] = {"ratio": float((dlt.norm(dim=-1) / (a * v.norm())).mean()), "cos": float(torch.nn.functional.cosine_similarity(dlt, v[None], dim=-1).mean())}
    w.remove()
    assert_clean(model, "after the alpha check")
    return out


def gate_compare(model, tok, st, rows, bs=8):
    """CAST's gate with P3 against SafeTune's own gate, on the same batches: decisions and similarities must be identical."""
    w = build(model, "cast", 1.0, st)
    RAWMAP.clear()
    same, n, fired, per_row = True, 0, {}, {}
    maxdiff = 0.0
    for i in range(0, len(rows), bs):
        part = rows[i : i + bs]
        e = r4lib.enc(tok, [r["prompt"] for r in part])
        f1, s1 = w._gate_fires(e["input_ids"], attention_mask=e["attention_mask"])
        with unpatched():
            f0, s0 = w._gate_fires(e["input_ids"], attention_mask=e["attention_mask"])
        same &= (list(f1) == list(f0))
        maxdiff = max(maxdiff, max(abs(a - b) for a, b in zip(s1, s0)))
        for r, f in zip(part, f1):
            side = "harmful" if r["set"] in L7.WJ_H or r.get("harm") == 1 else "harmless"
            fired.setdefault(side, []).append(int(f))
            per_row[rk(r)] = int(f)
        n += len(part)
    w.remove()
    return {"n": n, "decisions_identical": bool(same), "similarity_max_abs_diff": maxdiff, "fires": {k: float(np.mean(v)) for k, v in fired.items()}}, per_row


def memtest(model, tok, st, suites=("wjx", "frz5", "xb6")):
    """CAST (alpha 1, gate with P3) on the heaviest batch of each suite at that suite's batching, 128 new tokens. Base on the same batch for scale."""
    out, ok = {}, True
    if not CUDA:
        return {"skipped": "no CUDA"}, True
    RAWMAP.clear()
    for su in suites:
        bl = batches(L7.suite_rows(su), L7.env(su))
        sizes = []
        for b in bl:
            e = r4lib.enc(tok, [r["prompt"] for r in b])
            sizes.append(int(e["input_ids"].shape[0] * (e["input_ids"].shape[1] + 128)))  # the cache grows by 128 tokens per row
        b = bl[int(np.argmax(sizes))]
        e = r4lib.enc(tok, [r["prompt"] for r in b])
        kw = dict(do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=128, pad_token_id=tok.pad_token_id)
        res = {"batch": list(e["input_ids"].shape)}
        for who in ("base", "cast"):
            free()
            torch.cuda.reset_peak_memory_stats()
            w = build(model, "cast", 1.0, st) if who == "cast" else model
            try:
                with torch.no_grad():
                    w.generate(**e, **kw)
                res[who] = {"ok": True, "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            except torch.cuda.OutOfMemoryError:
                res[who] = {"ok": False}
            finally:
                if who == "cast":
                    w.remove()
            free()
        if su == suites[0]:  # the decision reads the development suite; on the test suites a batch that does not fit is split by OomSafe (recorded)
            ok &= res["cast"]["ok"]
        out[su] = res
        log(f"memtest {su}: {res}")
    assert_clean(model, "after the memory test")
    return out, bool(ok)


def castcheck(model, tok, st, run_cells=True):
    """Stage 'pick cast', before its grid. Writes results/round7/castcheck.json (numbers only; replies under runs_private/round7/gen)."""
    p = L7.OUT / "castcheck.json"
    if p.exists():
        return L7.J(p)
    rows = L7.suite_rows("cc64")
    res = {"rows": "cc64: 32 harmful + 32 harmless WildJailbreak development rows (random.Random(0))", "behaviour_layers": sorted(int(k) for k in st["steering_vectors"]),
           "vector_norms": {str(k): float(v.norm()) for k, v in sorted(st["steering_vectors"].items())},
           "condition": {k: (v if not torch.is_tensor(v) else None) for k, v in st["condition"].items() if k != "condition_vector"}}
    res["gate"], fires = gate_compare(model, tok, st, rows)  # gate decisions per prompt, batches of 8 in list order
    res["alpha_in_forward"] = alpha_reach(model, tok, st, [r["prompt"] for r in rows][:8])
    if run_cells:  # replies at several alphas (64 new tokens), batches of 8, with a Base and alpha 0 in the same batching
        su = "cc64_b8"
        old = os.environ.get("R7_MAXNEW")
        os.environ["R7_MAXNEW"] = "64"
        try:
            base = run(model, tok, su, f"base_{su}", rows)
            gen = {}
            cells = {}
            for a in (0.0, 0.01, 0.03, 0.1, 0.3, 1.0):
                with method(model, "cast", a, st) as w:
                    lab = run(w, tok, su, L7.cell(su, "cast", a), rows)
                gen[a] = {rk(r): r["response"] for r in read_jsonl(L7.RAW / "gen" / f"{L7.cell(su, 'cast', a)}.jsonl")}
                cells[a] = lab
        finally:
            if old is None:
                os.environ.pop("R7_MAXNEW", None)
            else:
                os.environ["R7_MAXNEW"] = old
        g = [k for k in fires if fires[k]]
        bgen = {rk(r): r["response"] for r in read_jsonl(L7.RAW / "gen" / f"base_{su}.jsonl")}
        res["generation"] = {"gated_rows": len(g), "alpha0_identical_to_base": float(np.mean([gen[0.0][k] == bgen[k] for k in gen[0.0]]))}
        for a in (0.01, 0.03, 0.1, 0.3, 1.0):
            res["generation"][f"{a:g}"] = {"differs_from_alpha0_among_gated": float(np.mean([gen[a][k] != gen[0.0][k] for k in g])) if g else None,
                                          "identical_to_alpha1_among_gated": float(np.mean([gen[a][k] == gen[1.0][k] for k in g])) if g else None,
                                          "refused_among_gated": float(np.mean([cells[a][k][0] for k in g]) * 100) if g else None,
                                          "refused_all": float(np.mean([v[0] for v in cells[a].values()]) * 100)}
        res["generation"]["base_refused_among_gated"] = float(np.mean([base[k][0] for k in g]) * 100) if g else None
    res["memtest"], mem_ok = memtest(model, tok, st)
    reach = all(abs(v["ratio"] - 1) < 0.1 and v["cos"] > 0.99 for k, v in res["alpha_in_forward"].items() if k in ("0.1", "1"))
    moves = bool(res.get("generation")) and any((res["generation"][f"{a:g}"]["differs_from_alpha0_among_gated"] or 0) > 0 for a in (0.01, 0.03, 0.1, 0.3, 1.0))
    res["decision"] = {"alpha_reaches_forward": bool(reach), "alpha_changes_replies": bool(moves),
                       "cast_batching": "suite" if mem_ok else "b8",
                       "cast_grid": (L7.GRID["cast"] if mem_ok else L7.GRID_CAST_B8) if (reach and moves) else [1.0],
                       "gate_patch_identical": res["gate"]["decisions_identical"],
                       "rule": "grid as declared in J0 if alpha reaches the forward pass (delta/(alpha|v|) within 10% of 1 at alpha 0.1 and 1) and changes at least one gated reply; "
                               "otherwise SafeTune's default alpha 1 alone. Suite batching if CAST fits the heaviest batch of wjx (frz5 and xb6 are measured too: a test batch that "
                               "does not fit is split in two and retried, and the split is recorded in run/cells.json); otherwise batches of 8 with its own Base."}
    res["versions"], res["written_utc"] = versions(), time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    L7.write_json(p, res)
    L7.sync()
    return res
