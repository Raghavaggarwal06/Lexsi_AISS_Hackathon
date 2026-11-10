"""Prelude for task cast_compile (prepended to every sandbox run). numpy only.
Model sizes: CFG (config dict). Internal names start with "_"; avoid redefining the public functions below.

Model: Qwen1.5-1.8B-Chat (Qwen2 architecture). Every tensor is /case/weights/<name>.npy (fp16); names and shapes
are in /case/weights/index.json. Functions:
  W(name)                    float32 weight with the CURRENT edit applied (cached)
  set_edit(edit)             set the in-memory edit used by W/forward ({} = unedited). Same format as save_edit.
  forward(ids, cache=None)   -> (hidden, logits_last, cache). hidden[l] = residual stream at the INPUT of decoder
                             layer l (hidden[0] = embeddings, hidden[24] = after the last layer, before the final norm)
  generate(ids, n=30)        greedy, repetition penalty 1.1, stops at <|im_end|> -> list of new token ids
  decode(ids)                token ids -> text
  cast()                     dict: c, v (dict layer -> vector), meta (threshold, comparison, alpha, layers, algorithm)
  dev()                      list of dev prompts: prompt, ids, harmful, gate_sim, gate_fired, cast_response
  gate_sim(hidden_at_Lc)     CAST statistic cos(h_mean, tanh(P_c h_mean)) for a (T, d) block of residual states
  save_edit(edit)            validate and write /work/edit.npz (the deliverable)
Edit format: {param_name: full delta (same shape)} and/or {param_name + '::A': (r, in), param_name + '::B': (out, r)}
for a low-rank delta B @ A on a 2-D weight. W' = W + delta. Budget: 15,000,000 delta parameters in total.
"""
import json
import numpy as np
from pathlib import Path

CASE = Path("/case")
CFG = json.loads((CASE / "config.json").read_text())
_D, _NL, _NH = CFG["hidden_size"], CFG["num_hidden_layers"], CFG["num_attention_heads"]
_HD, _EPS, _THETA = _D // _NH, CFG["rms_norm_eps"], CFG["rope_theta"]
IM_END = 151645
BUDGET = 15_000_000
SHAPES = {n: tuple(s) for n, s in json.loads((CASE / "weights" / "index.json").read_text())}
_base, _cache, _EDIT = {}, {}, {}


def _load(name):
    if name not in _base:
        _base[name] = np.load(CASE / "weights" / f"{name}.npy", mmap_mode="r")
    return _base[name]


def _delta(name):
    if name in _EDIT:
        return np.asarray(_EDIT[name], np.float32)
    if name + "::A" in _EDIT:
        return np.asarray(_EDIT[name + "::B"], np.float32) @ np.asarray(_EDIT[name + "::A"], np.float32)
    return None


def W(name):
    if name not in _cache:
        w = np.asarray(_load(name), np.float32)
        d = _delta(name)
        _cache[name] = w + d if d is not None else w
    return _cache[name]


def set_edit(edit):
    global _EDIT
    _EDIT = dict(edit)
    for k in [k for k in _cache if k != "model.embed_tokens.weight"]:
        del _cache[k]
    if any(k.startswith("model.embed_tokens.weight") for k in _EDIT):
        _cache.pop("model.embed_tokens.weight", None)


def _rms(x, g):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + _EPS) * g


_INV = 1.0 / (_THETA ** (np.arange(0, _HD, 2, dtype=np.float64) / _HD))


def _rope(x, pos):                       # x: (T, _NH, _HD); rotate_half convention
    ang = np.outer(pos, _INV)            # (T, _HD/2)
    cos = np.cos(np.concatenate([ang, ang], -1)).astype(np.float32)[:, None, :]
    sin = np.sin(np.concatenate([ang, ang], -1)).astype(np.float32)[:, None, :]
    x1, x2 = x[..., :_HD // 2], x[..., _HD // 2:]
    return x * cos + np.concatenate([-x2, x1], -1) * sin


def forward(ids, cache=None):
    """ids: list of new token ids (the whole prompt, or the next token when a cache is given)."""
    ids = list(ids)
    start = 0 if cache is None else cache["n"]
    cache = cache or {"n": 0, "k": [None] * _NL, "v": [None] * _NL}
    pos = np.arange(start, start + len(ids))
    emb = _load("model.embed_tokens.weight")
    x = np.asarray(emb[ids], np.float32)
    d = _delta("model.embed_tokens.weight")
    if d is not None:
        x = x + d[ids]
    hidden = [x]
    for l in range(_NL):
        p = f"model.layers.{l}."
        h = _rms(x, W(p + "input_layernorm.weight"))
        q = h @ W(p + "self_attn.q_proj.weight").T + W(p + "self_attn.q_proj.bias")
        k = h @ W(p + "self_attn.k_proj.weight").T + W(p + "self_attn.k_proj.bias")
        v = h @ W(p + "self_attn.v_proj.weight").T + W(p + "self_attn.v_proj.bias")
        T = len(ids)
        q, k, v = (_rope(q.reshape(T, _NH, _HD), pos), _rope(k.reshape(T, _NH, _HD), pos), v.reshape(T, _NH, _HD))
        if cache["k"][l] is not None:
            k = np.concatenate([cache["k"][l], k]); v = np.concatenate([cache["v"][l], v])
        cache["k"][l], cache["v"][l] = k, v
        S = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(_HD)
        kpos = np.arange(k.shape[0])
        S = np.where(kpos[None, None, :] <= pos[None, :, None], S, -np.inf)
        S = np.exp(S - S.max(-1, keepdims=True)); S /= S.sum(-1, keepdims=True)
        a = np.einsum("hqk,khd->qhd", S, v).reshape(T, _D)
        x = x + a @ W(p + "self_attn.o_proj.weight").T
        h = _rms(x, W(p + "post_attention_layernorm.weight"))
        g = h @ W(p + "mlp.gate_proj.weight").T
        u = h @ W(p + "mlp.up_proj.weight").T
        x = x + ((g / (1 + np.exp(-g))) * u) @ W(p + "mlp.down_proj.weight").T
        hidden.append(x)
    cache["n"] = start + len(ids)
    logits = _rms(x[-1], W("model.norm.weight")) @ W("lm_head.weight").T
    return hidden, logits, cache


def generate(ids, n=30, rep=1.1):
    _, logits, cache = forward(ids)
    seen, out = set(ids), []
    for _ in range(n):
        lg = logits.copy()
        idx = np.array(sorted(seen))
        lg[idx] = np.where(lg[idx] > 0, lg[idx] / rep, lg[idx] * rep)
        t = int(lg.argmax())
        out.append(t)
        if t == IM_END:
            break
        seen.add(t)
        _, logits, cache = forward([t], cache)
    return out


def _byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_VOCAB = None


def decode(ids):
    global _VOCAB
    if _VOCAB is None:
        tj = json.loads((CASE / "tokenizer.json").read_text())
        _VOCAB = {i: t for t, i in tj["model"]["vocab"].items()}
        for t in tj.get("added_tokens", []):
            _VOCAB[t["id"]] = t["content"]
        _VOCAB["_bd"] = _byte_decoder()
    bd = _VOCAB["_bd"]
    out = bytearray()
    for i in ids:
        t = _VOCAB.get(int(i), "")
        if t.startswith("<|") and t.endswith("|>"):
            out += t.encode()
        else:
            out += bytes(bd.get(ch, 32) for ch in t)
    return out.decode("utf-8", errors="replace")


def cast():
    z = np.load(CASE / "cast" / "cast.npz")
    meta = json.loads((CASE / "cast" / "cast.json").read_text())
    return {"c": z["c"], "v": {int(k[2:]): z[k] for k in z.files if k.startswith("v_")}, "meta": meta}


def dev():
    return [json.loads(x) for x in open(CASE / "dev" / "dev.jsonl")]


def gate_sim(h_block, c=None):
    c = cast()["c"] if c is None else c
    h = np.asarray(h_block, np.float32).mean(0)
    proj = np.tanh((h @ c) / (c @ c) * c)
    return float(h @ proj / (np.linalg.norm(h) * np.linalg.norm(proj) + 1e-12))


def save_edit(edit, path="/work/edit.npz"):
    n, clean = 0, {}
    for k, v in edit.items():
        v = np.asarray(v, np.float32)
        if "::" in k:
            name, part = k.split("::")
            assert part in ("A", "B") and name in SHAPES and len(SHAPES[name]) == 2, f"bad low-rank key {k}"
            assert (name + "::A") in edit and (name + "::B") in edit, f"{name}: need both ::A and ::B"
            if part == "A":
                assert v.shape[1] == SHAPES[name][1], f"{k}: A must be (r, {SHAPES[name][1]})"
            else:
                assert v.shape[0] == SHAPES[name][0], f"{k}: B must be ({SHAPES[name][0]}, r)"
        else:
            assert k in SHAPES, f"unknown parameter {k}"
            assert v.shape == SHAPES[k], f"{k}: shape {v.shape} != {SHAPES[k]}"
        assert np.isfinite(v).all(), f"{k}: non-finite values"
        n += v.size
        clean[k] = v
    assert n <= BUDGET, f"edit has {n:,} parameters > budget {BUDGET:,}"
    np.savez(path, **clean)
    return f"saved {len(clean)} arrays, {n:,} parameters -> {path}"
