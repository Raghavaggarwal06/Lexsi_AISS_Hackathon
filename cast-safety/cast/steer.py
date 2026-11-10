"""Conditional activation steering (paper Sec 3.1, App A; release leash_layer.py), as forward pre-hooks.

At the INPUT of decoder layer l (= hidden_states[l]):
  condition layer, prompt pass only:  h_bar = mean of the prompt's hidden states (or the last one)
                                      sim   = cos(h_bar, tanh(P_c h_bar)),  P_c = c c^T / (c . c)
                                      met  |= (sim < theta)  if cmp == '<'   (release name: "larger")
                                              (sim > theta)  if cmp == '>'   (release name: "smaller")
  behaviour layer, every pass:         h <- h + alpha * v_l  for every position, in rows whose rule holds
                                      (the prompt pass included; release default apply_behavior_on_first_call=True)
A rule is  ([condition indices], 'or'|'and', behaviour index, negate=False). With no conditions, behaviours always apply.
Conditions are computed once per prompt; layers run in order, so a behaviour layer below its condition layer is not
steered on the prompt pass (same as the release). Batched: every row has its own condition state.
"""
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Condition:
    vecs: dict            # layer -> (d,) array
    layers: list          # condition layers (OR over layers, release check_condition)
    threshold: float
    cmp: str              # '<' or '>': the condition is met when sim <cmp> threshold
    mode: str = "mean"    # 'mean' | 'last'


@dataclass
class Behavior:
    vecs: dict
    layers: list
    alpha: float


@dataclass
class Rule:
    conds: list
    behavior: int
    op: str = "or"
    negate: bool = False  # duality: steer on the complement of the condition set (paper's "not c")


class Steerer:
    def __init__(self, lm, conditions=(), behaviors=(), rules=None, apply_on_prompt=True, record_layers=()):
        self.lm, self.conditions, self.behaviors = lm, list(conditions), list(behaviors)
        if rules is None:
            rules = [Rule(conds=[0], behavior=0)] if self.conditions and self.behaviors else \
                    [Rule(conds=[], behavior=b) for b in range(len(self.behaviors))]
        self.rules, self.apply_on_prompt = rules, apply_on_prompt
        dev, dt = lm.device, torch.float32
        self.cvec = [{l: torch.tensor(c.vecs[l], device=dev, dtype=dt) for l in c.layers} for c in self.conditions]
        self.bvec = [{l: torch.tensor(b.alpha * b.vecs[l], device=dev, dtype=dt) for l in b.layers}
                     for b in self.behaviors]
        self.record_layers = set(record_layers)       # also record sims at these layers (for the grid search)
        self.handles = []
        self.mask = None

    # ---- lifecycle
    def __enter__(self):
        layers = set(self.record_layers)
        for c in self.conditions: layers |= set(c.layers)
        for b in self.behaviors: layers |= set(b.layers)
        for l in sorted(layers):
            self.handles.append(self.lm.layers[l].register_forward_pre_hook(self._hook(l), with_kwargs=True))
        return self

    def __exit__(self, *a):
        for h in self.handles: h.remove()
        self.handles = []

    def begin(self, attention_mask):
        """Call before each generate/forward on a new batch of prompts."""
        B = attention_mask.shape[0]
        self.mask = attention_mask.float()
        self.met = torch.zeros((len(self.conditions), B), dtype=torch.bool, device=attention_mask.device)
        self.calls = {}
        self.sims = {}                                 # (cond index, layer) -> (B,) tensor

    # ---- rules
    def _active(self, rule):
        if not rule.conds:
            return torch.ones(self.met.shape[1], dtype=torch.bool, device=self.met.device)
        m = self.met[rule.conds]
        a = m.any(0) if rule.op == "or" else m.all(0)
        return ~a if rule.negate else a

    # ---- hook
    def _hook(self, l):
        def fn(module, args, kwargs):
            if args:
                h, rest, key = args[0], args[1:], None
            else:
                h, rest, key = kwargs["hidden_states"], (), "hidden_states"
            first = self.calls.get(l, 0) == 0
            self.calls[l] = self.calls.get(l, 0) + 1
            if first:
                hf = h.float()
                for ci, c in enumerate(self.conditions):
                    if l not in self.cvec[ci] and l not in self.record_layers:
                        continue
                    vec = self.cvec[ci].get(l)
                    if vec is None:
                        continue
                    if c.mode == "mean":
                        w = self.mask / self.mask.sum(1, keepdim=True)
                        hb = (hf * w.unsqueeze(-1)).sum(1)
                    else:
                        hb = hf[:, -1]                  # prompts are left-padded for generation
                    proj = torch.tanh((hb @ vec / (vec @ vec)).unsqueeze(1) * vec.unsqueeze(0))
                    sim = torch.nn.functional.cosine_similarity(hb, proj, dim=1)
                    self.sims[(ci, l)] = sim.detach()
                    if l in c.layers:
                        self.met[ci] |= (sim < c.threshold) if c.cmp == "<" else (sim > c.threshold)
            add = None
            for r in self.rules:
                b = self.bvec[r.behavior].get(l)
                if b is None or (first and not self.apply_on_prompt):
                    continue
                act = self._active(r).to(h.dtype)
                if act.any():
                    term = act[:, None, None] * b.to(h.dtype)[None, None, :]
                    add = term if add is None else add + term
            if add is None:
                return None
            h = h + add
            return ((h,) + tuple(rest), kwargs) if key is None else (args, {**kwargs, key: h})
        return fn


@torch.no_grad()
def generate(lm, prompts, steerer=None, max_new_tokens=50, batch_size=32, repetition_penalty=1.1, return_ids=False):
    """Greedy generation as in the release's respond(): chat template + generation prompt, 50 new tokens,
    repetition_penalty 1.1. Returns (responses, per-prompt dict of condition info[, new token ids up to and
    including the first EOS])."""
    texts = [lm.chat(p, gen_prompt=True) for p in prompts]
    order = np.argsort([-len(t) for t in texts])
    resp, info, rids = [None] * len(texts), [None] * len(texts), [None] * len(texts)
    eos = lm.model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {lm.tok.eos_token_id}
    for s in range(0, len(texts), batch_size):
        idx = order[s:s + batch_size]
        ids, att, _ = lm.encode([texts[i] for i in idx], side="left")
        if steerer is not None:
            steerer.begin(att)
        out = lm.model.generate(input_ids=ids, attention_mask=att, do_sample=False, max_new_tokens=max_new_tokens,
                                repetition_penalty=repetition_penalty, pad_token_id=lm.pad_id)
        for j, i in enumerate(idx):
            resp[i] = lm.tok.decode(out[j, ids.shape[1]:], skip_special_tokens=True)
            new = out[j, ids.shape[1]:].tolist()
            for k, t in enumerate(new):
                if t in eos:
                    new = new[:k + 1]
                    break
            rids[i] = new
            if steerer is not None:
                info[i] = {"met": [bool(x) for x in steerer.met[:, j].tolist()],
                           "sims": {f"c{ci}_L{l}": float(v[j]) for (ci, l), v in steerer.sims.items()}}
    return (resp, info, rids) if return_ids else (resp, info)


@torch.no_grad()
def condition_sims_multi(lm, prompts, cvecs_by_name, layers, mode="mean", batch_size=32):
    """sim(h, tanh(P_c h)) at the input of each layer for each prompt (prompt pass only, no steering):
    exactly the quantity the condition hook thresholds. Returns {name: {layer: (N,) array}}."""
    texts = [lm.chat(p, gen_prompt=True) for p in prompts]
    order = np.argsort([-len(t) for t in texts])
    res = {n: {l: np.zeros(len(texts), np.float32) for l in layers} for n in cvecs_by_name}
    vec = {n: {l: torch.tensor(cv[l], device=lm.device, dtype=torch.float32) for l in layers}
           for n, cv in cvecs_by_name.items()}
    for s in range(0, len(texts), batch_size):
        idx = order[s:s + batch_size]
        ids, att, _ = lm.encode([texts[i] for i in idx], side="left")
        hs = lm.model(input_ids=ids, attention_mask=att, output_hidden_states=True, logits_to_keep=1).hidden_states
        w = (att.float() / att.float().sum(1, keepdim=True)).unsqueeze(-1)
        for l in layers:
            hf = hs[l].float()                         # INPUT of decoder layer l
            hb = (hf * w).sum(1) if mode == "mean" else hf[:, -1]
            for n in cvecs_by_name:
                v = vec[n][l]
                proj = torch.tanh((hb @ v / (v @ v)).unsqueeze(1) * v.unsqueeze(0))
                res[n][l][idx] = torch.nn.functional.cosine_similarity(hb, proj, dim=1).cpu().numpy()
        del hs
    return res


def condition_sims(lm, prompts, cvecs, layers, mode="mean", batch_size=32):
    return condition_sims_multi(lm, prompts, {"c": cvecs}, layers, mode, batch_size)["c"]
