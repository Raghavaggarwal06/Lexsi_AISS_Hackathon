"""The fold: one dormant unit of layers[22].mlp rewritten from measured vectors (no training).
  gate row = -kappa0*m - kappa*q + s*alpha*w     (s, the knob, sits behind SiLU)
  up row   = m                                    (1 at the marker, ~0 elsewhere)
  down col = gain*r/alpha
mode 'weights' edits the three weight slices in place; mode 'hook' zeroes the unit and adds the same function,
computed in float32 from the MLP input, to the layer output. Run as a script for the parity test."""
import sys
from r4lib import *
UNIT = 6666  # most dormant layers[22] unit in round 3 (block 3.3)


class Fold:
    def __init__(self, model, s, mode="weights", gain=None, kappa=None, unit=UNIT, file="fold.npz", layer=22, cap_beta=None, unit2=1896):
        P = np.load(CACHE / file)
        alpha, kappa_f, gain_f, theta0, _, _ = P["consts"]
        self.alpha, self.kappa, self.gain, self.theta0 = float(alpha), float(kappa if kappa is not None else kappa_f), float(gain if gain is not None else gain_f), float(theta0)
        r = np.load(ROOT / "results/day1/directions.npz")["r_raw"][23]
        self.mlp, self.block, self.n, self.mode, self.s = model.model.layers[layer].mlp, model.model.layers[layer], unit, mode, float(s)
        W = self.mlp.gate_proj.weight
        t32 = lambda v: torch.tensor(np.asarray(v, np.float32), device=W.device)
        self.g32 = t32(-self.alpha * self.theta0 * P["m"] - self.kappa * P["q"] + self.s * self.alpha * P["w"])
        self.m32, self.d32 = t32(P["m"]), t32(self.gain * r / self.alpha)
        self.fire, self.handles, self.saved = [], [], None
        self.second, self.dscale = None, self.gain / self.alpha
        if cap_beta:  # second dormant unit subtracts the part of the ramp above 1/beta, so the push saturates at gain*r
            self.g2 = t32(-self.alpha * (self.theta0 + 1.0 / cap_beta) * P["m"] - self.kappa * P["q"] + self.s * self.alpha * P["w"])
            self.d32 = t32(self.gain * cap_beta * r / self.alpha)
            self.second, self.dscale = unit2, self.gain * cap_beta / self.alpha

    def _pre(self, _m, a):
        self.x = a[0]

    def _rec(self, _m, a):  # input of down_proj: the unit's activation
        if a[0].shape[1] > 1:
            act = a[0][:, -1, self.n].float() - (a[0][:, -1, self.second].float() if self.second is not None else 0)
            self.fire.append((act * self.dscale).cpu())

    def _add(self, _m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        x = self.x.float()
        act = torch.nn.functional.silu(x @ self.g32) * (x @ self.m32)
        if self.second is not None:
            act = act - torch.nn.functional.silu(x @ self.g2) * (x @ self.m32)
        if h.shape[1] > 1:
            self.fire.append((act[:, -1] * self.dscale).cpu())
        pushed = h + (act.unsqueeze(-1) * self.d32).to(h.dtype)
        return (pushed, *out[1:]) if isinstance(out, tuple) else pushed

    def __enter__(self):
        G, Uw, D = self.mlp.gate_proj.weight, self.mlp.up_proj.weight, self.mlp.down_proj.weight
        self.saved = (G.data[self.n].clone(), Uw.data[self.n].clone(), D.data[:, self.n].clone())
        if self.second is not None:
            k = self.second
            self.saved2 = (G.data[k].clone(), Uw.data[k].clone(), D.data[:, k].clone())
            if self.mode == "weights":
                G.data[k], Uw.data[k], D.data[:, k] = self.g2.to(G.dtype), self.m32.to(G.dtype), (-self.d32).to(G.dtype)
            else:
                D.data[:, k] = 0
        if self.mode == "weights":
            G.data[self.n], Uw.data[self.n], D.data[:, self.n] = self.g32.to(G.dtype), self.m32.to(G.dtype), self.d32.to(G.dtype)
            self.handles = [self.mlp.down_proj.register_forward_pre_hook(self._rec)]
        else:
            D.data[:, self.n] = 0
            self.handles = [self.mlp.register_forward_pre_hook(self._pre), self.block.register_forward_hook(self._add)]
        return self

    def __exit__(self, *a):
        G, Uw, D = self.mlp.gate_proj.weight, self.mlp.up_proj.weight, self.mlp.down_proj.weight
        G.data[self.n], Uw.data[self.n], D.data[:, self.n] = self.saved
        if self.second is not None:
            G.data[self.second], Uw.data[self.second], D.data[:, self.second] = self.saved2
        for h in self.handles:
            h.remove()


@torch.no_grad()
def logits_at(model, tok, rows, resp, ctx=None, n_gen=16, batch=8):
    """log-probs at the marker (next token) and at up to n_gen teacher-forced reply positions, right-aligned per row."""
    first, rest = [], []
    for s in range(0, len(rows), batch):
        part = rows[s : s + batch]
        texts = [render(tok, r["prompt"]) + resp[rk(r)] for r in part]
        plen = [len(tok(render(tok, r["prompt"]), add_special_tokens=False)["input_ids"]) for r in part]
        tok.padding_side = "right"
        e = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN + 200, add_special_tokens=False).to(DEV)
        tok.padding_side = "left"
        if ctx is None:
            lg = model(**e, use_cache=False).logits
        else:
            ctx.fire = []
            with ctx:
                lg = model(**e, use_cache=False).logits
        for i, p in enumerate(plen):
            T = int(e["attention_mask"][i].sum())
            lp = torch.log_softmax(lg[i, p - 1 : min(T, p + n_gen)].float(), -1).cpu()
            first.append(lp[0])
            rest.append(lp[1:])
        del lg
    return first, rest


def kl(a, b):
    return float((a.exp() * (a - b)).sum(-1).mean())


def main():
    model, tok = load()
    rng = random.Random(11)
    allr = sets()
    rows = [r for n in ("wj_adversarial_harmful", "wj_adversarial_benign", "dev_ho_h", "dev_ho_b", "harmful", "xstest_safe", "orbench_hard", "benign") for r in rng.sample([x for x in allr if x["set"] == n], 15)]
    resp = {rk(r): r["response"] for r in read_jsonl(RAW / "gen" / "base_dev.jsonl")}
    out = {"n_prompts": len(rows), "unit": UNIT, "note": "KL(reference || variant), natural log, mean over prompts (first token) or over prompts x reply positions (teacher-forced on the base reply)."}
    b_first, b_rest = logits_at(model, tok, rows, resp)
    def cmp(ref, var):
        return {"kl_first_token": float(np.mean([kl(a, b) for a, b in zip(ref[0], var[0])])), "kl_reply_positions": float(np.mean([kl(a, b) for a, b in zip(ref[1], var[1]) if len(a)])),
                "argmax_agree_first": float(np.mean([int(a.argmax() == b.argmax()) for a, b in zip(ref[0], var[0])]))}
    out["base_vs_itself"] = cmp((b_first, b_rest), logits_at(model, tok, rows, resp))
    CAP = len(sys.argv) > 1 and sys.argv[1] == "cap"
    for gain in ([1.0] if CAP else [float(a) for a in sys.argv[1:]] or [2.0]):
        kw = dict(gain=gain, cap_beta=4.0) if CAP else dict(gain=gain)
        for s in (0.0, 1.0, 1.25, 2.0, -1.0) if CAP else (0.0, 1.0, 2.0, -1.0):
            fw = logits_at(model, tok, rows, resp, Fold(model, s, "weights", **kw))
            fh = logits_at(model, tok, rows, resp, Fold(model, s, "hook", **kw))
            key = ("cap_" if CAP else f"gain{gain:g}_") + f"s{s:g}"
            out[key] = {"hook_vs_weights": cmp(fh, fw), "base_vs_weights": cmp((b_first, b_rest), fw)}
            log(f"{key}: hook_vs_weights {out[key]['hook_vs_weights']} | base_vs_weights {out[key]['base_vs_weights']}")
    (OUT / "fold").mkdir(exist_ok=True)
    (OUT / ("fold/parity_logits_cap.json" if CAP else "fold/parity_logits.json")).write_text(json.dumps(out, indent=1))
    sync()
    log("R4_FOLD_PARITY DONE")


if __name__ == "__main__":
    main()
