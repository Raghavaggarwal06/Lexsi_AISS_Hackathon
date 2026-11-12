"""Wang et al. (arXiv 2410.03415) false-refusal vector ablation, re-implemented per notes/methods_spec.md (b).
[ours] deviations, all forced by data on hand: pseudo-harmful train = the orbench_hard_fit prompts the base refuses (judge label,
not the first-token score), about 40 instead of 128; candidates restricted to block inputs 8..27 (cached) x the 4 template-tail tokens.
  python r4_wang.py <rows: loosen|screen> [lambdas, default 1.0,0.8]"""
import sys
from r4lib import *
import r4_steer as rs
rows_kind = sys.argv[1]
lams = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1.0,0.8").split(",")]
model, tok = load()
A = lambda n: np.load(CACHE / f"acts_{n}.npy", mmap_mode="r")
IDS = lambda n: json.loads((CACHE / f"ids_{n}.json").read_text())["ids"]
fitlab, devlab = labels("base_fit"), labels("base_dev")
allr = {rk(r): r for r in sets()}
rng = random.Random(3)
ref_idx = [i for i, x in enumerate(IDS("orbench_hard_fit")) if fitlab[f"orbench_hard_fit|{x}"][0] == 1]
rng.shuffle(ref_idx)
val_idx, tr_idx = ref_idx[:16], ref_idx[16:]
ben_ids = IDS("benign_fit")
ben_val = rng.sample(range(len(ben_ids)), 16)
val_ph = [allr[f"orbench_hard_fit|{IDS('orbench_hard_fit')[i]}"] for i in val_idx]
val_b = [allr[f"benign_fit|{ben_ids[i]}"] for i in ben_val]

# refusal-opener token set R, from base replies on development data
first = {}
for r in read_jsonl(RAW / "gen" / "base_dev.jsonl"):
    t = tok(r["response"], add_special_tokens=False)["input_ids"][:1]
    if t and devlab[rk(r)][2] == 0:
        c = first.setdefault(t[0], [0, 0]); c[1 - devlab[rk(r)][0]] += 1
R = sorted(t for t, (ref, ans) in first.items() if ref >= 5 and ref / (ref + ans) >= 0.9)
cover = sum(first[t][0] for t in R) / max(1, sum(v[0] for v in first.values()))
log(f"refusal opener set: {len(R)} tokens, covers {cover:.3f} of refused replies")
Rt = torch.tensor(R, device=DEV)


class Ablate:
    def __init__(self, w):
        p = next(model.parameters())
        self.w = None if w is None else torch.as_tensor(w, device=p.device, dtype=p.dtype)
        self.fire, self.handles = [], []

    def proj(self, h):
        return h - (h @ self.w).unsqueeze(-1) * self.w

    def pre(self, _m, args, kwargs):
        return (self.proj(args[0]), *args[1:]), kwargs

    def post(self, _m, _i, out):
        return (self.proj(out[0]), *out[1:]) if isinstance(out, tuple) else self.proj(out)

    def __enter__(self):
        if self.w is not None:
            for L in model.model.layers:
                self.handles += [L.register_forward_pre_hook(self.pre, with_kwargs=True), L.self_attn.register_forward_hook(self.post), L.mlp.register_forward_hook(self.post)]
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def first_logp(rows, w=None):
    out = []
    for s in range(0, len(rows), 32):
        e = enc(tok, [r["prompt"] for r in rows[s : s + 32]])
        with Ablate(w):
            lg = model(**e, use_cache=False).logits[:, -1].float()
        out.append(torch.log_softmax(lg, -1))
    return torch.cat(out)


def refusal_score(lp):
    pr = lp[:, Rt].exp().sum(-1).clamp(1e-9, 1 - 1e-9)
    return (pr.log() - (1 - pr).log()).mean().item()


base_ph, base_b = first_logp(val_ph), first_logp(val_b)
res = {"n_train_pseudo_harmful_refused": len(tr_idx), "n_val": 16, "opener_tokens": len(R), "opener_coverage": cover, "base_val_refusal_score": refusal_score(base_ph), "selected": {}}
for lam in lams:
    best, table = None, []
    for L in range(8, 28):
        for p in range(1, 5):
            ph = np.asarray(A("orbench_hard_fit")[tr_idx, L - 8, p], np.float32).mean(0)
            hl = np.asarray(A("benign_fit")[:, L - 8, p], np.float32).mean(0)
            hf = np.asarray(A("harmful_fit")[:, L - 8, p], np.float32).mean(0)
            w, v = ph - hl, hf - hl
            w = w - lam * (w @ v) / (v @ v) * v
            w = w / np.linalg.norm(w)
            sc = refusal_score(first_logp(val_ph, w))
            lb = first_logp(val_b, w)
            klv = float((base_b.exp() * (base_b - lb)).sum(-1).mean())
            table.append([L, p, sc, klv])
            if klv < 0.2 and (best is None or sc < best[0]):
                best = (sc, L, p, w, klv)
    res["selected"][str(lam)] = {"layer": best[1], "position": ["last_instruction", "END_OF_TURN", "START_OF_TURN", "CHATBOT", "START_RESPONSE"][best[2]], "val_refusal_score": best[0], "kl_harmless_first_token": best[4],
                                 "candidates": len(table), "candidates_passing_kl": sum(t[3] < 0.2 for t in table)}
    log(f"lambda {lam}: {res['selected'][str(lam)]}")
    rows = rs.rows_for(rows_kind)
    bname = {"loosen": "base_loosen", "screen": "selfcheck@0"}[rows_kind]
    base = labels(bname)
    lab = run_cell(model, tok, f"wang@{lam:g}", rows, Ablate(best[3]))
    s = {}
    for tag, names in (("H", rs.WJ_H), ("B", rs.WJ_B), *[(o, [o]) for o in rs.OWN]):
        ks = [rk(r) for r in rows if r["set"] in names]
        if ks:
            a, b0 = np.array([lab[k] for k in ks], float), np.array([base[k] for k in ks], float)
            s[tag] = {"n": len(ks), "ref": round(a[:, 0].mean() * 100, 1), "uns": round(a[:, 1].mean() * 100, 1), "deg": round(a[:, 2].mean() * 100, 1), "base_ref": round(b0[:, 0].mean() * 100, 1), "base_uns": round(b0[:, 1].mean() * 100, 1)}
    res["selected"][str(lam)]["rates"] = s
    log(f"wang@{lam:g} | " + " | ".join(f"{t} ref {v['base_ref']}->{v['ref']} uns {v['base_uns']}->{v['uns']} deg {v['deg']}" for t, v in s.items()))
    (OUT / "steer").mkdir(exist_ok=True)
    (OUT / "steer" / f"wang_{rows_kind}.json").write_text(json.dumps(res, indent=1))
    sync()
if rows_kind == "loosen":
    for tag, obj in (("adversarial_benign", rs.WJ_B), ("orbench_hard", ["orbench_hard"]), ("xstest_safe", ["xstest_safe"])):
        try:
            e = rs.board_loosen("wang", "ungated", obj)
            log(f"LOOSEN wang vs ungated on {tag}: {rs.brief(e)} | unsafe rises {[round(x, 1) for x in e['method_pick']['budget_rise']]}")
        except Exception as err:
            log(f"board failed {type(err).__name__} {err}")
log("R4_WANG DONE")
