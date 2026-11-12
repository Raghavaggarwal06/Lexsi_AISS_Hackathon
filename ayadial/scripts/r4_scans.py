"""SCANS (Cao et al., arXiv 2408.11491) re-implemented from the paper and the authors' code, per notes/methods_spec.md (a).
[ours] mappings: band = decoder layers 11..21 of 36; classifier layers 11..35; alpha is c times the norm of the raw
difference-of-means vector of each layer (the paper's alpha = 3.5 on a unit vector is tied to Llama-2 residual norms).
  python r4_scans.py <rows: screen|wj|own> <c1,c2,...> [T: plain|wj]"""
import sys
from r4lib import *
import r4_steer as rs
BAND, CLS = list(range(11, 22)), list(range(11, 36))
rows_kind, cs = sys.argv[1], [float(c) for c in sys.argv[2].split(",")]
tmode = sys.argv[3] if len(sys.argv) > 3 else "plain"
model, tok = load()
allr = sets()
rng = random.Random(0)
anchors_h = rng.sample([r for r in allr if r["set"] == "harmful_fit"], 64)
anchors_b = rng.sample([r for r in allr if r["set"] == "benign_fit"], 64)
SURE = tok("Sure", add_special_tokens=False)["input_ids"]


@torch.no_grad()
def states(rows, batch=32):
    """marker state and transition (marker minus the appended 'Sure' token) for decoder layers 0..35: [n, 36, 2048] each."""
    P, Tr = [], []
    for s in range(0, len(rows), batch):
        e = enc(tok, [r["prompt"] for r in rows[s : s + batch]])
        ids = torch.cat([e["input_ids"], torch.tensor([SURE] * len(e["input_ids"]), device=DEV)], 1)
        att = torch.cat([e["attention_mask"], torch.ones((len(ids), len(SURE)), dtype=e["attention_mask"].dtype, device=DEV)], 1)
        hs = model.model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False).hidden_states
        m = torch.stack([hs[l + 1][:, -1 - len(SURE)] for l in range(36)], 1).float()
        e_ = torch.stack([hs[l + 1][:, -1] for l in range(36)], 1).float()
        P.append(m.cpu()); Tr.append((m - e_).cpu())
    return torch.cat(P), torch.cat(Tr)


cache = CACHE / "scans_fit.pt"
if cache.exists():
    S = torch.load(cache)
else:
    Ph, Th = states(anchors_h); Pb, Tb = states(anchors_b)
    raw = Ph.mean(0) - Pb.mean(0)  # [36, 2048], refusal steering vector per layer (Eq. 1)
    S = {"raw_norm": raw.norm(dim=-1), "v": raw / raw.norm(dim=-1, keepdim=True), "d_harm": Th.mean(0)}
    torch.save(S, cache)


def score(rows):
    _, T = states(rows)
    cos = torch.nn.functional.cosine_similarity(T[:, CLS], S["d_harm"][CLS].unsqueeze(0), dim=-1)
    return cos.mean(1).numpy()


# threshold T: the paper fixes one per model; here it is the best balanced-accuracy cut on fit data only
tpath = CACHE / f"scans_T_{tmode}.json"
if tpath.exists():
    TINFO = json.loads(tpath.read_text())
else:
    if tmode == "plain":
        hs_, bs_ = [r for r in allr if r["set"] == "harmful_fit"], [r for r in allr if r["set"] in ("benign_fit", "xstest_safe_fit", "orbench_hard_fit")]
    else:
        r2 = random.Random(1)
        hs_, bs_ = r2.sample([r for r in allr if r["set"] == "trfit_h"], 400), r2.sample([r for r in allr if r["set"] == "trfit_b"], 400)
    sh_, sb_ = score(hs_), score(bs_)
    cand = np.unique(np.concatenate([sh_, sb_]))
    acc = [((sh_ >= t).mean() + (sb_ < t).mean()) / 2 for t in cand]
    TINFO = {"T": float(cand[int(np.argmax(acc))]), "balanced_acc_fit": float(max(acc)), "auroc_fit": auroc(sh_, sb_), "mode": tmode, "n": [len(sh_), len(sb_)]}
    tpath.write_text(json.dumps(TINFO))
log(f"SCANS T {TINFO}")


class Scans:
    def __init__(self, c):
        p = next(model.parameters())
        self.vec = {l: (c * S["raw_norm"][l] * S["v"][l]).to(p.device, p.dtype) for l in BAND}
        self.fire, self.sign, self.handles = [], None, []

    def before(self, rows):
        s = score(rows)
        self.sign = torch.tensor(np.where(s < TINFO["T"], -1.0, 1.0), device=DEV, dtype=next(model.parameters()).dtype)

    def hook(self, l):
        def f(_m, _i, out):
            h = out[0] if isinstance(out, tuple) else out
            add = self.sign.view(-1, 1, 1) * self.vec[l]
            if h.shape[1] > 1:  # prefill: the marker only; user tokens are not steered
                mask = torch.zeros((1, h.shape[1], 1), device=h.device, dtype=h.dtype); mask[:, -1] = 1
                add = add * mask
                if l == BAND[0]:
                    self.fire.append(self.sign.float().cpu())
            pushed = h + add
            return (pushed, *out[1:]) if isinstance(out, tuple) else pushed
        return f

    def __enter__(self):
        self.handles = [model.model.layers[l].register_forward_hook(self.hook(l)) for l in BAND]
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


rows = rs.rows_for(rows_kind)
bname = "base_dev" if os.environ.get("R4_SMOKE") else {"screen": "selfcheck@0", "wj": "base_wjx", "own": "base_own"}[rows_kind]
base = labels(bname)
(OUT / "steer").mkdir(exist_ok=True)
logp = OUT / "steer" / f"scans_{tmode}_{rows_kind}.json"
summ = json.loads(logp.read_text()) if logp.exists() else {"T": TINFO}
for c in cs:
    name = ("smoke_" if os.environ.get("R4_SMOKE") else "") + f"scans_T{tmode}@{c:g}"
    lab = run_cell(model, tok, name, rows, Scans(c))
    gen = {rk(r): r for r in read_jsonl(RAW / "gen" / f"{name}.jsonl")}
    s = {}
    for tag, names in (("H", rs.WJ_H), ("B", rs.WJ_B), *[(o, [o]) for o in rs.OWN]):
        ks = [rk(r) for r in rows if r["set"] in names]
        if ks:
            a, b0 = np.array([lab[k] for k in ks], float), np.array([base[k] for k in ks], float)
            s[tag] = {"n": len(ks), "ref": round(a[:, 0].mean() * 100, 1), "uns": round(a[:, 1].mean() * 100, 1), "deg": round(a[:, 2].mean() * 100, 1), "base_ref": round(b0[:, 0].mean() * 100, 1), "base_uns": round(b0[:, 1].mean() * 100, 1),
                      "steered_toward_refusal": round(float(np.mean([gen[k]["fire"] > 0 for k in ks])), 3)}
    summ[name] = s
    logp.write_text(json.dumps(summ, indent=1))
    log(f"{name} | " + " | ".join(f"{t} ref {v['base_ref']}->{v['ref']} uns {v['base_uns']}->{v['uns']} deg {v['deg']} +sign {v['steered_toward_refusal']}" for t, v in s.items()))
    sync()
try:
    for b, e in rs.board(f"scans_T{tmode}", "ungated", base=bname).items():
        log(f"BOARD scans_T{tmode} vs ungated budget {b}: {rs.brief(e)}")
except Exception as err:
    log(f"board failed: {type(err).__name__} {err}")
log("R4_SCANS DONE")
