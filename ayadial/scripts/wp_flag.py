"""Weight programming, experiment A, kill test (forward passes only): a compiled position flag.
Idea (notes/weight-programming/REPORT.md, section 4): the position guard (m, q) is a ridge fit that has to recognise the reply-start token from the
layer-22 state, and it drifts with the context. Token identity is exact at the input. So write a flag there: add c*f to the embedding row of
<|START_RESPONSE|> (one row of the embedding matrix; the same thing can be done by one layer-0 MLP unit), with f a direction the model hardly uses at
layer 22, and let the layer-22 units read the flag through phi instead of m.
  python scripts/wp_flag.py test       K1: is the flag at layer 22 clean (high at every reply-start token, near 0 elsewhere)?  K2: does writing it
                                       change the model (mean next-token KL against the base at the reply-start token, bar 1e-3)?
Writes results/round6/wp/flag_test.json (numbers only) and cache_r6/wp_flag.npz (f, phi)."""
import sys
from r6lib import *
import r5_robust as RB

S, U, C, R, E = RB.S, RB.U, RB.C, RB.R, RB.E
FORMATS = ["default", "sys_bank", "sys_long", "sys_persona", "nosys", "mt_benign", "mt_refused", "inj_tail", "sys_car", "sys_law", "sys_de", "mt3", "mt_sys"]


def directions(k=6):
    """Candidate flag directions: the lowest-variance directions of the layer-22 MLP input over cached positions (round 5's held-out positions and
    round 6's format positions), excluding the direction LayerNorm removes."""
    xs = [np.load(CACHE / "gpos_eval.npz")["x"].astype(np.float32)]
    if (CACHE / "gpos6_fit.npz").exists():
        xs.append(np.load(CACHE / "gpos6_fit.npz")["x"].astype(np.float32))
    X = torch.tensor(np.concatenate(xs), dtype=torch.float64)
    X = X - X.mean(0, keepdim=True)
    Cv = X.T @ X / len(X)
    ev, V = torch.linalg.eigh(Cv)
    return ev.numpy(), V.numpy(), len(X)


@torch.no_grad()
def test():
    model, tok = load()
    rid = tok.convert_tokens_to_ids(R)
    L22 = model.model.layers[LYR]
    gamma = L22.input_layernorm.weight.detach().float().cpu().numpy()
    ev, V, npos = directions()
    # prompts: 16 from each development format suite (harmful and harmless), plus 4 prompts in each of nine other languages
    rng = random.Random(77)
    texts, tags = [], []
    for f_ in FORMATS:
        rows = read_jsonl(RAW / "suites" / f"rb_{f_}.jsonl")
        for r in rng.sample(rows, 16):
            texts.append(r["raw"]); tags.append(f_)
    for lg in ("zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"):
        for r in rng.sample([x for x in sets() if x["set"] == f"mj_{lg}_dev"], 4):
            texts.append(render(tok, r["prompt"])); tags.append("lang")
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    batches = [order[i : i + 16] for i in range(0, len(order), 16)]
    emb = model.model.embed_tokens
    store = {}
    hk = [L22.mlp.register_forward_pre_hook(lambda _m, a: store.__setitem__("x", a[0].float()))]
    norms = {}
    def mk(l):
        def hook(_m, a, kw):
            h_in = a[0] if a else kw["hidden_states"]
            norms.setdefault(l, []).append(float(h_in.float().pow(2).mean(-1).sqrt().mean()))
        return hook
    for l in (0, 1, 2, 4, 8, 12, 16, 20, 22):
        hk.append(model.model.layers[l].register_forward_pre_hook(mk(l), with_kwargs=True))
    def run_all(delta):
        """Forward every batch with delta (a d-vector, or None) added to the embedding row of the reply-start token. Returns per-prompt
        (x22 at all valid positions, ids, last-position log-probs)."""
        row0 = emb.weight.data[rid].clone()
        if delta is not None:
            emb.weight.data[rid] = row0 + delta.to(row0.dtype)
        out = {}
        try:
            for b in batches:
                e = tok([texts[i] for i in b], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN, add_special_tokens=False).to(DEV)
                h = model.model(**e, use_cache=False).last_hidden_state
                lp = torch.log_softmax(model.lm_head(h[:, -1]).float(), -1)
                for j, i in enumerate(b):
                    v = e["attention_mask"][j].bool()
                    out[i] = (store["x"][j][v].cpu(), e["input_ids"][j][v].cpu(), lp[j].cpu())
        finally:
            emb.weight.data[rid] = row0
        return out
    base = run_all(None)
    prof = {int(l): round(float(np.mean(v)), 4) for l, v in sorted(norms.items())}
    for h_ in hk[1:]:
        h_.remove()
    row_rms = float(emb.weight.data[rid].float().pow(2).mean().sqrt())
    res = {"n_prompts": len(texts), "n_positions_for_covariance": int(npos), "residual_rms_by_layer_input": prof, "embedding_row_rms_of_reply_start_token": round(row_rms, 5),
           "smallest_variances_of_layer22_mlp_input": [float(x) for x in ev[:8]], "median_variance": float(np.median(ev)), "candidates": []}
    best = None
    for k in range(1, 5):  # eigenvector 0 is the direction LayerNorm removes (variance ~0); try the next four
        u = V[:, k]
        f = gamma * u
        f = f - f.mean()
        f = f / np.linalg.norm(f)
        phi0 = torch.tensor(u, dtype=torch.float32)
        noise = float(np.sqrt(max(ev[k], 0)))
        for c in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
            amp = c * row_rms  # the flag's norm is c times the norm of the reply-start token's own embedding row
            got = run_all(torch.tensor(f * amp * np.sqrt(2048), dtype=torch.float32, device=emb.weight.device))
            mark, other, early, kl = [], [], [], []
            for i in got:
                x, ids, lp = got[i]
                raw = x @ phi0  # the flag as layer 22 reads it
                is_r = ids == rid
                mark.append(float(raw[-1]))
                early += raw[:-1][is_r[:-1]].tolist()
                other += raw[:-1][~is_r[:-1]].tolist()
                kl.append(float((base[i][2].exp() * (base[i][2] - lp)).sum()))
            mark, other, early = np.array(mark), np.array(other), np.array(early)
            med = float(np.median(mark))
            e_ = {"eigen_index": k, "c_relative_to_embedding_row_norm": c, "natural_std_along_phi": round(noise, 4), "flag_at_true_marker_min_median_max": [round(float(mark.min()), 3), round(med, 3), round(float(mark.max()), 3)],
                  "flag_elsewhere_p01_p50_p99_absmax": [round(float(x), 3) for x in np.percentile(other, [1, 50, 99])] + [round(float(np.abs(other).max()), 3)],
                  "flag_at_other_reply_start_tokens_min_median": [round(float(early.min()), 3), round(float(np.median(early)), 3)] if len(early) else None,
                  "scaled_marker_min": round(float(mark.min() / med), 3) if med else None, "scaled_elsewhere_absmax": round(float(np.abs(other).max() / abs(med)), 3) if med else None,
                  "kl_next_token_mean_max": [float(np.mean(kl)), float(np.max(kl))], "argmax_same_pct": round(100 * float(np.mean([int(base[i][2].argmax() == got[i][2].argmax()) for i in got])), 1),
                  "by_format_marker_min": {t: round(float(min(m_ for m_, tg in zip(mark, [tags[i] for i in got]) if tg == t)), 3) for t in sorted(set(tags))}}
            e_["K1"] = bool(med > 0 and e_["scaled_marker_min"] >= 0.9 and e_["scaled_elsewhere_absmax"] <= 0.1)
            e_["K2"] = bool(e_["kl_next_token_mean_max"][0] <= 1e-3)
            res["candidates"].append(e_)
            log(f"k={k} c={c}: marker {e_['flag_at_true_marker_min_median_max']} elsewhere absmax {e_['flag_elsewhere_p01_p50_p99_absmax'][3]} scaled min {e_['scaled_marker_min']} / max elsewhere {e_['scaled_elsewhere_absmax']} | KL {e_['kl_next_token_mean_max'][0]:.2e} same token {e_['argmax_same_pct']} | K1 {e_['K1']} K2 {e_['K2']}")
            if e_["K1"] and e_["K2"] and (best is None or e_["kl_next_token_mean_max"][0] < best[0]["kl_next_token_mean_max"][0]):
                best = (e_, f * amp * np.sqrt(2048), u, med)
    res["passes_K1_and_K2"] = best is not None
    if best is not None:
        res["chosen"] = {k_: best[0][k_] for k_ in ("eigen_index", "c_relative_to_embedding_row_norm", "kl_next_token_mean_max", "scaled_marker_min", "scaled_elsewhere_absmax")}
        np.savez(CACHE / "wp_flag.npz", delta=best[1].astype(np.float32), phi=(best[2] / best[3]).astype(np.float32), median=np.float64(best[3]))
    (OUT / "wp").mkdir(exist_ok=True)
    (OUT / "wp" / "flag_test.json").write_text(json.dumps(res, indent=1))
    sync()
    log(f"WP_FLAG_TEST DONE passes={res['passes_K1_and_K2']}")


if __name__ == "__main__":
    test()
