"""Refit the position guard (m, q) with positions from the new distributions, by v1's recipe (ridge, marker rows weighted 30). No training loop.
Why: tonight's lab shows that q, which should be 0 at the reply-start token, sits at +0.06 to +0.10 there on in-the-wild templates, other
languages and random-search prompts. The unit fires when s*z > theta0*m + (kappa/alpha)*q, so that offset raises the threshold by 0.2 to 0.3 in z.
v1's guard was fit on single-turn English prompts from WildJailbreak and our own sets only.
  python scripts/r5_guard.py capture        forward passes: x at sampled positions (system preamble, user text, the 5 tail tokens, the base reply)
  python scripts/r5_guard.py fit <name>     -> cache_r5/vec_guard_<name>.npz, results/round5/guard/<name>.json
Fit sample: v1's kind of prompts plus the 'a' halves of the in-the-wild, MultiJail and Aya development slices, GCG and PAIR pool artifacts and suffix
transplants. Held-out sample for the report: round 4 development prompts, the 'b' halves, the held-out JailbreakBench behaviours, PHTest."""
import collections, sys
from r5lib import *
from r5_lab import half

LG = ["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]
KN = {0: "preamble", 1: "user", 2: "last_instruction", 3: "END_OF_TURN", 4: "START_OF_TURN", 5: "CHATBOT", 6: "START_RESPONSE(marker)", 7: "generated"}
FIT_PLAN = [("trfit_h", None, 150), ("trfit_b", None, 150), ("harmful_fit", None, 60), ("benign_fit", None, 60), ("xstest_safe_fit", None, 40), ("orbench_hard_fit", None, 40),
            ("itw_h_dev", "a", 200), ("itw_b_dev", "a", 200), ("jbb_GCG_pool", None, 40), ("jbb_PAIR_pool", None, 40), ("aug_gcg_h_fit", None, 40), ("aug_gcg_b_fit", None, 40)] + \
           [(f"mj_{l}_dev", "a", 15) for l in LG] + [(f"aya_{l}_dev", "a", 30) for l in LG]
EVAL_PLAN = [(n, None, 60) for n in ("wj_adversarial_harmful", "wj_adversarial_benign", "dev_ho_h", "dev_ho_b", "harmful", "harmful_jb", "benign", "orbench_hard", "xstest_safe")] + \
            [("itw_h_dev", "b", 200), ("itw_b_dev", "b", 200), ("pht_harmless_dev", None, 40), ("pht_contro_dev", None, 20), ("sfx_h_dev", None, 30), ("sfx_b_dev", None, 30)] + \
            [(f"jbb_{m_}_dev", None, 25) for m_ in ("GCG", "DSN", "PAIR", "JBC", "RS")] + [(f"mj_{l}_dev", "b", 15) for l in LG] + [(f"aya_{l}_dev", "b", 30) for l in LG]


def family(s):
    if s.startswith("itw_"):
        return "in-the-wild templates"
    if s.startswith("mj_") or s.startswith("aya_"):
        return "English (MultiJail, Aya)" if "_en_" in s else "9 other languages (MultiJail, Aya)"
    if s.startswith("jbb_RS") or s.startswith("jbb_JBC"):
        return "JailbreakBench JBC, random search"
    if s.startswith("jbb_") or s.startswith("sfx_") or s.startswith("aug_"):
        return "suffix attacks and transplants, PAIR"
    return "round 4 kind (WildJailbreak, own sets, PHTest)"


def pick(plan, seed):
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in sets():
        by[r["set"]].append(r)
    rows = []
    for n, hf, k in plan:
        pool = [r for r in by[n] if hf is None or half(n, r["id"]) == hf]
        rows += rng.sample(pool, min(k, len(pool)))
    return rows


@torch.no_grad()
def capture():
    model, tok = load()
    store = {}
    hk = model.model.layers[LYR].mlp.register_forward_pre_hook(lambda _m, a: store.__setitem__("x", a[0]))
    resp = {}
    for f in (R4RAW / "gen" / "base_fit.jsonl", RAW / "gen" / "base_wjx.jsonl", RAW / "gen" / "base_own.jsonl", RAW / "gen" / "base_dev5.jsonl"):
        resp.update({rk(r): r["response"] for r in read_jsonl(f)})
    K = len(tok(render(tok, "x"), add_special_tokens=False)["input_ids"]) - 5 - len(tok("x", add_special_tokens=False)["input_ids"])
    for tag, plan, seed in (("fit", FIT_PLAN, 5), ("eval", EVAL_PLAN, 6)):
        rows = pick(plan, seed)
        rng = random.Random(seed + 10)
        xs, meta = [], []
        for j, r in enumerate(rows):
            p_ids = tok(render(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
            g_ids = tok(resp[rk(r)], add_special_tokens=False)["input_ids"][:96] if rk(r) in resp else []
            P = len(p_ids)
            if P > MAXLEN or P - 5 - K < 1:
                continue
            model.model(input_ids=torch.tensor([p_ids + g_ids], device=DEV), use_cache=False)
            pos = [(i, 0) for i in rng.sample(range(K), 20)] + [(i, 1) for i in rng.sample(range(K, P - 5), min(24, P - 5 - K))] + [(P - 5 + i, 2 + i) for i in range(5)]
            pos += [(i, 7) for i in rng.sample(range(P, P + len(g_ids)), min(30, len(g_ids)))]
            ii = torch.tensor([i for i, _ in pos], device=DEV)
            xs.append(store["x"][0, ii].float().cpu().numpy().astype(np.float16))
            meta += [(j, k, i) for i, k in pos]
        np.savez(CACHE / f"gpos_{tag}.npz", x=np.concatenate(xs), meta=np.array(meta, np.int32), sets=np.array([r["set"] for r in rows]), ids=np.array([r["id"] for r in rows]))
        log(f"positions {tag}: {len(rows)} prompts, {len(meta)} positions")
    hk.remove()
    log("R5_GUARD_CAPTURE DONE")


def ridge(X, y, lam):
    X, y = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    G = X.T @ X
    G = G + lam * G.diagonal().mean() * torch.eye(G.shape[0], dtype=torch.float64)
    return torch.linalg.solve(G, X.T @ y).numpy()


def fit(name, weight=30.0):
    torch.set_num_threads(8)
    pf, pe = np.load(CACHE / "gpos_fit.npz"), np.load(CACHE / "gpos_eval.npz")
    xf, kf = pf["x"].astype(np.float64), pf["meta"][:, 1]
    xe, ke, je = pe["x"].astype(np.float64), pe["meta"][:, 1], pe["meta"][:, 0]
    fam_e = np.array([family(s) for s in pe["sets"]])[je]
    # extra marker rows, as v1 did: the marker x of every 4th prompt of v1's fit sets, and of every prompt of the new fit sets
    extra = []
    for n in ("trfit_h", "trfit_b", "harmful_fit", "benign_fit", "xstest_safe_fit", "orbench_hard_fit", "xstest_unsafe_fit"):
        extra.append(np.load(CACHE / f"act_{n}.npz")["x"][::4].astype(np.float64))
    for n, hf, _ in FIT_PLAN[6:]:
        P = np.load(CACHE / f"act_{n}.npz")
        keep = np.array([hf is None or half(n, str(i)) == hf for i in P["ids"]])
        extra.append(P["x"][keep].astype(np.float64))
    extra = np.concatenate(extra)
    Xp = np.concatenate([xf, extra])
    ym = np.concatenate([(kf == 6).astype(np.float64), np.ones(len(extra))])
    sw = np.sqrt(np.where(ym > 0.5, weight, 1.0))
    V1 = vec("v1")
    best = None
    for lam in (1e-4, 1e-3, 1e-2):
        m = ridge(Xp * sw[:, None], ym * sw, lam)
        q = ridge(Xp * sw[:, None], (1 - ym) * sw, lam)
        sd = float((xe[ke == 6] @ q).std())
        if best is None or sd < best[1]:
            best = (lam, sd, m, q)
    lam, _, m, q = best
    def describe(m_, q_):
        o = {"q_at_marker_mean_sd": [round(float((xe[ke == 6] @ q_).mean()), 4), round(float((xe[ke == 6] @ q_).std()), 4)], "m_at_marker_min_p1_p50": [round(float(v), 3) for v in np.percentile(xe[ke == 6] @ m_, [0, 1, 50])],
             "m_elsewhere_max": round(float((xe[ke != 6] @ m_).max()), 3), "q_elsewhere_min": round(float((xe[ke != 6] @ q_).min()), 3),
             "margin_m": round(float((xe[ke == 6] @ m_).min() - (xe[ke != 6] @ m_).max()), 3), "margin_q": round(float((xe[ke != 6] @ q_).min() - (xe[ke == 6] @ q_).max()), 3),
             "by_family_at_marker": {}, "m_by_kind_p0_p50_p100": {KN[k]: [round(float(v), 3) for v in np.percentile(xe[ke == k] @ m_, [0, 50, 100])] for k in sorted(set(ke.tolist()))},
             "q_by_kind_p0_p50_p100": {KN[k]: [round(float(v), 3) for v in np.percentile(xe[ke == k] @ q_, [0, 50, 100])] for k in sorted(set(ke.tolist()))}}
        for f_ in sorted(set(fam_e.tolist())):
            mk = (ke == 6) & (fam_e == f_)
            o["by_family_at_marker"][f_] = {"n": int(mk.sum()), "m_mean": round(float((xe[mk] @ m_).mean()), 3), "m_min": round(float((xe[mk] @ m_).min()), 3), "q_mean": round(float((xe[mk] @ q_).mean()), 3), "q_sd": round(float((xe[mk] @ q_).std()), 3),
                                              "threshold_shift_in_z": round(float((V1["theta0"] * ((xe[mk] @ m_) - 1) + KAPPA / ALPHA * (xe[mk] @ q_)).mean()), 3)}
        return o
    out = {"name": name, "marker_weight": weight, "lambda": lam, "n_fit_positions": int(len(xf)), "n_extra_marker_rows": int(len(extra)), "n_eval_positions": int(len(xe)), "n_eval_prompts": int(len(pe["ids"])),
           "cos_m_with_v1": float(m @ V1["m"] / (np.linalg.norm(m) * np.linalg.norm(V1["m"]))), "cos_q_with_v1": float(q @ V1["q"] / (np.linalg.norm(q) * np.linalg.norm(V1["q"]))),
           "new_guard_on_held_out_positions": describe(m, q), "v1_guard_on_the_same_positions": describe(V1["m"].astype(np.float64), V1["q"].astype(np.float64)),
           "note": "threshold_shift_in_z = theta0*(m-1) + (kappa/alpha)*q at the reply-start token: how far the unit's effective threshold sits from theta0 (0 is ideal)"}
    np.savez(CACHE / f"vec_guard_{name}.npz", m=m.astype(np.float32), q=q.astype(np.float32))
    (OUT / "guard").mkdir(exist_ok=True)
    (OUT / "guard" / f"{name}.json").write_text(json.dumps(out, indent=1))
    for tag in ("new_guard_on_held_out_positions", "v1_guard_on_the_same_positions"):
        o = out[tag]
        print(tag, {k: v for k, v in o.items() if not k.startswith("by_") and not k.endswith("p100")})
        for f_, v in o["by_family_at_marker"].items():
            print(f"   {f_:48s} {v}")
        print("   m by kind", o["m_by_kind_p0_p50_p100"])
        print("   q by kind", o["q_by_kind_p0_p50_p100"])
    sync()
    print("R5_GUARD_FIT DONE")


def writes(vname, theta_lax=None):
    """What the units of a vector file write away from the reply-start token, on the held-out positions (never used to fit a guard).
    python scripts/r5_guard.py writes <vec> [theta_lax]     -> results/round5/guard/writes_<vec>.json   (push in units of r)"""
    pe = np.load(CACHE / "gpos_eval.npz")
    xe, ke = pe["x"].astype(np.float64), pe["meta"][:, 1]
    V = vec(vname)
    m, q, w, th = V["m"].astype(np.float64), V["q"].astype(np.float64), V["w"].astype(np.float64), V["theta0"]
    silu = lambda v: v / (1 + np.exp(-np.clip(v, -60, 60)))
    xm, xq, xw = xe @ m, xe @ q, xe @ w
    off, on = ke != 6, ke == 6
    out = {"vec": vname, "theta0": th, "n_positions_elsewhere": int(off.sum()), "n_markers": int(on.sum()), "strict": {}, "lax": {}}
    def stats(p):
        a = np.abs(p[off])
        worst = int(ke[off][a.argmax()])
        return {"elsewhere_max": round(float(a.max()), 4), "elsewhere_p999": round(float(np.quantile(a, 0.999)), 4), "elsewhere_share_above_0.05": round(float((a > 0.05).mean()), 5), "elsewhere_share_above_0.5": round(float((a > 0.5).mean()), 6),
                "worst_kind": KN[worst], "at_marker_mean": round(float(p[on].mean()), 3), "at_marker_share_above_0.5": round(float((np.abs(p[on]) > 0.5).mean()), 3)}
    for s_ in (1.0, 1.25, 2.0, 3.0):
        g = -ALPHA * th * xm - KAPPA * xq + s_ * ALPHA * xw
        out["strict"][f"s={s_:g}"] = stats((silu(g) - silu(g - ALPHA / BETA * xm)) * xm * BETA / ALPHA)
    if theta_lax is not None:
        g = LAX_ALPHA * (theta_lax * xm - xw) - KAPPA * xq
        out["lax"] = {"theta": theta_lax, "t=1": stats((silu(g) - silu(g - LAX_ALPHA / LAX_BETA * xm)) * xm * LAX_BETA / LAX_ALPHA)}
    (OUT / "guard").mkdir(exist_ok=True)
    (OUT / "guard" / f"writes_{vname}.json").write_text(json.dumps(out, indent=1))
    print(vname, json.dumps(out["strict"]), json.dumps(out["lax"]))
    sync()


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        capture()
    elif sys.argv[1] == "writes":
        writes(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        fit(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 30.0)
