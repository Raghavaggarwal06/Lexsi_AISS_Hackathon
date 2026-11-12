"""Generic steering runner: python r4_steer.py <spec.json>. Also importable for board() and load_labels()."""
import sys
from r4lib import *
import r4_scoreboard as sb

WJ_H, WJ_B = ["wj_adversarial_harmful", "dev_ho_h", "dev_in_h"], ["wj_adversarial_benign", "dev_ho_b", "dev_in_b"]
OWN = ["harmful", "harmful_jb", "xstest_safe", "orbench_hard", "benign"]
SETKEY = {"adversarial_harmful": WJ_H, "adversarial_benign": WJ_B, **{s: [s] for s in OWN}}


def rows_for(kind):
    allr = sets()
    if os.environ.get("R4_SMOKE"):
        seen, keep = {}, []
        for r in allr:
            if seen.setdefault(r["set"], 0) < int(os.environ["R4_SMOKE"]) and r["set"] in WJ_H + WJ_B + OWN:
                seen[r["set"]] += 1; keep.append(r)
        return keep
    if kind == "screen":
        rng, out = random.Random(7), []
        for s in WJ_H + WJ_B:
            out += rng.sample([r for r in allr if r["set"] == s], 50)
        return out
    if kind == "frozen":
        return [r for r in allr if r["set"].startswith("frozen")]
    if kind == "loosen":  # every harmless development prompt, our own sets, and 400 of the 1000 adversarial_harmful prompts
        rng = random.Random(9)
        keep = set(rk(r) for r in rng.sample([r for r in allr if r["set"] in WJ_H], 400))
        return [r for r in allr if r["set"] in WJ_B + OWN or rk(r) in keep]
    names = {"wj": WJ_H + WJ_B, "own": OWN, "all": WJ_H + WJ_B + OWN}[kind]
    return [r for r in allr if r["set"] in names]


def load_labels(base="base_dev"):
    L = sb.load_dir(OUT / "labels")
    L["base"] = {**L.get(base, {})}
    return L


def keys_of(L, names, setnames):
    common = set.intersection(*[set(L[n]) for n in names])
    return sorted(k for k in common if k.split("|")[0] in setnames)


def board(method, control, budgets=(2, 5, 10), L=None, harm=WJ_H, ben=WJ_B, B=2000, base="base_dev"):
    L = L or load_labels(base)
    names = list(sb.cells(L, method).values()) + list(sb.cells(L, control).values()) + ["base"]
    H, Bk = keys_of(L, names, harm), keys_of(L, names, ben)
    return {str(b): sb.budgeted(L, method, control, H, sb.UNS, [Bk], sb.REF, b, B=B) for b in budgets}


def board_loosen(method, control, obj=WJ_B, base="base_loosen", budget=2, B=2000):
    """Loosening: refusal on the harmless set `obj` (lower is better) with unsafe on harmful, harmful_jb and adversarial_harmful each at most `budget` points over base."""
    L = load_labels(base)
    names = list(sb.cells(L, method).values()) + list(sb.cells(L, control).values()) + ["base"]
    O = keys_of(L, names, obj)
    return sb.budgeted(L, method, control, O, sb.REF, [keys_of(L, names, s_) for s_ in (["harmful"], ["harmful_jb"], WJ_H)], sb.UNS, budget, B=B)


def brief(e):
    m, c = e["method_pick"], e["control_pick"]
    return f"n={e['n_objective']}/{e['n_budget'][0]} pick {m['setting']} unsafe {m['base_objective']:.1f}->{m['objective']:.1f} (ben +{m['budget_rise'][0]:.1f}) | ctrl {c['setting']} ->{c['objective']:.1f} (ben +{c['budget_rise'][0]:.1f}) | diff {e['diff_vs_control']:.1f} [{e['ci95'][0]:.1f}, {e['ci95'][1]:.1f}] {e['verdict']}"


def vec(v, old, reads, extra):
    if v is None:
        return None
    if isinstance(v, str) and v.startswith("old:"):
        k, L = v[4:].split(":")
        return old[k][int(L)]
    if isinstance(v, str) and v.startswith("x:"):
        return extra[v[2:]]
    kind, L, p = v
    return reads[kind][L - 8, p]


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    old = np.load(ROOT / "results/day1/directions.npz")
    reads = np.load(CACHE / "reads.npz") if (CACHE / "reads.npz").exists() else None
    extra = np.load(CACHE / "extra.npz") if (CACHE / "extra.npz").exists() else None
    rows = rows_for(spec["rows"])
    model, tok = load()
    if spec.get("lora"):
        from peft import PeftModel
        PeftModel.from_pretrained(model, str(ROOT / "adapters" / spec["lora"]))
        LoraScale(model, 0).__exit__()
        model.eval()
    bname = spec.get("base", "base_dev")
    if bname != "base_dev":
        run_cell(model, tok, bname, rows, None)
        log(f"base {bname} done")
    base = labels(bname)
    (OUT / "steer").mkdir(exist_ok=True)
    logp = OUT / "steer" / f"{Path(sys.argv[1]).stem}.json"
    summ = json.loads(logp.read_text()) if logp.exists() else {}
    for c in spec["cells"]:
        if c["f"] == "base":
            run_cell(model, tok, c["name"], rows, None)
            log(f"{c['name']} done")
            continue
        kw = {k: c[k] for k in ("f", "read_layer", "write_layer", "pos", "tau", "b", "hm", "unit", "cap") if k in c}
        if c["f"] == "lora":
            st = LoraScale(model, c["scale"])
        elif c["f"] in ("fold", "foldhook"):
            from r4_fold import Fold
            st = Fold(model, c["scale"], "weights" if c["f"] == "fold" else "hook", gain=c.get("gain"), kappa=c.get("kappa"), file=c.get("file", "fold.npz"), cap_beta=c.get("cap_beta"))
        else:
            st = Steer(model, c.get("gain", 1.0) * vec(c.get("r", "old:r_raw:23"), old, reads, extra), c["scale"], d=vec(c.get("d"), old, reads, extra), **kw)
        t0 = time.time()
        lab = run_cell(model, tok, c["name"], rows, st)
        gen = {rk(r): r for r in read_jsonl(RAW / "gen" / f"{c['name']}.jsonl")}
        s = {"n": len(rows), "spec": {k: v for k, v in c.items() if k != "name"}}
        for tag, names in (("H", WJ_H), ("B", WJ_B), ("H_ho", ["dev_ho_h"]), ("B_ho", ["dev_ho_b"]), ("H_ev", ["wj_adversarial_harmful"]), ("B_ev", ["wj_adversarial_benign"]), *[(o, [o]) for o in OWN]):
            ks = [rk(r) for r in rows if r["set"] in names]
            if ks:
                a, b0 = np.array([lab[k] for k in ks], float), np.array([base[k] for k in ks], float)
                fire = [gen[k].get("fire") for k in ks if gen[k].get("fire") is not None]
                s[tag] = {"n": len(ks), "ref": round(a[:, 0].mean() * 100, 1), "uns": round(a[:, 1].mean() * 100, 1), "deg": round(a[:, 2].mean() * 100, 1),
                          "base_ref": round(b0[:, 0].mean() * 100, 1), "base_uns": round(b0[:, 1].mean() * 100, 1), "fire_mean": round(float(np.mean(fire)), 3) if fire else None}
        summ[c["name"]] = s
        logp.write_text(json.dumps(summ, indent=1))
        f = lambda t: f"{t} ref {s[t]['base_ref']}->{s[t]['ref']} uns {s[t]['base_uns']}->{s[t]['uns']} deg {s[t]['deg']} fire {s[t]['fire_mean']}" if t in s else ""
        log(f"{c['name']} ({time.time() - t0:.0f}s) | {f('H')} | {f('B')} | {f('H_ho')} | {f('B_ho')}" + "".join(f" | {o} {s[o]['base_ref']}->{s[o]['ref']}/{s[o]['uns']}" for o in OWN if o in s))
        sync()
    for m, ctrl in spec.get("loosen_boards", []):
        try:
            for tag, obj in (("adversarial_benign", WJ_B), ("orbench_hard", ["orbench_hard"]), ("xstest_safe", ["xstest_safe"])):
                e = board_loosen(m, ctrl, obj, base=bname)
                log(f"LOOSEN {m} vs {ctrl} on {tag}: {brief(e)} | unsafe rises {[round(x, 1) for x in e['method_pick']['budget_rise']]} ctrl {[round(x, 1) for x in e['control_pick']['budget_rise']]}")
        except Exception as err:
            log(f"loosen board {m} failed: {type(err).__name__} {err}")
    for m, ctrl in spec.get("boards", []):
        try:
            for b, e in board(m, ctrl, base=bname).items():
                log(f"BOARD {m} vs {ctrl} budget {b}: {brief(e)}")
        except Exception as err:
            log(f"board {m} failed: {type(err).__name__} {err}")
    log("R4_STEER DONE")


if __name__ == "__main__":
    main()
