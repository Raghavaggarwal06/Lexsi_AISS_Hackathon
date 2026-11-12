"""Round 6: refit the READ with deployment formats in its fit (v5), by v3g's recipe.
The positions lab of the guard refit (journal J4) shows that with a long developer preamble the harm read itself falls (mean z of harmful requests
0.98 in the default format, 0.75 and 0.67 with the two long preambles) while the threshold stays where the default format put it. So the read needs
the formats, not only the guard.
  python scripts/r6_read.py capture     marker activations of the fit-format suites (gf_<fmt>) -> cache_r6/act_gf_h.npz, act_gf_b.npz
  python scripts/r6_read.py lab         v5 = v3g's fit sets + those rows; saved with the format-aware guard g3 as cache_r6/vec_v5g.npz; results/round6/lab/v5g.json
Recipe unchanged (round 5's r5_lab.py): shrunk LDA (0.1) on h, z scaled by the default-format WildJailbreak fit rows, theta0 = 98th percentile of z on
trfit_b, ridge x -> z. The fit formats use fit-side requests and fit preambles only (r6_guard.py)."""
import sys
from r6lib import *
import r6_guard as GD


def capture_fmt():
    model, tok = load()
    XH, HH, IH, XB, HB, IB = [], [], [], [], [], []
    for fmt in GD.FIT_FORMATS:
        rows = read_jsonl(RAW / "suites" / f"gf_{fmt}.jsonl")
        RAWMAP.clear()
        RAWMAP.update({r["prompt"]: r["raw"] for r in rows})
        X, H = capture(model, tok, rows, tokens=40000, max_batch=64)
        hm = np.array([r["harm"] == 1 for r in rows])
        XH.append(X[hm]); HH.append(H[hm]); IH += [f"{fmt}:{r['id']}" for r in rows if r["harm"] == 1]
        XB.append(X[~hm]); HB.append(H[~hm]); IB += [f"{fmt}:{r['id']}" for r in rows if r["harm"] == 0]
        log(f"{fmt}: {int(hm.sum())} harmful, {int((~hm).sum())} harmless")
    RAWMAP.clear()
    np.savez(CACHE / "act_gf_h.npz", x=np.concatenate(XH), h=np.concatenate(HH), ids=np.array(IH))
    np.savez(CACHE / "act_gf_b.npz", x=np.concatenate(XB), h=np.concatenate(HB), ids=np.array(IB))
    log("R6_READ_CAPTURE DONE")


def lab():
    os.environ["R5_GUARD"] = "g3"
    import r5_lab as LB
    LB.CAND["v5"] = (LB.V2_H + LB.ITW_H + LB.MJ_H + ["gf_h"], LB.V2_B + LB.ITW_B + LB.MJ_B + ["gf_b"])
    sys.argv = ["r5_lab.py", "v5"]
    LB.main()
    V5, V4 = vec("v5g"), vec("v4")
    print("cos(w v5, w v4)", float(V5["w"] @ V4["w"] / (np.linalg.norm(V5["w"]) * np.linalg.norm(V4["w"]))), "theta0", V5["theta0"], "(v4:", V4["theta0"], ")")
    print("R6_READ_LAB DONE")


if __name__ == "__main__":
    capture_fmt() if sys.argv[1] == "capture" else lab()
