"""Mac test of the GPU worker (wsp/worker.py) talking to the real driver-side client (passes.RemoteBackend) through a queue folder.
One model load (MPS, fp16), tiny jobs: at most 16 prompts per job and 24 new tokens in the behave job. No network (HF_HUB_OFFLINE=1).

  .venv/bin/python wsp/tests/test_worker.py --private <folder outside wsp/>      (replies of the behave job are written there)

The queue and the driver-side run live in wsp/runs/_worker_test/ so that scripts/pipeline/check_no_harmful_text.py wsp scans them.
Checks: start-up (checkpoint equality, reference fire rates); (1) a read through the driver's run_forward, RemoteBackend and
validate_response; (2) original vs current; (3) an edited arm with a lowered threshold; (4) layer 22's MLP hash after every job;
(5) a failed job is not counted; (6) restart recovery (a job claimed by a dead worker, planted before the worker starts) and reuse of
a finished answer after a client restart; a behave job; (7) the text checker. Writes wsp/tests/worker_report.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WSP = Path(__file__).resolve().parents[1]
BUILD = WSP.parent
PY = str(BUILD / ".venv/bin/python")
sys.path.insert(0, str(WSP))
import config as C  # noqa: E402
import driver as D  # noqa: E402
import edits as E  # noqa: E402
import passes as P  # noqa: E402


def silu(x):
    return x / (1 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    T = WSP / "runs" / "_worker_test"
    if T.exists():
        shutil.rmtree(T)
    queue = T / "queue"
    queue.mkdir(parents=True)
    results, numbers = [], {}

    def check(name, ok, detail=""):
        results.append({"check": name, "pass": bool(ok), "detail": str(detail)[:600]})
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {str(detail)[:300]}" if detail != "" else ""), flush=True)

    rid = "workertest-" + time.strftime("%Y%m%d-%H%M%S")
    cfg = C.load(WSP / "configs" / "pilot.json", [f"remote.queue_dir={queue}", "remote.poll_s=1", "remote.timeout_s=1800"])
    cfg["run_id"] = rid
    run_dir = T / "run"
    run_dir.mkdir()
    run = D.Run(cfg, run_dir, False, D.Log(run_dir / "driver.log", echo=False))
    run.create()
    sets = json.loads((run.case / "sets.json").read_text())
    formats = json.loads((run.case / "formats.json").read_text())
    knob = json.loads((run.case / "knob.json").read_text())
    KV = np.load(run.case / "knob_vectors.npz")

    # (6, part 1) a job claimed by a worker that died: planted before the worker starts, with a partial output folder
    cfg_r = json.loads(json.dumps(cfg))
    cfg_r["run_id"] = "recovery-test"
    rc = P.RemoteBackend(cfg_r)
    norm_r, picks_r = P.validate_request({"type": "read", "weights": "current", "scale": 1.0, "set": "xstest_safe", "n": 4, "format": "sys_long",
                                          "positions": "marker", "return": {"mlp22_units": "knob", "first_token": True}}, sets, formats, cfg)
    key_r, eh_r, sh_r = rc.submit(900, norm_r, picks_r, run.current_edit(), run.start_edit(), set())
    os.rename(queue / "fwd_900.json", queue / "fwd_900.json.claimed")
    (queue / "fwd_900.tmp").mkdir()
    (queue / "fwd_900.tmp" / "hidden_22.npy").write_bytes(b"partial output of a crashed worker")

    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    wlog = open(T / "worker_stdout.log", "w")
    t_start = time.time()
    proc = subprocess.Popen([PY, str(WSP / "worker.py"), "--queue", str(queue), "--private", a.private, "--device", a.device, "--poll", "1"],
                            stdout=wlog, stderr=subprocess.STDOUT, env=env, cwd=str(BUILD))
    info = None
    try:
        while time.time() - t_start < 1500:
            if (queue / "worker_info.json").exists():
                info = json.loads((queue / "worker_info.json").read_text())
                if info.get("ready") or "reference_check" in info:
                    break
            if proc.poll() is not None:
                break
            time.sleep(2)
        numbers["worker_start_s"] = round(time.time() - t_start, 1)
        check("the worker started: model loaded once, layer 22's MLP equals the checkpoint", info and info.get("layer22_mlp_equals_checkpoint"),
              {k: info.get(k) for k in ("device", "dtype", "model_load_s", "layer22_mlp_hash")} if info else (T / "worker_stdout.log").read_text()[-800:])
        rcheck = (info or {}).get("reference_check", {})
        numbers["reference_check"] = rcheck
        check("start-up reference: AyaDial-Improved at s = 1 on the 200 format-pool requests reproduces positions_v3g.json "
              "(fire within 2 prompts per side, mean z within 0.02)", rcheck.get("pass"), json.dumps(rcheck.get("formats")))
        if not (info and info.get("ready")):
            raise RuntimeError("the worker is not ready")

        # (6, part 2) the planted job ran again on start; then a client restart reuses the finished answer
        rwork = T / "recovery_work"
        rwork.mkdir()
        res_r = rc.wait(900, norm_r, key_r, eh_r, sh_r, rwork)
        meta_r = rc.last_meta
        ok_files = all((rwork / "fwd_900" / f).exists() for f in ("mlp22_units.npy", "first_token.npy", "meta.json"))
        check("(6) restart recovery: a job claimed by a dead worker is run again when the worker starts; its partial output was removed",
              meta_r.get("recovered") and not (queue / "fwd_900.tmp").exists() and ok_files and res_r["by_set"]["xstest_safe"]["n"] == 4,
              {"recovered": meta_r.get("recovered"), "partial_dir_left": (queue / "fwd_900.tmp").exists(), "summary": res_r["by_set"]})
        mt = (queue / "fwd_900.done.json").stat().st_mtime
        key2, eh2, sh2 = rc.submit(900, norm_r, picks_r, run.current_edit(), run.start_edit(), set())
        rc.wait(900, norm_r, key2, eh2, sh2, rwork)
        check("(6) a client restart re-sends nothing: the same job key reuses the finished answer (no new job file, answer untouched)",
              key2 == key_r and (queue / "fwd_900.done.json").stat().st_mtime == mt and not (queue / "fwd_900.json").exists())

        def last_result():
            return json.loads((run.dir / "tool_calls.jsonl").read_text().splitlines()[-1])["result"]

        base = {"type": "read", "scale": 1.0, "set": "format_pool", "n": 16, "format": "default", "positions": "last:4",
                "return": {"hidden": [18, 22, 23], "mlp22_in": True, "mlp22_units": "allowed", "first_token": True}}
        # (1) a read through the driver
        t0 = time.time()
        out1, kind1 = run.run_forward(json.dumps({**base, "weights": "current"}))
        numbers["read_job_wall_s_driver_side"] = round(time.time() - t0, 1)
        res1, meta1 = last_result(), run.backend.last_meta
        numbers["read_job_s_worker_side"] = meta1["seconds"]
        d1 = run.work / "fwd_1"
        want = {"hidden_18": [16, 4, 2048], "hidden_22": [16, 4, 2048], "hidden_23": [16, 4, 2048], "mlp22_in": [16, 4, 2048],
                "mlp22_units": [16, 4, 10], "first_token": [16]}
        got = {k: list(np.load(d1 / f"{k}.npy").shape) for k in want}
        dt = {k: str(np.load(d1 / f"{k}.npy").dtype) for k in want}
        check("(1) a read through the driver's run_forward and RemoteBackend: counted, arrays copied into WORK with the promised shapes, "
              "float16, meta.json lists set, id and label",
              kind1 == "read" and run.state["reads"] == 1 and got == want and set(dt.values()) == {"float16"}
              and len(json.loads((d1 / "meta.json").read_text())["rows"]) == 16 and "Forward pass 1 done" in out1,
              {"kind": kind1, "shapes": got, "summary": res1["by_set"]})
        try:
            P.validate_response({"type": "read"}, res1)
            ok = True
        except ValueError:
            ok = False
        check("(1) validate_response accepts the worker's summary (no text field; by_set and shapes present)", ok and "_worker" not in res1)

        # (2) original vs current
        out2, kind2 = run.run_forward(json.dumps({**base, "weights": "original", "scale": 0.0}))
        res2, d2 = last_result(), run.work / "fwd_2"
        f1, f2 = res1["by_set"]["format_pool"], res2["by_set"]["format_pool"]
        same_inputs = all(np.array_equal(np.load(d1 / f"{k}.npy"), np.load(d2 / f"{k}.npy")) for k in ("hidden_18", "hidden_22", "mlp22_in"))
        h23_marker_differs = not np.array_equal(np.load(d1 / "hidden_23.npy")[:, -1], np.load(d2 / "hidden_23.npy")[:, -1])
        check("(2) original: fire rate 0 on both sides; inputs of layer 22 identical to current (hidden 18, 22, mlp22_in bit-equal); "
              "the output of layer 22 at the marker differs",
              kind2 == "read" and f2["fire_rate"] == 0 and all(v["fire_rate"] == 0 for v in f2["by_label"].values()) and same_inputs
              and h23_marker_differs and f2["z_mean"] == f1["z_mean"],
              {"original": f2, "current": f1})
        # current against the reference and against the kernel's formula on the returned arrays
        ids_u = list(np.load(d1 / "mlp22_unit_ids.npy"))
        A1 = np.load(d1 / "mlp22_units.npy").astype(np.float32)[:, -1]
        meta_rows = json.loads((d1 / "meta.json").read_text())["rows"]
        lab = np.array([r["label"] for r in meta_rows])
        fire_arr = 0.2 * A1[:, ids_u.index(6666)] - 0.2 * A1[:, ids_u.index(1896)]  # coef = gain*beta/alpha = 0.2, opposite signs
        x = np.load(d1 / "mlp22_in.npy").astype(np.float32)[:, -1]
        al, ka, be, th = knob["alpha"], knob["kappa"], knob["beta"], knob["theta0"]

        def formula(theta, s=1.0):
            xm, xq, xw = x @ KV["m"], x @ KV["q"], x @ KV["w"]
            g = -al * theta * xm - ka * xq + s * al * xw
            return (silu(g) - silu(g - al / be * xm)) * xm * be / al

        ref = json.loads((BUILD / "results/round6/robust/positions_v3g.json").read_text())["formats"]["rb_default"]
        rep = {lb: {"worker_fire_rate": f1["by_label"][lb]["fire_rate"], "from_unit_arrays": round(float((fire_arr[lab == lb] > 0.5).mean()), 3),
                    "kernel_formula_on_returned_mlp22_in": round(float((formula(th)[lab == lb] > 0.5).mean()), 3), "n": int((lab == lb).sum())}
               for lb in ("harmful", "harmless")}
        rep["reference_rb_default_200_prompts"] = {"harmful": ref["strict_s1_fires_harmful"], "harmless": ref["strict_s1_fires_harmless"]}
        f_cur = formula(th)
        rep["prompts_where_arrays_and_formula_disagree"] = int(((fire_arr > 0.5) != (f_cur > 0.5)).sum())
        rep["of_which_within_0.05_of_the_threshold"] = int((((fire_arr > 0.5) != (f_cur > 0.5)) & (np.abs(f_cur - 0.5) < 0.05)).sum())
        rep["max_abs_diff_write_arrays_vs_formula"] = round(float(np.abs(fire_arr - f_cur).max()), 4)
        numbers["current_fire_16"] = rep
        agree = all(rep[lb]["worker_fire_rate"] == rep[lb]["from_unit_arrays"] for lb in ("harmful", "harmless")) and \
            rep["prompts_where_arrays_and_formula_disagree"] == rep["of_which_within_0.05_of_the_threshold"]
        check("(2) current: the worker's fire rate equals the one recomputed from the returned unit arrays, and the kernel formula on "
              "the returned mlp22_in agrees per prompt (except within 0.05 of the threshold); the 200-prompt start-up check is the "
              "match to positions_v3g.json", agree, rep)

        # (3) an edited arm: the starting knob with its threshold lowered by 0.45 (gate install rows += alpha * 0.45 * m)
        st = run.start_edit()
        g = st[E.SHORT["gate_proj"]]
        g2 = {**g, "install": (g["install"] + np.float32(al * 0.45) * KV["m"][None, :]).astype(np.float32)}
        new = dict(st)
        new[E.SHORT["gate_proj"]] = g2
        run.save_edit_state(new)
        out3, kind3 = run.run_forward(json.dumps({**base, "weights": "edited"}))
        res3, d3 = last_result(), run.work / "fwd_3"
        f3 = res3["by_set"]["format_pool"]
        A3 = np.load(d3 / "mlp22_units.npy").astype(np.float32)[:, -1]
        fire3 = 0.2 * A3[:, ids_u.index(6666)] - 0.2 * A3[:, ids_u.index(1896)]
        f_ed = formula(th - 0.45)
        pred = f_ed > 0.5
        rep3 = {lb: {"current": f1["by_label"][lb]["fire_rate"], "edited": f3["by_label"][lb]["fire_rate"],
                     "predicted_by_formula": round(float(pred[lab == lb].mean()), 3)} for lb in ("harmful", "harmless")}
        rep3["prompts_whose_fire_changed"] = int(((fire3 > 0.5) != (fire_arr > 0.5)).sum())
        miss = (fire3 > 0.5) != pred
        rep3["prompts_where_worker_and_formula_disagree"] = int(miss.sum())
        rep3["of_which_within_0.05_of_the_threshold"] = int((miss & (np.abs(f_ed - 0.5) < 0.05)).sum())
        rep3["edit_hash"] = run.state["edit_hash"]
        numbers["edited_threshold_minus_0.45"] = rep3
        up = all(f3["by_label"][lb]["fire_rate"] >= f1["by_label"][lb]["fire_rate"] for lb in ("harmful", "harmless")) and \
            any(f3["by_label"][lb]["fire_rate"] > f1["by_label"][lb]["fire_rate"] for lb in ("harmful", "harmless"))
        check("(3) edited arm (threshold lowered by 0.45): fire goes up and never down on any prompt, as the formula predicts per prompt "
              "(except within 0.05 of the threshold); the worker applied the driver's edit hash",
              kind3 == "read" and up and rep3["prompts_where_worker_and_formula_disagree"] == rep3["of_which_within_0.05_of_the_threshold"]
              and np.all((fire3 > 0.5) >= (fire_arr > 0.5)) and run.backend.last_meta["edit_hash_applied"] == run.state["edit_hash"], rep3)

        # behave: three arms on 12 prompts (6 read before, 6 never read), 24 new tokens
        breq = {"type": "behave", "arms": [{"weights": "original"}, {"weights": "current", "scale": 1}, {"weights": "edited", "scale": 1}],
                "sets": ["format_pool", "harmful"], "n": 6, "format": "default", "max_new_tokens": 24}
        t0 = time.time()
        outb, kindb = run.run_forward(json.dumps(breq))
        numbers["behave_job_wall_s_driver_side"] = round(time.time() - t0, 1)
        resb, metab = last_result(), run.backend.last_meta
        lines = [json.loads(l) for l in (run.work / f"beh_{run.state['pass_k']}.jsonl").read_text().splitlines()]
        numbers["behave"] = {k: metab.get(k) for k in ("seconds", "judge_load_s", "judge_seconds", "judge_calls", "judge_unparsed",
                                                        "generated_tokens", "decode_steps", "seconds_generating", "batches")}
        numbers["behave"]["seconds_per_decode_step_per_batch"] = round(metab["seconds_generating"] / max(1, metab["decode_steps"]), 3)
        numbers["behave"]["seconds_per_generated_token"] = round(metab["seconds_generating"] / max(1, metab["generated_tokens"]), 4)
        fp = resb["arms"][1]["by_set"]
        check("(B) a behave job (3 arms, 12 prompts, 24 tokens, real Qwen3Guard-Gen-0.6B from the local cache): counted; rates by arm "
              "and set; 3 paired comparisons with intervals; 6 prompts read before and 6 never read; labels by id in WORK",
              kindb == "behave" and run.state["behaves"] == 1 and len(resb["arms"]) == 3 and len(resb["pairs"]) == 3
              and fp["format_pool"]["n_read_before"] == 6 and fp["harmful"]["n_never_read"] == 6 and len(lines) == 12
              and all(len(l["labels"]) == 3 for l in lines) and metab["judge_unparsed"] == 0,
              {"rates_current": {s: {k: v for k, v in e.items() if k in ("n", "refusal", "unsafe", "degenerate")} for s, e in fp.items()},
               "pair_0_1": {k: resb["pairs"][0][k] for k in ("unsafe_diff_b_minus_a", "ci95", "refusal_diff_b_minus_a", "refusal_ci95")}})
        priv = list(Path(a.private).rglob("fwd_*.replies.jsonl"))
        check("(B) replies and requests of the behave job went to the private folder, outside the queue", priv and str(Path(a.private).resolve()) not in str(queue.resolve()),
              [str(p.name) for p in priv])

        # (5) a failed job: the client sends a wrong edit hash, the worker refuses it, the driver does not count the pass
        reads0, k0 = run.state["reads"], run.state["pass_k"]
        real_hash = E.state_hash
        E.state_hash = lambda st: "0" * 16
        try:
            out5, kind5 = run.run_forward(json.dumps({**base, "weights": "edited", "scale": 1.25}))
        finally:
            E.state_hash = real_hash
        k5 = k0 + 1
        check("(5) a failed job writes fwd_<k>.failed.json; the driver reports it, does not count it and does not advance k",
              kind5 == "forward_failed" and run.state["reads"] == reads0 and run.state["pass_k"] == k0 and (queue / f"fwd_{k5}.failed.json").exists()
              and (run.dir / "requests" / f"fwd_{k5}.failed.json").exists() and "does not match" in out5, out5[:300])
        out5b, kind5b = run.run_forward(json.dumps({**base, "weights": "edited", "scale": 1.25}))
        check("(5) the same request sent again with the right hash runs under the same k (the stale failure is cleared)",
              kind5b == "read" and run.state["pass_k"] == k5 and not (queue / f"fwd_{k5}.failed.json").exists(), out5b[:120])

        # (4) layer 22's MLP after every job equals the original
        hashes = []
        for p in sorted(queue.glob("fwd_*.done.json")):
            m = json.loads(p.read_text())["_worker"]
            hashes.append((p.name, m["weights_hash_original"], m["weights_hash_after"]))
        check("(4) after every job layer 22's MLP hash (gate, up, down as stored) equals the hash at start",
              hashes and all(h0 == h1 == info["layer22_mlp_hash"] for _, h0, h1 in hashes), hashes)
    except Exception as e:  # noqa: BLE001
        import traceback
        check("the test ran to the end without an unexpected error", False, traceback.format_exc()[-1500:])
    finally:
        (queue / "STOP").write_text("")
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
        wlog.close()
    check("the worker stopped cleanly on the STOP file", proc.returncode == 0, proc.returncode)

    # (7) no prompt or reply text in the queue (or anywhere in wsp/)
    r = subprocess.run(["python3", str(BUILD / "scripts/pipeline/check_no_harmful_text.py"), "wsp"], cwd=str(BUILD), capture_output=True, text=True)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    numbers["text_checker_last_line"] = last
    check("(7) scripts/pipeline/check_no_harmful_text.py wsp finds nothing", r.returncode == 0, last)
    rep = {"run_id": rid, "results": results, "passed": sum(x["pass"] for x in results), "total": len(results), "numbers": numbers}
    (WSP / "tests" / "worker_report.json").write_text(json.dumps(rep, indent=1))
    print(f"{rep['passed']} of {rep['total']} worker checks passed")
    print(json.dumps(numbers, indent=1))
    return 0 if rep["passed"] == rep["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
