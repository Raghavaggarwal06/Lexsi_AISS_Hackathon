"""GGUF export and per-request scales for both sides of the knob.
  python scripts/r5_gguf.py prep                       CPU only: build llama.cpp with CUDA, convert the base model to GGUF (f16, Q4_K_M) with round 4's stop-token fix
  python scripts/r5_gguf.py test <theta> <t> [n]       lax adapters -> PEFT -> GGUF; llama-server with four adapters; the same prompts through HF as reference
Adapters, in the order the server loads them:
  0 strict install: units 6666 and 1896 at s = 0 (gate, up, down)      1 strict knob: alpha * w on both gate rows (per-request scale = s)
  2 lax install: gate and up rows of units 1785 and 4466, their down columns set to zero      3 lax knob: the two down columns, rank 1 (scale = t)
All four are built from one vector file (default v3g: the refit read and guard).
One user-facing number s maps to scales (1, max(s, 0), 1, max(-s, 0)). The lax pair can be withheld without touching the strict pair.
Private: adapters stay in runs_private/round5/adapters."""
import glob, sys, urllib.request
from r5lib import *

LC = "/content/llama.cpp"
AD = RAW / "adapters"


def sh(c, t=3600, cwd=None):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t, cwd=cwd)
    return (r.stdout + r.stderr)[-600:]


def prep():
    src = glob.glob("/root/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global/snapshots/*")[0]
    if not os.path.exists(LC):
        log(sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {LC}", 600))
    if not os.path.exists(f"{LC}/build-cuda/bin/llama-server"):
        log(sh("cmake -B build-cuda -DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF > /content/cuda_cfg.log 2>&1; cmake --build build-cuda -j8 --target llama-server llama-quantize > /content/cuda_build.log 2>&1; ls -la build-cuda/bin | head", 5400, LC))
    if not os.path.exists("/content/base_f16.gguf"):
        log(sh(f"pip -q install gguf sentencepiece; python convert_hf_to_gguf.py {src} --outfile /content/base_f16.gguf --outtype f16", 2400, LC)[-300:])
    fix = "python gguf-py/gguf/scripts/gguf_new_metadata.py {a} {b} --special-token-by-id eos 3 --special-token-by-id eot 261001 --force"
    if not os.path.exists("/content/base_f16_fix.gguf"):
        log(sh(fix.format(a="/content/base_f16.gguf", b="/content/base_f16_fix.gguf"), 1200, LC)[-200:])
    if not os.path.exists("/content/base_q4km.gguf"):
        log(sh("./build-cuda/bin/llama-quantize /content/base_f16.gguf /content/base_q4km.gguf Q4_K_M", 2400, LC)[-200:])
    if not os.path.exists("/content/base_q4km_fix.gguf"):
        log(sh(fix.format(a="/content/base_q4km.gguf", b="/content/base_q4km_fix.gguf"), 1200, LC)[-200:])
    log("ls: " + sh("ls -la /content/*.gguf; git -C /content/llama.cpp rev-parse HEAD"))
    log("R5_GGUF_PREP DONE")


def post(path, body, port, timeout=1200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def test(theta, t, nper, svec="v3g"):
    from peft import LoraConfig
    from safetensors.torch import save_file
    from concurrent.futures import ThreadPoolExecutor
    src = glob.glob("/root/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global/snapshots/*")[0]
    res = {"theta": theta, "t": t, "vectors": svec, "lax_units": list(LAX_UNITS), "strict_units": list(STRICT_UNITS), "llama_cpp_commit": sh(f"git -C {LC} rev-parse HEAD").strip()}
    model, tok = load()
    V = vec(svec)
    mlp = model.model.layers[LYR].mlp
    G, Uw, Dw = (x.weight.data.float().cpu() for x in (mlp.gate_proj, mlp.up_proj, mlp.down_proj))
    T = lambda v: torch.tensor(np.asarray(v, np.float32))
    pre = "base_model.model.model.layers.22.mlp."
    def selector(units):
        E = torch.zeros((11008, len(units)))
        for j, u in enumerate(units):
            E[u["n"], j] = 1.0
        return E
    # strict side: install = the two units at s = 0 (gate, up, down); knob = alpha * w on both gate rows, scaled by s per request
    su = strict_units(V, 0.0)
    Es = selector(su)
    strict_install = {pre + "gate_proj.lora_A.weight": torch.stack([T(u["gate"]) - G[u["n"]] for u in su]), pre + "gate_proj.lora_B.weight": Es.clone(),
                      pre + "up_proj.lora_A.weight": torch.stack([T(u["up"]) - Uw[u["n"]] for u in su]), pre + "up_proj.lora_B.weight": Es.clone(),
                      pre + "down_proj.lora_A.weight": Es.T.clone(), pre + "down_proj.lora_B.weight": torch.stack([T(u["down"]) - Dw[:, u["n"]] for u in su], 1)}
    wA = T(ALPHA * V["w"])
    strict_knob = {pre + "gate_proj.lora_A.weight": torch.stack([wA for _ in su]), pre + "gate_proj.lora_B.weight": Es.clone()}
    # lax side: install = gate and up rows, down columns zeroed; knob = the two down columns (rank 1), scaled by t per request
    lu = lax_units(V, 1.0, theta)
    El = selector(lu)
    lax_install = {pre + "gate_proj.lora_A.weight": torch.stack([T(u["gate"]) - G[u["n"]] for u in lu]), pre + "gate_proj.lora_B.weight": El.clone(),
                   pre + "up_proj.lora_A.weight": torch.stack([T(u["up"]) - Uw[u["n"]] for u in lu]), pre + "up_proj.lora_B.weight": El.clone(),
                   pre + "down_proj.lora_A.weight": El.T.clone(), pre + "down_proj.lora_B.weight": torch.stack([-Dw[:, u["n"]] for u in lu], 1)}
    dcol = T(lu[1]["down"])  # +beta/alpha * r (the second unit); the first unit carries the opposite sign
    lax_knob = {pre + "down_proj.lora_A.weight": (El[:, 1] - El[:, 0]).unsqueeze(0).clone(), pre + "down_proj.lora_B.weight": dcol.unsqueeze(1).clone()}
    res["knob_check_max_abs"] = float((T(lu[0]["down"]) + dcol).abs().max())
    for name, sd, mods, rank in (("strict_install", strict_install, ["gate_proj", "up_proj", "down_proj"], 2), ("strict_knob", strict_knob, ["gate_proj"], 2),
                                 ("lax_install", lax_install, ["gate_proj", "up_proj", "down_proj"], 2), ("lax_knob", lax_knob, ["down_proj"], 1)):
        d = AD / name
        d.mkdir(parents=True, exist_ok=True)
        cfg = LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0, target_modules=[f"model.layers.22.mlp.{m}" for m in mods], task_type="CAUSAL_LM")
        cfg.base_model_name_or_path = MODEL
        cfg.save_pretrained(d)
        save_file({k: v.contiguous() for k, v in sd.items()}, str(d / "adapter_model.safetensors"))
        out = AD / f"{name}.gguf"
        res[f"convert_{name}"] = sh(f"python convert_lora_to_gguf.py --base {src} --outtype f32 --outfile {out} {d}", 900, LC)[-160:]
        res[f"{name}_gguf_bytes"] = out.stat().st_size if out.exists() else None
        res[f"{name}_sha256"] = hashlib.sha256((d / "adapter_model.safetensors").read_bytes()).hexdigest()
    log(json.dumps({k: v for k, v in res.items() if "bytes" in k or "check" in k}))
    # prompts: harmless prompts the base refuses (where the lax side should act), harmful ones, plain benign ones
    base = {}
    for f in ("base_wjx", "base_own"):
        base.update(labels(f))
    rng = random.Random(21)
    allr = sets()
    pick = lambda name, refused: rng.sample([r for r in allr if r["set"] == name and (refused is None or base[rk(r)][0] == refused)], nper)
    rows = pick("wj_adversarial_benign", 1) + pick("orbench_hard", 1) + pick("dev_ho_b", None) + pick("benign", None) + pick("dev_ho_h", None) + pick("harmful", None) + pick("wj_adversarial_harmful", None)
    harmful = np.array([r["set"] in ("dev_ho_h", "harmful", "wj_adversarial_harmful") for r in rows])
    ref_h = np.array([r["set"] in ("wj_adversarial_benign", "orbench_hard") for r in rows])  # harmless and refused by the base
    del model
    torch.cuda.empty_cache()
    CFG = {"orig": (0, 0, 0, 0), "installed": (1, 0, 1, 0), "strict1": (1, 1, 1, 0), f"lax{t:g}": (1, 0, 1, t), f"both1_{t:g}": (1, 1, 1, t)}
    BIN = f"{LC}/build-cuda/bin/llama-server"
    rates = {}
    def summarize(lab):
        a = np.array([lab[rk(r)] for r in rows], float)
        return {"harmful_refusal": float(a[harmful, 0].mean() * 100), "harmful_unsafe": float(a[harmful, 1].mean() * 100), "base_refused_harmless_refusal": float(a[ref_h, 0].mean() * 100),
                "other_harmless_refusal": float(a[~harmful & ~ref_h, 0].mean() * 100), "degenerate": float(a[:, 2].mean() * 100), "n": len(rows)}
    for tag, gguf in (("f16", "/content/base_f16_fix.gguf"), ("q4km", "/content/base_q4km_fix.gguf")):
        if not os.path.exists(gguf):
            res[f"{tag}_server"] = "base gguf missing"; continue
        port = 8120 + (tag == "q4km")
        srv = subprocess.Popen(f"exec {BIN} -m {gguf} --lora {AD}/strict_install.gguf --lora {AD}/strict_knob.gguf --lora {AD}/lax_install.gguf --lora {AD}/lax_knob.gguf --lora-init-without-apply --port {port} -c 16384 -np 8 --jinja -ngl 99",
                               shell=True, stdout=open(f"/content/srv5_{tag}.log", "w"), stderr=subprocess.STDOUT)
        ok = False
        for _ in range(150):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); ok = True; break
            except Exception:
                time.sleep(2)
        if not ok:
            res[f"{tag}_server"] = "did not start: " + open(f"/content/srv5_{tag}.log").read()[-400:]
            srv.kill(); continue
        try:
            res[f"{tag}_lora_list"] = [{k: a.get(k) for k in ("id", "scale")} for a in json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/lora-adapters", timeout=10).read())]
        except Exception as e:
            res[f"{tag}_lora_list"] = type(e).__name__
        for cname, sc in CFG.items():
            name = f"gguf5_{tag}_{cname}"
            path = RAW / "gen" / f"{name}.jsonl"
            if not path.exists():
                lora = [{"id": i, "scale": float(x)} for i, x in enumerate(sc)]
                def one(r):
                    c = post("/v1/chat/completions", {"messages": [{"role": "user", "content": r["prompt"]}], "max_tokens": 128, "temperature": 0, "lora": lora}, port)
                    return {"id": r["id"], "set": r["set"], "prompt": r["prompt"], "response": c["choices"][0]["message"]["content"], "finish": c["choices"][0]["finish_reason"]}
                with ThreadPoolExecutor(8) as ex:
                    write_jsonl(path, list(ex.map(one, rows)))
            got = read_jsonl(path)
            judge(got)
            lab = {rk(r): label(r) for r in got}
            (OUT / "labels" / f"{name}.json").write_text(json.dumps(lab))
            rates[name] = {**summarize(lab), "finish_stop": float(np.mean([r.get("finish") == "stop" for r in got]) * 100)}
            log(f"{name} {rates[name]}")
        srv.kill(); time.sleep(5)
    model, tok = load()
    HF = {k: (None if v is None else dict(f="both", svec=svec, lvec=svec, theta=theta, **v)) for k, v in {"orig": None, "installed": dict(s=0.0, t=0.0), "strict1": dict(s=1.0, t=0.0), f"lax{t:g}": dict(s=0.0, t=t), f"both1_{t:g}": dict(s=1.0, t=t)}.items()}
    os.environ["R4_TOK"], os.environ["R4_BATCH"] = "90000", "160"
    for cname, c in HF.items():
        name = f"hfref5_{cname}"
        lab = run_cell(model, tok, name, rows, None if c is None else context(model, c))
        rates[name] = summarize(lab)
        for tag in ("f16", "q4km"):
            g = OUT / "labels" / f"gguf5_{tag}_{cname}.json"
            if g.exists():
                gl = json.loads(g.read_text())
                rates[name][f"refusal_label_agreement_with_{tag}"] = float(np.mean([gl[rk(r)][0] == lab[rk(r)][0] for r in rows]) * 100)
        log(f"{name} {rates[name]}")
    res["rates"] = rates
    (OUT / "gguf").mkdir(exist_ok=True)
    (OUT / "gguf" / "export_both_sides.json").write_text(json.dumps(res, indent=1))
    sync()
    log("R5_GGUF DONE")


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep()
    else:
        test(float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 20, sys.argv[5] if len(sys.argv) > 5 else "v3g")
