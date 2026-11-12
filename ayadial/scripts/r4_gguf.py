"""Export the fold as two LoRA adapters and test them in llama.cpp.
  install adapter (always scale 1): rewires the dormant unit(s) of layers[22].mlp (gate row, up row, down column).
  knob adapter (scale = s): rank-1 (or rank-2) on gate_proj only, B = e_unit, A = alpha*w.
  python r4_gguf.py <variant: g4|cap> [n_per_set]"""
import glob, sys, urllib.request
from r4lib import *
from r4_fold import Fold
variant = sys.argv[1]
NPER = int(sys.argv[2]) if len(sys.argv) > 2 else 30
kw = {"g4": dict(gain=4.0), "cap": dict(gain=1.0, cap_beta=4.0)}[variant]
LC, SRC = "/content/llama.cpp", glob.glob("/root/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global/snapshots/*")[0]
AD = ROOT / "adapters" / f"fold_{variant}"
res = {"variant": variant, "kw": kw}


def sh(c, t=3600, cwd=None):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t, cwd=cwd)
    return (r.stdout + r.stderr)[-800:]


# ---------------------------------------------------------------- 1. adapters in PEFT format
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
model, tok = load()
f0 = Fold(model, 0.0, **kw)
P = np.load(CACHE / "fold.npz")
mlp = model.model.layers[22].mlp
units = [(f0.n, f0.g32, f0.d32)] + ([(f0.second, f0.g2, -f0.d32)] if f0.second is not None else [])
G, Uw, Dw = (x.weight.data.float() for x in (mlp.gate_proj, mlp.up_proj, mlp.down_proj))
rank = len(units)
E = torch.zeros((11008, rank), device=DEV)
for j, (n, _, _) in enumerate(units):
    E[n, j] = 1.0
pre = "base_model.model.model.layers.22.mlp."
install = {pre + "gate_proj.lora_A.weight": torch.stack([g - G[n] for n, g, _ in units]), pre + "gate_proj.lora_B.weight": E.clone(),
           pre + "up_proj.lora_A.weight": torch.stack([f0.m32 - Uw[n] for n, _, _ in units]), pre + "up_proj.lora_B.weight": E.clone(),
           pre + "down_proj.lora_A.weight": E.T.clone(), pre + "down_proj.lora_B.weight": torch.stack([d - Dw[:, n] for n, _, d in units], 1)}
wA = torch.tensor(f0.alpha * P["w"], device=DEV, dtype=torch.float32)
knob = {pre + "gate_proj.lora_A.weight": torch.stack([wA for _ in units]), pre + "gate_proj.lora_B.weight": E.clone()}
for name, sd, mods in (("install", install, ["gate_proj", "up_proj", "down_proj"]), ("knob", knob, ["gate_proj"])):
    d = AD / name
    d.mkdir(parents=True, exist_ok=True)
    cfg = LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0, target_modules=[f"model.layers.22.mlp.{m}" for m in mods], task_type="CAUSAL_LM")
    cfg.base_model_name_or_path = MODEL
    cfg.save_pretrained(d)
    save_file({k: v.contiguous().cpu() for k, v in sd.items()}, str(d / "adapter_model.safetensors"))
res["adapter_shapes"] = {k.split("mlp.")[1]: list(v.shape) for k, v in {**install, **{("knob:" + k): v for k, v in knob.items()}}.items()}
del model
torch.cuda.empty_cache()

# ---------------------------------------------------------------- 2. GGUF conversion
for name in ("install", "knob"):
    out = AD / f"{name}.gguf"
    if not out.exists():
        res[f"convert_{name}"] = sh(f"python convert_lora_to_gguf.py --base {SRC} --outtype f32 --outfile {out} {AD / name}", 900, LC)[-300:]
    res[f"{name}_gguf_bytes"] = out.stat().st_size if out.exists() else None
if not os.path.exists("/content/base_f16_fix.gguf"):
    res["fix_f16"] = sh("python gguf-py/gguf/scripts/gguf_new_metadata.py /content/base_f16.gguf /content/base_f16_fix.gguf --special-token-by-id eos 3 --special-token-by-id eot 261001 --force", 900, LC)[-200:]
log(json.dumps({k: v for k, v in res.items() if "convert" in k or "bytes" in k})[:600])

# ---------------------------------------------------------------- 3. llama-server, per-request scale
allr = sets()
rng = random.Random(21)
rows = [r for n in ("wj_adversarial_harmful", "wj_adversarial_benign", "dev_ho_h", "dev_ho_b", "harmful", "benign") for r in rng.sample([x for x in allr if x["set"] == n], NPER)]
BIN = f"{LC}/build-cuda/bin/llama-server" if os.path.exists(f"{LC}/build-cuda/bin/llama-server") else f"{LC}/build/bin/llama-server"
res["server_binary"] = BIN.split("llama.cpp/")[1]


def post(path, body, port, timeout=1200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


from concurrent.futures import ThreadPoolExecutor
out_rates = {}
for tag, gguf in (("f16", "/content/base_f16_fix.gguf"), ("q4km", "/content/base_q4km_fix.gguf")):
    port = 8110 + (tag == "q4km")
    srv = subprocess.Popen(f"exec {BIN} -m {gguf} --lora {AD}/install.gguf --lora {AD}/knob.gguf --lora-init-without-apply --port {port} -c 16384 -np 8 --jinja -ngl 99", shell=True,
                           stdout=open(f"/content/srv_gguf_{tag}.log", "w"), stderr=subprocess.STDOUT)
    ok = False
    for _ in range(150):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); ok = True; break
        except Exception:
            time.sleep(2)
    if not ok:
        res[f"{tag}_server"] = "did not start: " + open(f"/content/srv_gguf_{tag}.log").read()[-400:]
        srv.terminate(); continue
    try:
        res[f"{tag}_lora_list"] = [{k: a.get(k) for k in ("id", "scale")} for a in json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/lora-adapters", timeout=10).read())]
    except Exception as e:
        res[f"{tag}_lora_list"] = f"{type(e).__name__}"
    for sname, lora in (("orig", [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}]), ("s0", [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}]), ("s1", [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 1.0}]),
                        ("s1.5", [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 1.5}])):
        name = f"gguf_{variant}_{tag}_{sname}"
        path = RAW / "gen" / f"{name}.jsonl"
        if not path.exists():
            def one(r):
                c = post("/v1/chat/completions", {"messages": [{"role": "user", "content": r["prompt"]}], "max_tokens": 128, "temperature": 0, "lora": lora}, port)
                return {"id": r["id"], "set": r["set"], "prompt": r["prompt"], "response": c["choices"][0]["message"]["content"], "finish": c["choices"][0]["finish_reason"]}
            with ThreadPoolExecutor(8) as ex:
                got = list(ex.map(one, rows))
            write_jsonl(path, got)
        got = read_jsonl(path)
        judge(got)
        lab = {rk(r): label(r) for r in got}
        (OUT / "labels" / f"{name}.json").write_text(json.dumps(lab))
        a = np.array([lab[rk(r)] for r in rows], float)
        hm = np.array([r["set"] in ("wj_adversarial_harmful", "dev_ho_h", "harmful") for r in rows])
        out_rates[name] = {"harmful_refusal": float(a[hm, 0].mean() * 100), "harmful_unsafe": float(a[hm, 1].mean() * 100), "harmless_refusal": float(a[~hm, 0].mean() * 100), "degenerate": float(a[:, 2].mean() * 100),
                           "finish_stop": float(np.mean([r.get("finish") == "stop" for r in got]) * 100), "n": len(rows)}
        log(f"{name} {out_rates[name]}")
    srv.terminate(); time.sleep(5)
res["rates"] = out_rates
# HF reference on the same prompts: base, fold weights at s = 0, 1, 1.5
model, tok = load()
for sname, ctx in (("orig", None), ("s0", Fold(model, 0.0, **kw)), ("s1", Fold(model, 1.0, **kw)), ("s1.5", Fold(model, 1.5, **kw))):
    name = f"hfref_{variant}_{sname}"
    lab = run_cell(model, tok, name, rows, ctx)
    a = np.array([lab[rk(r)] for r in rows], float)
    hm = np.array([r["set"] in ("wj_adversarial_harmful", "dev_ho_h", "harmful") for r in rows])
    out_rates[name] = {"harmful_refusal": float(a[hm, 0].mean() * 100), "harmful_unsafe": float(a[hm, 1].mean() * 100), "harmless_refusal": float(a[~hm, 0].mean() * 100), "degenerate": float(a[:, 2].mean() * 100), "n": len(rows)}
    for tag in ("f16", "q4km"):
        g = OUT / "labels" / f"gguf_{variant}_{tag}_{sname}.json"
        if g.exists():
            gl = json.loads(g.read_text())
            out_rates[name][f"refusal_label_agreement_with_{tag}"] = float(np.mean([gl[rk(r)][0] == lab[rk(r)][0] for r in rows]) * 100)
    log(f"{name} {out_rates[name]}")
(OUT / "gguf").mkdir(exist_ok=True)
(OUT / "gguf" / f"export_{variant}.json").write_text(json.dumps(res, indent=1))
sync()
log("R4_GGUF DONE")
