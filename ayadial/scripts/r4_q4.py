"""Q4_K_M stop problem: CPU-only llama.cpp, f16 vs Q4_K_M, harmless prompts. Prints token ids so the cause is visible."""
import json, os, subprocess, sys, time, urllib.request, glob
lc = "/content/llama.cpp"
def sh(c, t=3600, cwd=None):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t, cwd=cwd); return (r.stdout + r.stderr)[-1500:]
if not os.path.exists(lc):
    print(sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {lc}", 600))
print("commit", sh("git rev-parse HEAD", 60, lc).strip())
if not os.path.exists(f"{lc}/build/bin/llama-server"):
    print(sh("cmake -B build -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF && cmake --build build -j10 --target llama-server llama-quantize", 3000, lc)[-600:])
src = glob.glob("/root/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global/snapshots/*")[0]
if not os.path.exists("/content/base_f16.gguf"):
    print(sh(f"pip -q install gguf sentencepiece; python convert_hf_to_gguf.py {src} --outfile /content/base_f16.gguf --outtype f16", 1800, lc)[-600:])
if not os.path.exists("/content/base_q4km.gguf"):
    print(sh("./build/bin/llama-quantize /content/base_f16.gguf /content/base_q4km.gguf Q4_K_M", 1800, lc)[-300:])
print(sh("python gguf-py/gguf/scripts/gguf_dump.py /content/base_q4km.gguf --no-tensors 2>/dev/null | grep -i -E 'token_id|eos|eot|chat_template' | cut -c1-200", 300, lc))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
gc = json.load(open(f"{src}/generation_config.json")) if os.path.exists(f"{src}/generation_config.json") else {}
print("hf generation_config", gc, "| tok.eos", tok.eos_token, tok.eos_token_id)
PROMPTS = ["What is the capital of France?", "Give me two tips for sleeping better.", "Translate 'good morning' into Spanish."]
def post(path, body, port):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())
out = {}
for tag, gguf, port in (("q4km", "/content/base_q4km.gguf", 8101), ("f16", "/content/base_f16.gguf", 8102)):
    srv = subprocess.Popen(f"./build/bin/llama-server -m {gguf} --port {port} -c 4096 -np 1 --jinja -t 10", shell=True, cwd=lc, stdout=open(f"/content/srv_{tag}.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); break
        except Exception:
            time.sleep(2)
    res = []
    for p in PROMPTS:
        chat = post("/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "max_tokens": 200, "temperature": 0}, port)
        raw_prompt = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        raw = post("/completion", {"prompt": raw_prompt, "n_predict": 200, "temperature": 0, "return_tokens": True, "special": True}, port)
        ids = raw.get("tokens", [])
        hf = tok(raw_prompt, return_tensors="pt", add_special_tokens=False)
        res.append({"chat_finish": chat["choices"][0]["finish_reason"], "chat_tokens": chat["usage"]["completion_tokens"], "raw_stop_type": raw.get("stop_type"), "raw_n": len(ids),
                    "raw_has_END_RESPONSE_261001": 261001 in ids, "raw_has_END_OF_TURN_6": 6 in ids, "raw_first_index_261001": ids.index(261001) if 261001 in ids else None,
                    "raw_tail_ids": ids[-6:], "after_end_response_ids": ids[ids.index(261001): ids.index(261001) + 8] if 261001 in ids else None})
    # candidate fix: explicit stop strings on the chat endpoint
    fix = post("/v1/chat/completions", {"messages": [{"role": "user", "content": PROMPTS[0]}], "max_tokens": 200, "temperature": 0, "stop": ["<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>"]}, port)
    res.append({"fix_stop_strings_finish": fix["choices"][0]["finish_reason"], "fix_tokens": fix["usage"]["completion_tokens"]})
    out[tag] = res
    srv.terminate(); time.sleep(3)
os.makedirs("/content/build/results/round4/q4", exist_ok=True)
json.dump(out, open("/content/build/results/round4/q4/stop_probe.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print("R4_Q4 DONE")
