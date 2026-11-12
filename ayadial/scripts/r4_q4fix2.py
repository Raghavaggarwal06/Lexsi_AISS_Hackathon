
import json, subprocess, time, urllib.request, os
lc = "/content/llama.cpp"
def sh(c, t=1800): 
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t, cwd=lc); return (r.stdout + r.stderr)[-500:]
def post(path, body, port):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())
out = {"help": sh("python gguf-py/gguf/scripts/gguf_new_metadata.py --help | grep -i -A2 special")}
if not os.path.exists("/content/base_q4km_fix.gguf"):
    out["rewrite"] = sh("python gguf-py/gguf/scripts/gguf_new_metadata.py /content/base_q4km.gguf /content/base_q4km_fix.gguf --special-token-by-id eos 3 --special-token-by-id eot 261001 --force")
out["dump"] = sh("python gguf-py/gguf/scripts/gguf_dump.py /content/base_q4km_fix.gguf --no-tensors 2>/dev/null | grep -i -E 'eos_token_id|eot_token_id' | cut -c1-120")
for tag, gguf, extra in (("metadata_fix", "/content/base_q4km_fix.gguf", ""), ("orig_special_stop", "/content/base_q4km.gguf", "--special")):
    srv = subprocess.Popen(f"./build/bin/llama-server -m {gguf} --port 8104 -c 4096 -np 1 --jinja -t 8 {extra}", shell=True, cwd=lc, stdout=open(f"/content/srv_fix2_{tag}.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:8104/health", timeout=2); break
        except Exception:
            time.sleep(2)
    res = []
    for p in ["What is the capital of France?", "Give me two tips for sleeping better.", "Translate 'good morning' into Spanish.", "Name three primary colours.", "What is 12 times 12?"]:
        body = {"messages": [{"role": "user", "content": p}], "max_tokens": 200, "temperature": 0}
        if extra: body["stop"] = ["<|END_RESPONSE|>", "<EOS_TOKEN>"]
        c = post("/v1/chat/completions", body, 8104)
        txt = c["choices"][0]["message"]["content"]
        res.append({"finish": c["choices"][0]["finish_reason"], "tokens": c["usage"]["completion_tokens"], "content_has_special_text": ("<|END" in txt or "<EOS" in txt or "<|START" in txt), "chars": len(txt)})
    out[tag] = res
    srv.terminate(); time.sleep(4)
json.dump(out, open("/content/build/results/round4/q4/stop_fix2.json", "w"), indent=1)
print(json.dumps(out)); print("R4_Q4FIX2 DONE")
