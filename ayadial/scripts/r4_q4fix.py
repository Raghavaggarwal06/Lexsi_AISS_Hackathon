
import json, subprocess, time, urllib.request
lc = "/content/llama.cpp"
def post(path, body, port):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())
out = {}
for tag, extra in (("default", ""), ("override_eos_3", "--override-kv tokenizer.ggml.eos_token_id=int:3")):
    srv = subprocess.Popen(f"./build/bin/llama-server -m /content/base_q4km.gguf --port 8103 -c 4096 -np 1 --jinja -t 8 {extra}", shell=True, cwd=lc, stdout=open(f"/content/srv_fix_{tag}.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:8103/health", timeout=2); break
        except Exception:
            time.sleep(2)
    res = []
    for p in ["What is the capital of France?", "Give me two tips for sleeping better.", "Translate 'good morning' into Spanish.", "Name three primary colours.", "What is 12 times 12?"]:
        c = post("/v1/chat/completions", {"messages": [{"role": "user", "content": p}], "max_tokens": 200, "temperature": 0}, 8103)
        txt = c["choices"][0]["message"]["content"]
        res.append({"finish": c["choices"][0]["finish_reason"], "tokens": c["usage"]["completion_tokens"], "content_has_special_text": ("<|END" in txt or "<EOS" in txt), "chars": len(txt)})
    out[tag] = res
    srv.terminate(); time.sleep(4)
out["cause"] = "The model ends a turn with <|END_RESPONSE|> (261001) then <EOS_TOKEN> (3). HF generation_config has eos_token_id 3. convert_hf_to_gguf writes tokenizer.ggml.eos_token_id = 6 (<|END_OF_TURN_TOKEN|>, the tokenizer's eos_token), which the model never emits, so llama.cpp never stops. f16 and Q4_K_M behave the same."
json.dump(out, open("/content/build/results/round4/q4/stop_fix.json", "w"), indent=1)
print(json.dumps(out, indent=1)); print("R4_Q4FIX DONE")
