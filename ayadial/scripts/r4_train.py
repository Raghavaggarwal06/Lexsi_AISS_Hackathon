"""Trained route: a LoRA on WildJailbreak train rows (fit split only).
  tighten: harmful prompts -> refusal (the dataset's own refusal completion when the base answers, the base's refusal otherwise);
           harmless prompts -> the base model's own reply (keep behaviour).
  python r4_train.py <name> <steps> [rank] [targets: all|mlp|attn]"""
import sys
from r4lib import *
from datasets import load_dataset
name, steps = sys.argv[1], int(sys.argv[2])
rank = int(sys.argv[3]) if len(sys.argv) > 3 else 8
tgt = sys.argv[4] if len(sys.argv) > 4 else "all"
BS, LR, MAXT, ACC, MAXP = 4, 1e-4, 96, 2, 1000  # batch 4 x 2 accumulation = 8 examples per update; prompts over 1000 tokens are skipped (memory)
out_dir = ROOT / "adapters" / name
drive_dir = DRIVE / "adapters" / name
torch.manual_seed(0); random.seed(0)

rows = sets(["trfit_h", "trfit_b", "harmful_fit", "benign_fit", "xstest_safe_fit", "orbench_hard_fit", "xstest_unsafe_fit"])
gen = {rk(r): r for r in read_jsonl(RAW / "gen" / "base_fit.jsonl")}
lab = labels("base_fit")
tr = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)["train"]
GENERIC = "I'm sorry, but I can't help with that request."
ex, stats = [], {"harm_base_refusal": 0, "harm_dataset_refusal": 0, "harm_generic_refusal": 0, "harmless_base_reply": 0}
for r in rows:
    k = rk(r)
    if k not in gen:
        continue
    harmful = r["set"] in ("trfit_h", "harmful_fit", "xstest_unsafe_fit")
    refused, deg = lab[k][0] == 1, lab[k][2] == 1
    if harmful:
        if refused:
            target, kind = gen[k]["response"], "harm_base_refusal"
        elif r["id"].startswith("wjtr-") and tr[int(r["id"][5:])]["completion"].strip():
            target, kind = tr[int(r["id"][5:])]["completion"].strip(), "harm_dataset_refusal"
        else:
            target, kind = GENERIC, "harm_generic_refusal"
    else:
        if deg:
            continue
        target, kind = gen[k]["response"], "harmless_base_reply"
    stats[kind] += 1
    ex.append((r["prompt"], target))
log(f"examples {len(ex)} {stats}")

from peft import LoraConfig, get_peft_model
model, tok = load()
mods = {"all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], "mlp": ["gate_proj", "up_proj", "down_proj"], "attn": ["q_proj", "k_proj", "v_proj", "o_proj"]}[tgt]
pm = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0, target_modules=mods, task_type="CAUSAL_LM"))
pm.print_trainable_parameters()
END = tok.convert_tokens_to_ids("<|END_RESPONSE|>")
opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 30) * max(0.1, 1 - i / steps))
pm.train()
order, pos, losses, skipped = list(range(len(ex))), 0, [], 0
random.shuffle(order)
tok.padding_side = "right"
for step in range(steps):
    if pos + BS > len(order):
        random.shuffle(order); pos = 0
    batch = [ex[i] for i in order[pos : pos + BS]]; pos += BS
    ids, labs = [], []
    for p, t in batch:
        a = tok(render(tok, p), add_special_tokens=False)["input_ids"]
        if len(a) > MAXP:
            skipped += 1
            continue
        b = tok(t, add_special_tokens=False)["input_ids"]
        b = b[:MAXT] + ([END, 3] if len(b) < MAXT else [])
        ids.append(a + b); labs.append([-100] * len(a) + b)
    if not ids:
        continue
    L = max(len(x) for x in ids)
    inp = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x in ids], device=DEV)
    lbl = torch.tensor([x + [-100] * (L - len(x)) for x in labs], device=DEV)
    att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids], device=DEV)
    hid = model.model(input_ids=inp, attention_mask=att, use_cache=False).last_hidden_state  # LoRA layers are injected in place
    sel = lbl[:, 1:] != -100                                                               # logits only where there is a target (vocab is 262k)
    logits = model.lm_head(hid[:, :-1][sel]) * model.logit_scale
    loss = torch.nn.functional.cross_entropy(logits.float(), lbl[:, 1:][sel])
    (loss / ACC).backward()
    if (step + 1) % ACC == 0:
        torch.nn.utils.clip_grad_norm_([p for p in pm.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    sched.step()
    losses.append(float(loss))
    if (step + 1) % 100 == 0:
        log(f"step {step + 1}/{steps} loss {np.mean(losses[-100:]):.4f} skipped_long {skipped}")
    if (step + 1) % 400 == 0 or step + 1 == steps:
        pm.save_pretrained(out_dir)
        subprocess.run(f"mkdir -p {drive_dir} && cp -r {out_dir}/. {drive_dir}/", shell=True)
(OUT / "trained").mkdir(exist_ok=True)
(OUT / "trained" / f"{name}_train.json").write_text(json.dumps({"name": name, "steps": steps, "rank": rank, "targets": mods, "batch": BS, "accumulation": ACC, "updates": steps // ACC, "skipped_prompts_over_1000_tokens": skipped, "lr": LR, "examples": len(ex), "example_kinds": stats,
                                                              "loss_first25": float(np.mean(losses[:25])), "loss_last25": float(np.mean(losses[-25:])), "max_target_tokens": MAXT}, indent=1))
sync()
log("R4_TRAIN DONE")
