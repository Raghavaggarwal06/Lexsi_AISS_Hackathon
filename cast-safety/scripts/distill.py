"""Compile CAST into a LoRA: the plain model + adapter should behave like the model + CAST hooks.

teacher  CAST-steered model (sec4 vectors + grid point) answers the Sec 4 TRAIN prompts (4,050 harmful + unique
         Alpaca harmless); we keep its token ids and its gate decision per prompt.
train    student = same weights + LoRA (all decoder layers, q/k/v/o/gate/up/down), no hooks. Loss = token-level
         KL(teacher || student) on the teacher's response tokens, where the teacher is the adapter-free model with
         the refusal vector added (at every position, as during CAST generation) iff its gate fired on that prompt.
         On prompts where the gate did not fire the teacher is the unmodified model, so the adapter is also
         trained to leave those alone.
eval     Sec 4 TEST (450 harmful / 500 harmless): base, CAST hooks, LoRA; refusal rates, per-prompt agreement
         with CAST, identical responses.
filler   harmless test prompts with k neutral sentences prepended (k = 0, 2, 8): does the LoRA inherit the gate's
         length bias?
All stages use one dtype (default bf16, since training needs it), so base / CAST / LoRA are compared like for like.
Outputs: runs/<model>/lora_<tag>/.
"""
import argparse, json, math, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cast import data, vectors, steer, lora
from cast.model import LM
from cast.judge import RefusalJudge
from sec4 import PAPER, log
from length_gate import FILLER

DT = {"bf16": torch.bfloat16, "fp16": torch.float16}
# neutral sentences for training-time length augmentation; disjoint from length_gate.FILLER (used at test time)
TRAIN_FILLER = ["Hello there.", "I have been wondering about something.", "Could you help me out?",
                "This is for a project I am working on.", "I appreciate any help you can give.",
                "Let me explain what I need.", "My friend suggested I ask you.", "I want to understand this better.",
                "Please be as clear as you can.", "It is a rainy afternoon here.", "I just got back from work.",
                "Hopefully this makes sense."]


def alpaca_extra(n, exclude, seed=0):
    """n more harmless Alpaca instructions (yahma/alpaca-cleaned, no-input items), none in `exclude`."""
    from datasets import load_dataset
    d = load_dataset("yahma/alpaca-cleaned", split="train")
    pool = [x["instruction"].strip() for x in d if not x["input"].strip()]
    pool = [p for p in dict.fromkeys(pool) if p not in exclude and 10 <= len(p) <= 300]
    random.Random(seed).shuffle(pool)
    return pool[:n]


class RowSteer:
    """Adds alpha * v_l at the input of each behaviour layer, at every position, in the rows where self.mask is
    True: the CAST teacher with its gate decision fixed per sequence."""
    def __init__(self, lm, beh):
        self.lm, self.mask, self.handles = lm, None, []
        self.vec = {l: torch.tensor(beh.alpha * beh.vecs[l], device=lm.device, dtype=torch.float32) for l in beh.layers}

    def __enter__(self):
        for l in self.vec:
            self.handles.append(self.lm.layers[l].register_forward_pre_hook(self._hook(l), with_kwargs=True))
        return self

    def __exit__(self, *a):
        for h in self.handles: h.remove()
        self.handles = []

    def _hook(self, l):
        def fn(mod, args, kwargs):
            if self.mask is None or not bool(self.mask.any()):
                return None
            h = args[0] if args else kwargs["hidden_states"]
            h = h + self.mask.to(h.dtype)[:, None, None] * self.vec[l].to(h.dtype)[None, None, :]
            return ((h,) + tuple(args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": h})
        return fn


def cast_parts(sec4dir, model, nL):
    bv = vectors.load(sec4dir / "behavior_refusal.npz")
    cv = vectors.load(sec4dir / "condition_harmful.npz")
    cp = json.loads((sec4dir / "condition_point.json").read_text())
    P = PAPER.get(model)
    blayers, alpha = P["beh"] if P else (list(range(nL // 3, 2 * nL // 3)), 1.0)
    return steer.Behavior(bv, blayers, alpha), steer.Condition(cv, [cp["layer"]], cp["threshold"], cp["cmp"])


def batches(rows, bs, shuffle, seed=0):
    idx = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    return [idx[i:i + bs] for i in range(0, len(idx), bs)]


def collate(lm, rows):
    seqs = [r["pids"] + r["ids"] for r in rows]
    n = max(map(len, seqs))
    inp = torch.full((len(seqs), n), lm.pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    pos_b, pos_t = [], []
    for b, (s, r) in enumerate(zip(seqs, rows)):
        inp[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1
        P, R = len(r["pids"]), len(r["ids"])
        pos_b += [b] * R; pos_t += list(range(P - 1, P + R - 1))
    met = torch.tensor([bool(r["met"]) for r in rows])
    return inp.to(lm.device), att.to(lm.device), torch.tensor(pos_b), torch.tensor(pos_t), met.to(lm.device)


def logprobs(lm, inp, att, pb, pt):
    h = lm.model.model(input_ids=inp, attention_mask=att).last_hidden_state
    return F.log_softmax(lm.model.lm_head(h[pb, pt]).float(), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen1.5-1.8b-chat")
    ap.add_argument("--stages", default="teacher,train,eval,filler")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--debug", type=int, default=0, help="smoke test: N prompts per class everywhere")
    ap.add_argument("--augment", action="store_true",
                    help="also train on extra harmless Alpaca prompts + filler-padded prompts whose teacher decision "
                         "is the gate's decision on the UNPADDED prompt (length-invariant CAST)")
    ap.add_argument("--n-extra", type=int, default=3300)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing in the student pass")
    ap.add_argument("--n-filler", type=int, default=1500, help="filler variants per class")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1] / "runs" / a.model
    out = root / f"lora_{a.tag or f'r{a.rank}'}"
    out.mkdir(parents=True, exist_ok=True)
    L = out / "log.txt"
    stages = a.stages.split(",")
    lm = LM(a.model, device=a.device, dtype=DT[a.dtype])
    nL = lm.n_layers
    beh, cond = cast_parts(root / "sec4", a.model, nL)
    log(f"{a.model} {a.dtype}: stages {stages}; CAST gate L{cond.layers} sim {cond.cmp} {cond.threshold}, "
        f"behaviour L{beh.layers[0]}-{beh.layers[-1]} alpha {beh.alpha}", L)
    tfile = out.parent / f"distill_teacher_{a.dtype}_{a.max_new}{f'_debug{a.debug}' if a.debug else ''}.jsonl"

    if "teacher" in stages and not tfile.exists():
        t = time.time()
        hp, hn = data.condition_harmful()
        hp, hn = list(dict.fromkeys(hp)), list(dict.fromkeys(hn))
        if a.debug:
            hp, hn = hp[:a.debug], hn[:a.debug]
        prompts, y = hp + hn, [1] * len(hp) + [0] * len(hn)
        s = steer.Steerer(lm, [cond], [beh])
        with s:
            resp, info, ids = steer.generate(lm, prompts, s, max_new_tokens=a.max_new, batch_size=a.gen_batch,
                                             return_ids=True)
        with open(tfile, "w") as fh:
            for p, yy, r, i, d in zip(prompts, y, resp, info, ids):
                fh.write(json.dumps({"prompt": p, "harmful": yy, "met": i["met"][0], "response": r, "ids": d}) + "\n")
        met = np.array([i["met"][0] for i in info]); yy = np.array(y)
        log(f"teacher: {len(prompts)} prompts ({len(hp)} harmful / {len(hn)} harmless) {time.time()-t:.0f}s; "
            f"gate fired {100*met[yy==1].mean():.1f}% / {100*met[yy==0].mean():.1f}%", L)

    afile = out.parent / f"distill_teacher_aug_{a.dtype}_{a.max_new}.jsonl"
    if a.augment and "teacher" in stages and not afile.exists():
        t = time.time()
        base_rows = [json.loads(x) for x in open(tfile)]
        H, A = data.sec4_test()
        excl = {r["prompt"] for r in base_rows} | set(H) | set(A) | {p for v in data.multiple("test").values() for p in v}
        extra = alpaca_extra(a.n_extra, excl)
        s = steer.Steerer(lm, [cond], [beh])
        with s:
            resp, info, ids = steer.generate(lm, extra, s, max_new_tokens=a.max_new, batch_size=a.gen_batch,
                                             return_ids=True)
        rows = [{"prompt": p, "harmful": 0, "met": i["met"][0], "response": r, "ids": d, "kind": "extra"}
                for p, r, i, d in zip(extra, resp, info, ids)]
        rng = random.Random(1)
        pick = rng.sample([r for r in base_rows if r["harmful"]], a.n_filler) + \
            rng.sample([r for r in base_rows + rows if not r["harmful"]], a.n_filler)
        var = []
        for r in pick:
            f = " ".join(rng.sample(TRAIN_FILLER, rng.choice([1, 2, 4, 8])))
            q = f + " " + r["prompt"] if rng.random() < 0.5 else r["prompt"] + " " + f
            var.append({"prompt": q, "harmful": r["harmful"], "met": r["met"], "kind": "filler", "orig": r["prompt"]})
        for flag in (True, False):            # teacher with the gate decision of the unpadded prompt
            grp = [v for v in var if v["met"] == flag]
            st = steer.Steerer(lm, behaviors=[beh]) if flag else None
            if st is not None:
                with st:
                    resp, _, ids = steer.generate(lm, [v["prompt"] for v in grp], st, max_new_tokens=a.max_new,
                                                  batch_size=a.gen_batch, return_ids=True)
            else:
                resp, _, ids = steer.generate(lm, [v["prompt"] for v in grp], None, max_new_tokens=a.max_new,
                                              batch_size=a.gen_batch, return_ids=True)
            for v, r, d in zip(grp, resp, ids):
                v["response"], v["ids"] = r, d
        with open(afile, "w") as fh:
            for r in rows + var:
                fh.write(json.dumps(r) + "\n")
        m = np.array([r["met"] for r in rows])
        log(f"augment teacher: {len(rows)} extra harmless (gate fired {100*m.mean():.1f}%), {len(var)} filler "
            f"variants with the unpadded decision; {time.time()-t:.0f}s", L)

    ad = lora.add_lora(lm, a.rank, a.alpha)
    nparam = sum(p.numel() for p in lora.params(ad))

    if "train" in stages:
        rows = [json.loads(x) for x in open(tfile)]
        if a.augment:
            rows += [json.loads(x) for x in open(afile)]
        for r in rows:
            r["pids"] = lm.tok(lm.chat(r["prompt"], True), add_special_tokens=lm.repo_bos)["input_ids"]
        rows = [r for r in rows if r["ids"]]
        random.Random(0).shuffle(rows)
        nv = 256 if not a.debug else max(4, len(rows) // 5)
        val, tr = rows[:nv], rows[nv:]
        opt = torch.optim.AdamW(lora.params(ad), lr=a.lr, weight_decay=0.0)
        nb = math.ceil(len(tr) / a.bs) * a.epochs
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(s, nb) / nb)))
        rs = RowSteer(lm, beh)
        log(f"train: {len(tr)} seqs, val {len(val)}, LoRA r{a.rank} alpha {a.alpha} on {len(ad)} linears "
            f"({nparam/1e6:.1f}M params), lr {a.lr}, {a.epochs} epochs x {nb // a.epochs} steps, bs {a.bs}", L)

        def kl_batch(rs_rows, grad):
            inp, att, pb, pt, met = collate(lm, rs_rows)
            with torch.no_grad():
                lora.set_lora(ad, False)
                rs.mask = met
                with rs:
                    lp_t = logprobs(lm, inp, att, pb, pt)
                rs.mask = None
                lora.set_lora(ad, True)
            with torch.set_grad_enabled(grad):
                lq = logprobs(lm, inp, att, pb, pt)
                kl = (lp_t.exp() * (lp_t - lq)).sum(-1)
            mb = met[pb.to(met.device)]
            return kl.mean(), kl.detach()[mb].mean().item() if mb.any() else float("nan"), \
                kl.detach()[~mb].mean().item() if (~mb).any() else float("nan")

        def validate():
            ks = [kl_batch([val[i] for i in b], False) for b in batches(val, a.bs, False)]
            return np.mean([k[0].item() for k in ks]), np.nanmean([k[1] for k in ks]), np.nanmean([k[2] for k in ks])

        if a.grad_ckpt:
            lm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            lm.model.train()          # checkpointing is active only in train mode (no dropout in these models)
        v = validate()
        log(f"val KL before training {v[0]:.4f} (gate on {v[1]:.4f} / off {v[2]:.4f})", L)
        step, t = 0, time.time()
        hist = []
        for ep in range(a.epochs):
            run = []
            for b in batches(tr, a.bs, True, seed=ep):
                loss, kon, koff = kl_batch([tr[i] for i in b], True)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora.params(ad), 1.0)
                opt.step(); sched.step(); step += 1
                run.append((loss.item(), kon, koff))
                if step % 50 == 0:
                    r = np.array(run[-50:])
                    log(f"step {step}/{nb} KL {r[:,0].mean():.4f} (gate on {np.nanmean(r[:,1]):.4f} / off "
                        f"{np.nanmean(r[:,2]):.4f}) lr {sched.get_last_lr()[0]:.2e} {time.time()-t:.0f}s", L)
            v = validate()
            hist.append({"epoch": ep + 1, "val_kl": float(v[0]), "val_kl_gate_on": float(v[1]), "val_kl_gate_off": float(v[2])})
            log(f"epoch {ep+1}: val KL {v[0]:.4f} (gate on {v[1]:.4f} / off {v[2]:.4f})", L)
        if a.grad_ckpt:
            lm.model.gradient_checkpointing_disable(); lm.model.eval()
        lora.save(ad, out / "lora.pt", {"model": a.model, "rank": a.rank, "alpha": a.alpha, "lr": a.lr,
                                        "epochs": a.epochs, "bs": a.bs, "dtype": a.dtype, "n_params": nparam,
                                        "teacher": str(tfile.name), "augment": a.augment, "grad_ckpt": a.grad_ckpt,
                                        "n_train": len(tr), "history": hist})

    if "eval" in stages or "filler" in stages or "filler_harmful" in stages:
        lora.load(ad, out / "lora.pt")
        judge = RefusalJudge(a.device)
        summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}

        def gen(prompts, setting):
            lora.set_lora(ad, setting == "lora")
            if setting == "cast":
                s = steer.Steerer(lm, [cond], [beh])
                with s:
                    r, info = steer.generate(lm, prompts, s, batch_size=a.gen_batch)
            else:
                r, info = steer.generate(lm, prompts, None, batch_size=a.gen_batch)
            lora.set_lora(ad, True)
            j = judge(r)
            return r, np.array([x["refusal"] for x in j]), info

    if "eval" in stages:
        H, A = data.sec4_test()
        if a.debug:
            H, A = H[:a.debug], A[:a.debug]
        prompts, y = H + A, np.array([1] * len(H) + [0] * len(A))
        R, F_ = {}, {}
        for setting in ("base", "cast", "lora"):
            t = time.time()
            R[setting], F_[setting], info = gen(prompts, setting)
            summary[setting] = {"harmful_refusal": 100 * F_[setting][y == 1].mean(),
                                "harmless_refusal": 100 * F_[setting][y == 0].mean()}
            if setting == "cast":
                met = np.array([i["met"][0] for i in info])
                summary["cast"]["gate_met"] = [100 * met[y == 1].mean(), 100 * met[y == 0].mean()]
            with open(out / f"test_{setting}.jsonl", "w") as fh:
                for p, yy, r, f in zip(prompts, y, R[setting], F_[setting]):
                    fh.write(json.dumps({"prompt": p, "harmful": int(yy), "response": r, "refusal": bool(f)}) + "\n")
            log(f"test {setting}: harmful {summary[setting]['harmful_refusal']:.1f}% harmless "
                f"{summary[setting]['harmless_refusal']:.1f}% [{time.time()-t:.0f}s]", L)
        agree = F_["lora"] == F_["cast"]
        same = np.array([x == z for x, z in zip(R["lora"], R["cast"])])
        summary["agreement_lora_vs_cast"] = {"refusal_agree_harmful": 100 * agree[y == 1].mean(),
                                             "refusal_agree_harmless": 100 * agree[y == 0].mean(),
                                             "identical_response_harmful": 100 * same[y == 1].mean(),
                                             "identical_response_harmless": 100 * same[y == 0].mean()}
        log("LoRA vs CAST: " + json.dumps({k: round(v, 1) for k, v in summary["agreement_lora_vs_cast"].items()}), L)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))

    if "filler" in stages:
        _, A = data.sec4_test()
        if a.debug:
            A = A[:a.debug]
        summary["filler"] = {}
        for k in (0, 2, 8):
            q = A if k == 0 else [" ".join(FILLER[:k]) + " " + p for p in A]
            row = {}
            for setting in ("cast", "lora"):
                _, f, info = gen(q, setting)
                row[setting] = 100 * f.mean()
                if setting == "cast":
                    row["cast_gate_met"] = 100 * np.mean([i["met"][0] for i in info])
            summary["filler"][k] = row
            log(f"filler k={k}: harmless refusal CAST {row['cast']:.1f}% (gate {row['cast_gate_met']:.1f}%) "
                f"LoRA {row['lora']:.1f}%", L)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))

    if "filler_harmful" in stages:
        # safety check for the augmented adapter: filler must not become a way around refusal
        H, _ = data.sec4_test()
        if a.debug:
            H = H[:a.debug]
        summary["filler_harmful"] = {}
        for k in (0, 8):
            q = H if k == 0 else [" ".join(FILLER[:k]) + " " + p for p in H]
            row = {}
            for setting in ("base", "cast", "lora"):
                _, f, _ = gen(q, setting)
                row[setting] = 100 * f.mean()
            summary["filler_harmful"][k] = row
            log(f"harmful + filler k={k}: refusal base {row['base']:.1f}% CAST {row['cast']:.1f}% LoRA {row['lora']:.1f}%", L)
        # refusal on the test set by prompt length (from the eval stage's responses)
        bins = [0, 30, 40, 60, 1000]
        summary["by_length"] = {}
        for setting in ("base", "cast", "lora"):
            rows = [json.loads(x) for x in open(out / f"test_{setting}.jsonl")]
            n = np.array([lm.n_tokens(lm.chat(r["prompt"], True)) for r in rows])
            yy = np.array([r["harmful"] for r in rows]); ff = np.array([r["refusal"] for r in rows])
            summary["by_length"][setting] = {f"{lo}-{hi}": {"n": int(((yy == 1) & (n >= lo) & (n < hi)).sum()),
                                                            "harmful_refusal": float(100 * ff[(yy == 1) & (n >= lo) & (n < hi)].mean())}
                                             for lo, hi in zip(bins[:-1], bins[1:]) if ((yy == 1) & (n >= lo) & (n < hi)).sum()}
        log("harmful refusal by prompt length: " + json.dumps({s_: {b: round(v["harmful_refusal"], 1) for b, v in d.items()}
                                                              for s_, d in summary["by_length"].items()}), L)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
