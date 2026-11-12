"""Phase 1: hidden state at the last token of every layer, for every prompt.

Saves results/day1/acts_<set>.npy, shape (N, 37, 2048), fp16, in the JSONL row order.
Index 0 of the 37 is the embedding output; index k is the output of decoder layer k-1.
Finished sets are skipped, so a crashed run can simply be restarted.
"""
import argparse

import numpy as np
import torch
from tqdm import tqdm

from common import DAY1, SETS, encode, load_config, load_model, load_set, load_tokenizer, pick_device, render


def is_oom(err):
    return "out of memory" in str(err).lower()


def clear(device):
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def forward(model, tok, texts, max_len, device):
    """Last-token hidden state per layer. Left padding puts the last real token at -1."""
    enc = encode(tok, texts, max_len, device)
    out = model.model(**enc, output_hidden_states=True, use_cache=False)
    stacked = torch.stack([h[:, -1, :].float().cpu() for h in out.hidden_states], dim=1)
    return stacked.numpy().astype(np.float16)


def extract(model, tok, texts, batch, max_len, device):
    chunks, start = [], 0
    bar = tqdm(total=len(texts), mininterval=20)
    while start < len(texts):
        try:
            chunk = forward(model, tok, texts[start : start + batch], max_len, device)
        except RuntimeError as err:
            if not is_oom(err) or batch == 1:
                raise
            batch = max(1, batch // 2)
            print(f"OOM, batch lowered to {batch}")
            clear(device)
            continue
        chunks.append(chunk)
        start += len(chunk)
        bar.update(len(chunk))
    bar.close()
    return np.concatenate(chunks), batch


def sanity_checks(model, tok, max_len, device):
    """One BOS only, and padding must not change the last-token state."""
    short, long = "Name a colour.", "Explain in two sentences how a bicycle stays upright while moving."
    ids = encode(tok, [render(tok, short)], max_len, device)["input_ids"][0].tolist()
    assert ids[0] == tok.bos_token_id and ids[1] != tok.bos_token_id, "BOS is missing or doubled"
    print(f"template length for a short prompt: {len(ids)} tokens")

    alone = forward(model, tok, [render(tok, short)], max_len, device)[0].astype(np.float32)
    padded = forward(model, tok, [render(tok, short), render(tok, long)], max_len, device)[0].astype(np.float32)
    gap = np.linalg.norm(alone - padded, axis=1) / (np.linalg.norm(alone, axis=1) + 1e-6)
    print(f"padding check, max relative difference over layers: {gap.max():.4f}")
    return float(gap.max())


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=cfg["batch"])
    ap.add_argument("--sets", nargs="*", default=SETS)
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = load_tokenizer(cfg["model"])
    model, dtype = load_model(cfg["model"], device)
    max_len = cfg["max_len"]
    sanity_checks(model, tok, max_len, device)

    batch = args.batch
    for name in args.sets:
        path = DAY1 / f"acts_{name}.npy"
        rows = load_set(name)
        if path.exists() and np.load(path, mmap_mode="r").shape[0] == len(rows):
            print(f"{name}: already done")
            continue
        texts = [render(tok, r["prompt"]) for r in rows]
        acts, batch = extract(model, tok, texts, batch, max_len, device)
        assert acts.shape[1:] == (37, 2048), f"unexpected shape {acts.shape}"
        np.save(path, acts)
        print(f"{name}: {acts.shape}, non-finite values: {int((~np.isfinite(acts)).sum())}")
        clear(device)
    print(f"Done. dtype={dtype}, final batch={batch}")


if __name__ == "__main__":
    main()
