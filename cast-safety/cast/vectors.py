"""Behaviour / condition vector extraction (paper Sec 3.3; release steering_vector.read_representations).

For every layer l: collect one pooled hidden state per text, centre each (positive, negative) pair on its own mean
("pca_pairwise", the release default that "reproduces results closer to those reported in the paper"), take the first
principal component, and flip its sign so that positives project above negatives in most pairs.

Layer indexing follows the release: the vector for layer l is read from hidden_states[l + 1] (the OUTPUT of decoder
layer l) but is used at the INPUT of decoder layer l (see steer.py). This one-layer offset is in the release and the
paper's reported layer numbers depend on it, so it is kept.

Pooling (two implementations):
  release (DEFAULT, extract_release / pooled_hiddens_release): exact emulation of the release. Texts are batched
    as [p0, n0, p1, n1, ...], right-padded with id 0, and averaged over ALL positions of the padded batch ('all'),
    or over the last k positions, k = token length of the first refusal suffix ('suffix'). Because both members of
    a pair share a batch, the huge first-token (attention-sink) state enters both with the same weight
    1/T_batch and cancels under pair-centring.
  masked (extract / pooled_hiddens): mean over each text's own real tokens or exact suffix tokens. The sink state
    then enters with weight 1/T_own, which differs within a pair, so PC1 becomes a 1/length direction (97-100%
    of the pair-centred variance on Qwen 1.8B / Hermes 8B). The vectors become length detectors; see README.
"""
import numpy as np
import torch
from sklearn.decomposition import PCA


@torch.no_grad()
def pooled_hiddens(lm, texts, layers, pool="all", prefix_lens=None, batch_size=32, store_dtype=np.float16):
    """{layer: (N, d) array} of pooled hidden_states[layer + 1]."""
    out = {l: [] for l in layers}
    order = np.argsort([-len(t) for t in texts])          # length-sorted batches = little padding
    for s in range(0, len(texts), batch_size):
        idx = order[s:s + batch_size]
        ids, att, lens = lm.encode([texts[i] for i in idx], side="right")
        hs = lm.model(input_ids=ids, attention_mask=att, output_hidden_states=True, logits_to_keep=1).hidden_states
        mask = att.clone().float()
        if pool == "suffix":
            for j, i in enumerate(idx):
                mask[j, :prefix_lens[i]] = 0
        elif pool == "last":
            mask.zero_()
            for j, n in enumerate(lens):
                mask[j, n - 1] = 1
        w = (mask / mask.sum(1, keepdim=True)).unsqueeze(-1)
        for l in layers:
            out[l].append(((hs[l + 1].float() * w).sum(1)).cpu().numpy().astype(store_dtype))
        del hs
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    return {l: np.concatenate(v)[inv] for l, v in out.items()}


@torch.no_grad()
def pooled_hiddens_release(lm, texts, layers, pool="all", suffix_len=None, batch_size=32, store_dtype=np.float16):
    """Exact emulation of the release's batched_get_hiddens: batches in the given order, tokenizer(batch,
    padding=True) with pad id 0 and the tokenizer's own padding side, and pooling over the LAST positions of the
    padded tensor ('all' = every position including pads; 'suffix' = the last suffix_len positions; 'last' = the last
    position), whatever is there."""
    tok = lm.tok
    old_pad, tok.pad_token_id = tok.pad_token_id, 0
    out = {l: [] for l in layers}
    try:
        for s in range(0, len(texts), batch_size):
            enc = tok(texts[s:s + batch_size], padding=True, return_tensors="pt",
                      add_special_tokens=lm.repo_bos).to(lm.device)
            hs = lm.model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states
            for l in layers:
                h = hs[l + 1].float()
                if pool == "all":
                    p = h.mean(1)
                elif pool == "suffix":
                    p = h[:, -suffix_len:].mean(1)
                else:
                    p = h[:, -1]
                out[l].append(p.cpu().numpy().astype(store_dtype))
            del hs
    finally:
        tok.pad_token_id = old_pad
    return {l: np.concatenate(v) for l, v in out.items()}


def extract_release(lm, pos_texts, neg_texts, layers, pool, suffix_len=None, batch_size=32):
    """Release order: train_strs = [p0, n0, p1, n1, ...] batched together."""
    inter = [t for pair in zip(pos_texts, neg_texts) for t in pair]
    H = pooled_hiddens_release(lm, inter, layers, pool, suffix_len, batch_size)
    vecs, ev, sep = {}, {}, {}
    for l in layers:
        hp, hn = H[l][0::2], H[l][1::2]
        vecs[l], ev[l] = pca_pairwise(hp, hn)
        sep[l] = float(np.mean(hp.astype(np.float32) @ vecs[l] > hn.astype(np.float32) @ vecs[l]))
    return vecs, ev, sep


def pca_pairwise(h_pos, h_neg):
    """First PC of pair-centred data (release method 'pca_pairwise') + sign fix. Returns (unit vector, explained var)."""
    h_pos, h_neg = h_pos.astype(np.float32), h_neg.astype(np.float32)
    c = (h_pos + h_neg) / 2
    train = np.empty((2 * len(c), c.shape[1]), np.float32)
    train[0::2], train[1::2] = h_pos - c, h_neg - c
    p = PCA(n_components=1, whiten=False).fit(train)
    v = p.components_[0].astype(np.float32)
    pp, pn = h_pos @ v, h_neg @ v
    if np.mean(pp < pn) > np.mean(pp > pn):
        v = -v
    return v, float(p.explained_variance_ratio_[0])


def extract(lm, pos_texts, neg_texts, layers, pool, pos_prefix=None, neg_prefix=None, batch_size=32):
    hp = pooled_hiddens(lm, pos_texts, layers, pool, pos_prefix, batch_size)
    hn = pooled_hiddens(lm, neg_texts, layers, pool, neg_prefix, batch_size)
    vecs, ev, sep = {}, {}, {}
    for l in layers:
        vecs[l], ev[l] = pca_pairwise(hp[l], hn[l])
        sep[l] = float(np.mean(hp[l].astype(np.float32) @ vecs[l] > hn[l].astype(np.float32) @ vecs[l]))
    return vecs, ev, sep


def behavior_vector(lm, questions, suffix_pairs, layers, batch_size=32, release=False):
    """Refusal behaviour vector: every question x every (refusal, compliance) suffix pair; pool over suffix tokens.
    release=True: release pooling (last k padded positions, k = token length of the first refusal suffix)."""
    pos, neg, ppre, npre = [], [], [], []
    base = {q: lm.chat(q, gen_prompt=False) for q in questions}
    blen = {q: lm.n_tokens(base[q]) for q in questions}
    for r, c in suffix_pairs:
        for q in questions:
            pos.append(base[q] + r); neg.append(base[q] + c)
            ppre.append(blen[q]); npre.append(blen[q])
    if release:
        k = len(lm.tok.encode(suffix_pairs[0][0], add_special_tokens=False))
        return extract_release(lm, pos, neg, layers, "suffix", k, batch_size)
    return extract(lm, pos, neg, layers, "suffix", ppre, npre, batch_size)


def condition_vector(lm, pos_prompts, neg_prompts, layers, pool="all", batch_size=32, release=False):
    pos = [lm.chat(p, gen_prompt=False) for p in pos_prompts]
    neg = [lm.chat(p, gen_prompt=False) for p in neg_prompts]
    if release:
        return extract_release(lm, pos, neg, layers, pool, batch_size=batch_size)
    return extract(lm, pos, neg, layers, pool, batch_size=batch_size)


def save(path, vecs, ev=None, sep=None, meta=None):
    np.savez(path, layers=np.array(sorted(vecs)), vecs=np.stack([vecs[l] for l in sorted(vecs)]),
             ev=np.array([ev[l] for l in sorted(vecs)]) if ev else np.zeros(len(vecs)),
             sep=np.array([sep[l] for l in sorted(vecs)]) if sep else np.zeros(len(vecs)),
             meta=np.array(str(meta or {})))


def load(path):
    z = np.load(path)
    return {int(l): z["vecs"][i] for i, l in enumerate(z["layers"])}
