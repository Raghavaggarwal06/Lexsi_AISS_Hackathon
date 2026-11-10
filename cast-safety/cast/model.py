"""Model loading and prompt formatting, following the conventions of IBM/activation-steering (the CAST release).

Formatting conventions copied from the release (they affect where vectors live, so they are kept on purpose):
  - prompts are rendered with tokenizer.apply_chat_template(..., tokenize=False) and then tokenized with the tokenizer's
    default add_special_tokens=True (so a template that already writes BOS gets a second one, exactly as in the release;
    set repo_bos=False to avoid that);
  - vector-extraction texts use add_generation_prompt=False; generation and condition checks use add_generation_prompt=True;
  - behaviour-vector texts are  <chat(user=q)> + suffix  (suffix appended right after the user turn, no assistant header).
"""
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = Path(os.environ.get("CAST_MODELS", "/scratch/scholar/pande1/models"))

# name -> (local dir, paper's pinned HF repo@revision). Paper Appendix D.
KNOWN = {
    "qwen1.5-1.8b-chat": ("qwen1.5-1.8b-chat", "Qwen/Qwen1.5-1.8B-Chat@e482ee3f73c375a627a16fdf66fd0c8279743ca6"),
    "hermes-2-pro-llama-3-8b": ("hermes-2-pro-llama-3-8b",
                                "NousResearch/Hermes-2-Pro-Llama-3-8B@8ab73a6800796d84448bc936db9bac5ad9f984ae"),
    "gemma-3-270m-it": ("gemma-3-270m-it", "google/gemma-3-270m-it"),
}


class LM:
    def __init__(self, name, device="cuda:0", dtype=torch.float16, repo_bos=True):
        path = MODELS / KNOWN[name][0] if name in KNOWN else Path(name)
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
        self.device = device
        self.repo_bos = repo_bos
        # a pad id that is not EOS, so batched repetition penalty never touches EOS (the release generates one prompt
        # at a time, so it has no padding at all)
        pid = self.tok.pad_token_id
        self.pad_id = pid if (pid is not None and pid != self.tok.eos_token_id) else 0
        self.layers = self._layers()
        self.n_layers = len(self.layers)

    def _layers(self):
        m = self.model
        for path in ("model.layers", "model.language_model.layers", "transformer.h"):
            obj = m
            try:
                for p in path.split("."):
                    obj = getattr(obj, p)
                return obj
            except AttributeError:
                continue
        raise ValueError("cannot find decoder layers")

    def chat(self, user, gen_prompt):
        return self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                            add_generation_prompt=gen_prompt)

    def encode(self, texts, side="right"):
        """Batch-tokenize formatted texts. Returns input_ids, attention_mask on device."""
        ids = [self.tok(t, add_special_tokens=self.repo_bos)["input_ids"] for t in texts]
        n = max(map(len, ids))
        inp = torch.full((len(ids), n), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(ids), n), dtype=torch.long)
        for i, x in enumerate(ids):
            if side == "right":
                inp[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
            else:
                inp[i, n - len(x):] = torch.tensor(x); att[i, n - len(x):] = 1
        return inp.to(self.device), att.to(self.device), [len(x) for x in ids]

    def n_tokens(self, text):
        return len(self.tok(text, add_special_tokens=self.repo_bos)["input_ids"])
