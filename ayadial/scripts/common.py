"""Shared helpers for the strictness-knob experiment. Run scripts from build/."""
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DAY1 = RESULTS / "day1"
SETS = ["harmful", "benign", "xstest_safe", "xstest_unsafe", "orbench_hard"]


def load_config(name="day1"):
    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_set(name):
    return read_jsonl(DATA / f"{name}.jsonl")


def load_splits():
    with open(DATA / "splits.json") as f:
        return json.load(f)


def pick_device(requested=None):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer(model_id):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    tok.truncation_side = "left"  # keeps the user prompt and the generation marker
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_id, device):
    """Load the CausalLM in bf16, falling back to fp16 if a forward pass fails."""
    from transformers import AutoModelForCausalLM

    for dtype in (torch.bfloat16, torch.float16):
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
            with torch.no_grad():
                model.model(input_ids=torch.tensor([[5, 6, 7]], device=device))
            print(f"loaded {model_id} on {device} as {dtype}")
            return model, dtype
        except (RuntimeError, TypeError, NotImplementedError) as err:
            print(f"{dtype} failed on {device}: {err}")
    raise RuntimeError(f"could not run {model_id} on {device}; try --device cpu")


def render(tok, prompt):
    """Chat-template text for one user prompt. The template already carries BOS."""
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def encode(tok, texts, max_len, device):
    """Left-pad and left-truncate. add_special_tokens=False avoids a second BOS."""
    return tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    ).to(device)
