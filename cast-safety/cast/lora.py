"""Minimal LoRA (Hu et al. 2021) for the decoder's linear layers: W x + (alpha / r) B A x, with A, B in fp32 and B = 0
at init, so a fresh adapter is the identity. set_lora(model, False) switches every adapter off; the same weights then
act as the unmodified model (used as the distillation teacher)."""
import math

import torch
import torch.nn as nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.r, self.scale, self.enabled = base, r, alpha / r, True
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale   # fp32 params, compute in x's dtype
        return y


def add_lora(lm, r=16, alpha=32, targets=TARGETS, layers=None):
    """Wrap target linears in the chosen decoder layers. Returns {name: LoRALinear}."""
    for p in lm.model.parameters():
        p.requires_grad_(False)
    out = {}
    for li, layer in enumerate(lm.layers):
        if layers is not None and li not in layers:
            continue
        for name, mod in list(layer.named_modules()):
            leaf = name.split(".")[-1]
            if leaf in targets and isinstance(mod, nn.Linear):
                parent = layer.get_submodule(name.rsplit(".", 1)[0]) if "." in name else layer
                w = LoRALinear(mod, r, alpha)
                setattr(parent, leaf, w)
                out[f"{li}.{name}"] = w
    return out


def set_lora(adapters, enabled):
    for m in adapters.values():
        m.enabled = enabled


def params(adapters):
    return [p for m in adapters.values() for p in (m.A, m.B)]


def save(adapters, path, meta):
    torch.save({"meta": meta, "weights": {k: (m.A.detach().cpu(), m.B.detach().cpu()) for k, m in adapters.items()}},
               path)


def load(adapters, path):
    d = torch.load(path, map_location="cpu")
    for k, (A, B) in d["weights"].items():
        adapters[k].A.data.copy_(A); adapters[k].B.data.copy_(B)
    return d["meta"]
