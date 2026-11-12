"""J2: void the RefusalDirection pick made with steering forced to layer 22 (no effect at any strength: SafeTune selected its direction at layer 10).
Moves methods.rd out of safetune_pick.json into safetune_pick_rd_layer22_void.json and sets steer_layers to None (every layer, SafeTune's default)
in the saved calibration, so that `r7_safetune.py pick rd` runs again on the new grid. The direction itself is unchanged."""
import json, sys
from pathlib import Path
import torch
sys.path.insert(0, "scripts")
import r7lib as L7
P = L7.OUT / "safetune_pick.json"
d = json.loads(P.read_text())
old = d["methods"].pop("rd", None)
if old is not None and old.get("setting_text", "").endswith("layer 22"):
    (L7.OUT / "safetune_pick_rd_layer22_void.json").write_text(json.dumps({"void": "steering was forced to layer 22; see journal J2", "rd": old}, indent=1))
    P.write_text(json.dumps(d, indent=1))
elif old is not None:
    d["methods"]["rd"] = old  # already the every-layer pick: leave it
sp = L7.RAW / "calib" / "st_rd.pt"
st = torch.load(sp, weights_only=False)
print("layer_selected", st.get("layer_selected"), "steer_layers before", st.get("steer_layers"))
st["steer_layers"] = None
torch.save(st, sp)
L7.sync()
print("R7_FIX_RD DONE", sorted(d["methods"]))
