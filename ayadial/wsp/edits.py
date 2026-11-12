"""Edit format and validation, shared by the sandboxed kernel (fast feedback to the editor) and the driver (the check that counts).

An edit lives only in layers[22].mlp and only on whole units. For unit j:
  gate_proj row j and up_proj row j (each 2048 numbers) and down_proj column j (2048 numbers).
Each edited tensor carries two parts:
  install  the values of those rows (or columns) at every knob scale s;
  knob     the part multiplied by s, so the weight at scale s is  install + s * knob.
At s = 0 only the install part acts, and it may differ from the original model only on the allowed units.

Compact form of one tensor's edit (an .npz): name (str), units (int64 [k]), install (float32 [k, 2048]), knob (float32 [k, 2048]).
Units whose install equals the original and whose knob is zero are dropped, so the compact form lists exactly the units that differ.
Pure numpy; no torch import, so the driver can use it without loading torch.
"""
import hashlib
import io
import json

import numpy as np

LAYER = 22
EDITABLE = {f"model.layers.{LAYER}.mlp.{t}.weight": t for t in ("gate_proj", "up_proj", "down_proj")}
SHORT = {v: k for k, v in EDITABLE.items()}
HIDDEN, INTER = 2048, 11008


class EditRefused(ValueError):
    pass


def unit_axis(name):
    """gate/up: units are rows (axis 0); down: units are columns (axis 1)."""
    return 1 if EDITABLE[name] == "down_proj" else 0


def take_units(name, W, units):
    """[k, 2048] float32 slice of the full tensor W for the given units."""
    W = np.asarray(W)
    units = np.asarray(units, dtype=np.int64)
    out = W[:, units].T if unit_axis(name) == 1 else W[units]
    return np.ascontiguousarray(out, dtype=np.float32)


def changed_units(name, new, orig):
    """Units where the full tensor `new` differs from `orig` in at least one element (exact float32 comparison)."""
    new = np.asarray(new, dtype=np.float32)
    orig = np.asarray(orig, dtype=np.float32)
    if new.shape != orig.shape:
        raise EditRefused(f"{name}: shape {tuple(new.shape)} differs from the original {tuple(orig.shape)}")
    ax = unit_axis(name)
    diff = new != orig
    return np.nonzero(diff.any(axis=1 - ax))[0].astype(np.int64)


def nonzero_units(name, knob):
    knob = np.asarray(knob, dtype=np.float32)
    ax = unit_axis(name)
    return np.nonzero((knob != 0).any(axis=1 - ax))[0].astype(np.int64)


def compact_from_full(name, install_full, knob_full, orig_full):
    """Build the compact form from full tensors. knob_full may be None (no knob part)."""
    if name not in EDITABLE:
        raise EditRefused(f"{name} is not editable; only {sorted(EDITABLE)} are")
    install_full = np.asarray(install_full, dtype=np.float32)
    if not np.isfinite(install_full).all():
        raise EditRefused(f"{name}: the tensor has non-finite values (nan or inf)")
    units = set(changed_units(name, install_full, orig_full).tolist())
    if knob_full is not None:
        knob_full = np.asarray(knob_full, dtype=np.float32)
        if knob_full.shape != install_full.shape:
            raise EditRefused(f"{name}: knob shape {tuple(knob_full.shape)} differs from {tuple(install_full.shape)}")
        if not np.isfinite(knob_full).all():
            raise EditRefused(f"{name}: the knob part has non-finite values")
        units |= set(nonzero_units(name, knob_full).tolist())
    units = np.array(sorted(units), dtype=np.int64)
    install = take_units(name, install_full, units)
    knob = take_units(name, knob_full, units) if knob_full is not None else np.zeros_like(install)
    return {"name": name, "units": units, "install": install, "knob": knob}


def canonical(c, orig_rows):
    """Drop units whose install equals the original rows and whose knob is zero. orig_rows: [k, 2048] for c['units']."""
    keep = [(i, u) for i, u in enumerate(c["units"].tolist())
            if (c["install"][i] != orig_rows[i]).any() or (c["knob"][i] != 0).any()]
    idx = [i for i, _ in keep]
    return {"name": c["name"], "units": np.array([u for _, u in keep], dtype=np.int64),
            "install": c["install"][idx].astype(np.float32), "knob": c["knob"][idx].astype(np.float32)}


def validate(c, rules, other_units=()):
    """All reasons to refuse one tensor's compact edit; [] means accepted.
    rules: {allowed_units, max_units, knob_tensors, max_row_norm{short: x}, max_knob_row_norm}.
    other_units: units edited in the OTHER tensors of the same edit state (they count toward max_units)."""
    errs = []
    name = c.get("name")
    if name not in EDITABLE:
        return [f"{name!r} is not editable; only layers[{LAYER}].mlp gate_proj, up_proj and down_proj are"]
    short = EDITABLE[name]
    units = np.asarray(c.get("units"))
    install = np.asarray(c.get("install"))
    knob = np.asarray(c.get("knob"))
    if units.ndim != 1 or (units.size and not np.issubdtype(units.dtype, np.integer)):
        return [f"{short}: units must be a 1-d integer array"]
    k = units.size
    if len(set(units.tolist())) != k:
        errs.append(f"{short}: a unit is listed twice")
    if install.shape != (k, HIDDEN) or knob.shape != (k, HIDDEN):
        return errs + [f"{short}: install and knob must have shape ({k}, {HIDDEN}); got {install.shape} and {knob.shape}"]
    if not (np.isfinite(install).all() and np.isfinite(knob).all()):
        errs.append(f"{short}: non-finite values")
    allowed = set(int(u) for u in rules["allowed_units"])
    outside = sorted(set(int(u) for u in units.tolist()) - allowed)
    if outside:
        shown = outside[:12]
        errs.append(f"{short}: units {shown}{' ...' if len(outside) > 12 else ''} ({len(outside)} in all) are not among the allowed units "
                    f"{sorted(allowed)}; an edit may change the model only on allowed units")
    if short not in rules["knob_tensors"] and (knob != 0).any():
        errs.append(f"{short}: a knob part (multiplied by s) is allowed only on {rules['knob_tensors']}")
    if k:
        norms = np.linalg.norm(install.astype(np.float64), axis=1)
        cap = rules["max_row_norm"][short]
        bad = [(int(u), round(float(n), 2)) for u, n in zip(units.tolist(), norms) if n > cap]
        if bad:
            errs.append(f"{short}: install norm above the cap {cap} for (unit, norm) {bad[:6]}")
        kn = np.linalg.norm(knob.astype(np.float64), axis=1)
        kbad = [(int(u), round(float(n), 2)) for u, n in zip(units.tolist(), kn) if n > rules["max_knob_row_norm"]]
        if kbad:
            errs.append(f"{short}: knob norm above the cap {rules['max_knob_row_norm']} for (unit, norm) {kbad[:6]}")
    total = set(int(u) for u in units.tolist()) | set(int(u) for u in other_units)
    if len(total) > rules["max_units"]:
        errs.append(f"the edit would change {len(total)} distinct units across gate, up and down; the limit is {rules['max_units']}")
    return errs


def apply_full(name, orig_full, c, scale=1.0):
    """The full tensor at knob scale s: original with the edited units replaced by install + s * knob."""
    W = np.array(orig_full, dtype=np.float32, copy=True)
    if c is None or len(c["units"]) == 0:
        return W
    vals = c["install"] + np.float32(scale) * c["knob"]
    if unit_axis(name) == 1:
        W[:, c["units"]] = vals.T
    else:
        W[c["units"]] = vals
    return W


def state_units(state):
    """{short name: [units]} and the union, for an edit state {name: compact}."""
    per = {EDITABLE[n]: sorted(int(u) for u in c["units"].tolist()) for n, c in state.items() if c is not None}
    union = sorted(set(u for us in per.values() for u in us))
    return per, union


def state_hash(state):
    """sha256 over the canonical bytes of an edit state (names sorted, float32 little-endian). '' for no edit."""
    h = hashlib.sha256()
    for name in sorted(state):
        c = state[name]
        if c is None or len(c["units"]) == 0:
            continue
        order = np.argsort(c["units"])
        h.update(name.encode())
        h.update(np.asarray(c["units"], dtype="<i8")[order].tobytes())
        h.update(np.asarray(c["install"], dtype="<f4")[order].tobytes())
        h.update(np.asarray(c["knob"], dtype="<f4")[order].tobytes())
    return h.hexdigest()[:16]


def save_npz(path, c):
    np.savez(path, name=np.array(c["name"]), units=np.asarray(c["units"], dtype=np.int64),
             install=np.asarray(c["install"], dtype=np.float32), knob=np.asarray(c["knob"], dtype=np.float32))


def load_npz(path, max_bytes=64 * 1024 * 1024):
    """Read a compact edit written by untrusted code: size-capped, no pickles, fixed keys only."""
    data = open(path, "rb").read(max_bytes + 1)
    if len(data) > max_bytes:
        raise EditRefused(f"edit file larger than {max_bytes} bytes")
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        keys = set(z.files)
        if not {"name", "units", "install", "knob"} <= keys:
            raise EditRefused(f"edit file lacks keys: has {sorted(keys)}")
        name = str(z["name"])
        units = z["units"]
        if units.ndim != 1:
            raise EditRefused("units must be 1-d")
        return {"name": name, "units": units.astype(np.int64), "install": z["install"].astype(np.float32), "knob": z["knob"].astype(np.float32)}


def describe(state):
    per, union = state_units(state)
    return json.dumps({"units_per_tensor": per, "distinct_units": union, "edit_hash": state_hash(state)})
