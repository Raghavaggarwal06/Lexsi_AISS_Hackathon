"""Round 6 shared library (runs on the Colab runtime): round 5's code pointed at the round 6 folders.
Import this BEFORE any r4/r5 module. Round 5's results are never written to: its cells, labels, caches and vectors are COPIED into the round 6
folders by scripts/r6_boot.sh, and every new file goes under results/round6, runs_private/round6, cache_r6 and Drive lexsi-knob/round6.
The round 5 frozen slices (suite frz5) are closed: this module refuses to build that suite, so no new cell can be generated on them."""
import r5lib
from r5lib import *

OUT, RAW, CACHE = ROOT / "results/round6", ROOT / "runs_private/round6", ROOT / "cache_r6"
DRIVE = Path("/content/drive/MyDrive/lexsi-knob/round6")
for p in (OUT, RAW, CACHE, RAW / "gen", RAW / "suites", OUT / "labels"):
    p.mkdir(parents=True, exist_ok=True)
for mod in (r4lib, r5lib):
    mod.OUT, mod.RAW, mod.CACHE, mod.DRIVE = OUT, RAW, CACHE, DRIVE


def sync():
    subprocess.run(f"mkdir -p {DRIVE}/results {DRIVE}/runs_private {DRIVE}/scripts {DRIVE}/cache; cp -ru {OUT}/. {DRIVE}/results/; cp -ru {RAW}/. {DRIVE}/runs_private/; "
                   f"cp -u {ROOT}/scripts/r6*.py {DRIVE}/scripts/; cp -u {CACHE}/vec_*.npz {DRIVE}/cache/ 2>/dev/null; cp -ru {ROOT}/specs6 {DRIVE}/ 2>/dev/null", shell=True)


r4lib.sync = r5lib.sync = sync
SUITE_ENV.update({"xb6": (70000, 160), "xd6": (70000, 160), "wjx8": (8000, 8), "xbf_xa_itwsys": (12000, 16)})  # exam B, its development slices, the scoreboard rows in batches of 8 (CAST),
# and the exam format with an in-the-wild template as the developer preamble: the batch heuristic counts the request only, the first run ran out of memory before any row was written (journal J11)
_suite_rows5 = r5lib.suite_rows


def suite_rows(suite):
    if suite == "frz5":
        raise RuntimeError("the round 5 frozen test is closed: no new cell may be generated on it")
    allr = sets()
    if suite == "xb6":  # exam B: every round 6 frozen row
        return [r for r in allr if r["set"].startswith("xb_")]
    if suite == "xd6":  # round 6 development rows
        return [r for r in allr if r["set"].startswith("xd_")]
    if suite == "wjx8":
        return [r for r in allr if r["set"] in WJ_H + WJ_B]
    rows = _suite_rows5(suite)
    return [r for r in rows if not r["set"].startswith(("xb_", "xd_"))]


r5lib.suite_rows = suite_rows


def run(model, tok, suite, name, ctx=None, rows=None, max_new=128):
    rows = suite_rows(suite) if rows is None else rows
    RAWMAP.clear()  # one format suite at a time: the map is keyed by request text, and the same request appears in several formats
    RAWMAP.update({r["prompt"]: r["raw"] for r in rows if r.get("raw")})  # rows that carry their own rendered chat text (formats)
    tb, bs = SUITE_ENV.get(suite, DEFAULT_ENV)
    os.environ["R4_TOK"], os.environ["R4_BATCH"] = str(tb), str(bs)
    return run_cell(model, tok, cname(name), rows, ctx, max_new=int(os.environ.get("R5_MAXNEW", max_new)))


r5lib.run = run


# --- weight programming, experiment A: the compiled position flag (scripts/wp_flag.py)
class Flagged:
    """A cell's units together with the compiled position flag: while the cell runs, the embedding row of the reply-start token carries `delta`
    (cache_r6/wp_flag.npz), and the layer-22 units read the flag through phi instead of a fitted marker direction."""
    def __init__(self, model, inner, delta):
        self.emb, self.inner = model.model.embed_tokens, inner
        self.delta = torch.tensor(np.asarray(delta, np.float32))
        self.rid = None

    @property
    def fire(self):
        return self.inner.fire

    @fire.setter
    def fire(self, v):
        self.inner.fire = v

    def __enter__(self):
        if self.rid is None:
            from transformers import AutoTokenizer
            self.rid = AutoTokenizer.from_pretrained(MODEL).convert_tokens_to_ids("<|START_RESPONSE|>")
        W = self.emb.weight
        self.row0 = W.data[self.rid].clone()
        W.data[self.rid] = self.row0 + self.delta.to(W.device, W.dtype)
        self.inner.__enter__()
        return self

    def __exit__(self, *a):
        self.inner.__exit__(*a)
        self.emb.weight.data[self.rid] = self.row0


_context5 = r5lib.context


def context(model, c):
    """Round 5's cell specs, plus "flag": true: the vector file's m is the flag read phi and q is 0 (vec_v5.npz), and the flag is written while the cell runs."""
    inner = _context5(model, {k: v for k, v in c.items() if k != "flag"})
    if c.get("flag"):
        return Flagged(model, inner, np.load(CACHE / "wp_flag.npz")["delta"])
    return inner


r5lib.context = context
