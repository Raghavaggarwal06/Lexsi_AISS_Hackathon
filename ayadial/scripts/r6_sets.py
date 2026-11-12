"""Build the round 6 sets: a new sealed exam ("exam B", sets xb_*) and development slices (xd_*), made only of rows that appear in no earlier set.
CPU only; runs on the Mac or on the runtime (repo at /content/build). ROOT is build/ (found from this file's location, as in common.py);
set R6_ROOT to override it.
  python scripts/r6_sets.py

Private rows -> runs_private/round6/sets.jsonl = every row of runs_private/round5/sets.jsonl, byte for byte and in the same order, then the new rows.
Also private: runs_private/round6/itw_system_templates.jsonl (40 unused in-the-wild templates, for use as system prompts by another script) and
runs_private/round6/format_requests.jsonl (request lists for the format tests). Ids, sizes, rules, revisions and hashes, never text ->
results/round6/sets_manifest.json. Round 5's files and scripts are only read.

Earlier rows = every row of runs_private/round5/sets.jsonl (fit, development and frozen; it carries round 4's non-frozen rows) plus round 4's frozen
rows (runs_private/round4/sets.jsonl; round 5's builder excluded them too), plus the ids round 5 dropped as near-duplicates before generation.
A new row is refused if (1) it is over the token limit, (2) its normalised text equals an earlier row or another new row, (3) for exam rows, its
character 3-gram Jaccard with an earlier row or a new development row is 0.85 or more, or (4) it reuses a source item an earlier row used
(WildJailbreak index or vanilla request, in-the-wild template, MultiJail request, Alpaca, Aya or OR-Bench row). Candidates are taken in a fixed
hash order (sha1 of "r6|<part>|<key>"), so the build is deterministic without any random generator.

Helpers below are copied from scripts/r5_sets.py unchanged (norm, hsh, hmod, ntok and MAXTOK, CORE, LANG, row, PH and fill, the 3-gram set of the
SORRY near-duplicate check); r5_sets.py is not imported because its module code reads round 3 files and writes round 5 folders.
Idempotent: the build always starts again from the round 5 file. It refuses to replace an existing round 6 file that holds rows other than
round 5's, xb_ and xd_, or to change exam B once a cell of suite xb6 or xd6 exists in runs_private/round6/gen.
"""
import collections, hashlib, heapq, json, math, os, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT as _ROOT, read_jsonl

ROOT = Path(os.environ.get("R6_ROOT") or _ROOT)
OUT, RAW = ROOT / "results/round6", ROOT / "runs_private/round6"
R5OUT, R5RAW, R4RAW = ROOT / "results/round5", ROOT / "runs_private/round5", ROOT / "runs_private/round4"
TOKENIZER = "CohereLabs/tiny-aya-global"

# ---- copied from scripts/r5_sets.py (unchanged)
norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
hsh = lambda t: hashlib.sha1(t.encode()).hexdigest()
hmod = lambda t, m: int(hsh(t), 16) % m
MAXTOK = 1350  # prompt tokens; the template adds about 60 and the run-time limit is 1536
CORE = ["Illegal Activity", "Hate Speech", "Malware", "Physical Harm", "Fraud", "Privacy Violence"]
LANG = {"en": ("eng", "English"), "zh": ("zho", "Simplified Chinese"), "it": ("ita", "Italian"), "vi": ("vie", "Vietnamese"), "ar": ("arb", "Standard Arabic"),
        "ko": ("kor", "Korean"), "th": ("tha", "Thai"), "bn": ("ben", "Bengali"), "sw": ("swh", "Swahili"), "jv": ("jav", "Javanese")}
PH = re.compile(r"\[\s*(?:insert\s+)?(?:your\s+)?(?:prompt|question|query)(?:\s+here)?\s*\]|\{\s*(?:prompt|question|query)\s*\}", re.I)
fill = lambda t, q: (PH.sub(lambda m: q, t), "placeholder") if PH.search(t) else (t.rstrip() + "\n\n" + q, "append")
grams = lambda t: {t[i:i + 3] for i in range(max(1, len(t) - 2))}  # r5_sets.py sorry(): applied to norm(text)


def row(set_, id_, prompt, src, harm, **kw):
    return {"id": id_, "set": set_, "prompt": prompt, "src": src, "harm": harm, **kw}
# ---- end of the copied helpers

T_NEAR = 0.85
okey = lambda part, k: hsh(f"r6|{part}|{k}")  # the fixed order in which candidates of a part are considered
RT = {"arabic": "ar", "english": "en", "filipino": "fil", "french": "fr", "hindi": "hi", "russian": "ru", "serbian": "sr", "spanish": "es"}
UNSEEN = ["fil", "fr", "hi", "ru", "sr", "es"]
AYB = {"fil": ("fil", "Filipino"), "fr": ("fra", "French"), "hi": ("hin", "Hindi"), "ru": ("rus", "Russian"), "sr": ("srp", "Serbian"), "es": ("spa", "Spanish")}
N = {"xd_art": 60, "xb_art": 150, "xb_art_ref": 100, "xd_ayab": 40, "xb_ayab": 60, "xb_wj_h": 300, "xb_wj_b": 300, "itw_templates": 150, "itw_sys": 40,
     "itw_cap": 12, "xb_aya": 40, "xb_orb": 250, "xb_alp": 250, "fmt_exam": 40}
FIT_FMT = {"h": [("trfit_h", 50), ("harmful_fit", 50)], "b": [("trfit_b", 34), ("benign_fit", 33), ("orbench_hard_fit", 33)]}
DATASETS = ["CohereLabs/aya_redteaming", "CohereLabs/aya_dataset", "allenai/wildjailbreak", "TrustAIRLab/in-the-wild-jailbreak-prompts",
            "TrustAIRLab/forbidden_question_set", "DAMO-NLP-SG/MultiJail", "tatsu-lab/alpaca", "bench-llm/or-bench"]


class Near:
    """Round 5's near-duplicate rule (r5_sets.py sorry() and r5_sets_plain.py): character 3-gram Jaccard on normalised text, >= 0.85, computed
    exactly (no MinHash estimate). A prefix-filter index finds the candidate pairs; it cannot miss a pair at or above the threshold
    (two sets with Jaccard >= t share a gram among the first |s| - ceil(t|s|) + 1 grams of each, under any fixed order of grams)."""

    def __init__(self, items, t=T_NEAR):
        self.t, self.intern, self.sets, self.tags, self.index = t, {}, [], [], collections.defaultdict(list)
        G = [(tag, self.gs(txt)) for tag, txt in items]
        self.freq = collections.Counter(g for _, s in G for g in s)  # frozen here; grams first seen later rank as frequency 0
        for tag, s in G:
            self._add(tag, s)

    def gs(self, text):
        return frozenset(self.intern.setdefault(g, g) for g in grams(norm(text)))

    def _prefix(self, s):
        return heapq.nsmallest(len(s) - math.ceil(self.t * len(s) - 1e-9) + 1, s, key=lambda g: (self.freq.get(g, 0), g))

    def _add(self, tag, s):
        j = len(self.sets)
        self.sets.append(s); self.tags.append(tag)
        for g in self._prefix(s):
            self.index[g].append(j)

    def add(self, tag, text):
        self._add(tag, self.gs(text))

    def query(self, text):
        s = self.gs(text)
        cand = set()
        for g in self._prefix(s):
            cand.update(self.index.get(g, ()))
        hits = set()
        for j in cand:
            h = self.sets[j]
            if not 0.7 <= len(h) / max(1, len(s)) <= 1.43:  # round 5's length window (Jaccard >= 0.85 already implies 0.85 to 1.18)
                continue
            i_ = len(s & h)
            if i_ / max(1, len(s) + len(h) - i_) >= self.t:
                hits.add(self.tags[j])
        return sorted(hits)


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


def ids_sha(rows, text=False):
    keys = sorted(f"{r['set']}|{r['id']}" + (f"|{hsh(r['prompt'])}" if text else "") for r in rows)
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def build():
    from datasets import load_dataset
    from huggingface_hub import dataset_info, model_info
    from transformers import AutoTokenizer

    # ---------- earlier rows
    man5 = json.loads((R5OUT / "sets_manifest.json").read_text())
    r5_bytes = (R5RAW / "sets.jsonl").read_bytes()
    r5 = [json.loads(l) for l in r5_bytes.splitlines() if l.strip()]
    assert len(r5) == man5["round4_rows_carried"] + man5["new_rows"], "runs_private/round5/sets.jsonl does not match its manifest"
    assert ids_sha([r for r in r5 if r["set"].endswith("_frz")]) == man5["frozen_ids_sha256"], "round 5 frozen ids differ from the manifest"
    assert (R4RAW / "sets.jsonl").exists(), "runs_private/round4/sets.jsonl is needed (round 4's frozen rows are earlier rows); copy it before running"
    r4_frozen = [r for r in read_jsonl(R4RAW / "sets.jsonl") if r["set"].startswith("frozen")]
    earlier = [("r5", r) for r in r5] + [("r4", r) for r in r4_frozen]
    r5_dropped = man5.get("near_duplicate_drop", {}).get("dropped", [])  # "set|id" keys that were in round 5's sets before its near-duplicate drop
    all_keys = [(src, r["set"], r["id"]) for src, r in earlier] + [("r5drop",) + tuple(k.split("|", 1)) for k in r5_dropped]
    earlier_norm = {norm(r["prompt"]) for _, r in earlier}

    used = collections.defaultdict(set)
    pat = {"wj_eval": r"^wj-(\d+)$", "wj_train": r"^wjtr-(\d+)$", "wj_vanilla_row": r"^aug-wjv-(\d+)$", "itw_template": r"^itw-(\d+)::",
           "alpaca": r"^alp-(?:fz-)?(\d+)$|::a(\d+)$|^sfx-al(\d+)$", "aya": r"^aya-(\d+)$", "mj_request": r"^mj-(\d+)-|^sfx-mj(\d+)$", "orbench": r"^orb-(?:fz-)?(\d+)$"}
    for _, s_, i_ in all_keys:
        for k, p in pat.items():
            m = re.search(p, i_)
            if m:
                used[k].add(int(next(g for g in m.groups() if g is not None)))
    for _, r in earlier:
        if r.get("src") == "itw":
            used["itw_template"].add(int(r["template"]))
        if r.get("src") == "multijail":
            used["mj_request"].add(int(r["request"]))
    used_vanilla_hash = {r["vanilla_hash"] for _, r in earlier if r.get("vanilla_hash")}
    frz_q = sorted({tuple(int(x) for x in re.search(r"::q(\d+)-(\d+)$", r["id"]).groups()) for r in r5 if r["set"] == "itw_h_frz"})

    # ---------- revisions; a dataset that cannot be read with this machine's token is skipped and recorded, never fetched elsewhere
    rev, skipped = {}, []
    for d in DATASETS:
        try:
            rev[d] = dataset_info(d).sha
        except Exception as e:
            rev[d] = None
            skipped.append({"dataset": d, "stage": "dataset_info", "error": f"{type(e).__name__}: {str(e)[:160]}"})
    tok_rev = model_info(TOKENIZER).sha
    tok = AutoTokenizer.from_pretrained(TOKENIZER, revision=tok_rev)
    ntok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])  # as in r5_sets.py

    def load(d, *a, **kw):
        if rev.get(d) is None:
            return None
        try:
            return load_dataset(d, *a, revision=rev[d], **kw)
        except Exception as e:
            skipped.append({"dataset": d, "stage": "load_dataset", "error": f"{type(e).__name__}: {str(e)[:160]}"})
            rev[d] = None
            return None

    t0 = time.time()
    REF = Near([(f"{src}:{r['set']}|{r['id']}", r["prompt"]) for src, r in earlier])
    print(f"near-duplicate index over {len(earlier)} earlier rows: {time.time() - t0:.0f} s", flush=True)

    new, taken = [], set()
    log = collections.defaultdict(collections.Counter)  # per set: refused candidates by reason
    near_refused = collections.defaultdict(list)  # per set: [would-be id, first earlier key it near-duplicates]

    def accept(rows_, exam, allow_internal_dup=False):
        """All rows of a unit (one prompt, one in-the-wild template, one MultiJail request) pass or none does."""
        s_ = rows_[0]["set"]
        if any(ntok(r["prompt"]) > MAXTOK for r in rows_):
            log[s_]["over_token_limit"] += 1; return False
        ns = [norm(r["prompt"]) for r in rows_]
        if any(n in earlier_norm or n in taken for n in ns):
            log[s_]["text_equal_to_an_earlier_or_new_row"] += 1; return False
        if len(set(ns)) < len(ns) and not allow_internal_dup:
            log[s_]["duplicate_inside_unit"] += 1; return False
        if exam:
            for r in rows_:
                h = REF.query(r["prompt"])
                if h:
                    log[s_]["near_duplicate"] += 1
                    near_refused[s_].append([r["id"], h[0]]); return False
        taken.update(ns)
        new.extend(rows_)
        return True

    parts = {}

    # ---------- A. aya_redteaming: development and exam, split by sha1(normalised text) mod 5
    AR = load("CohereLabs/aya_redteaming")
    art = {}
    if AR is not None:
        for split, lg in RT.items():
            d = AR[split].select_columns(["prompt", "harm_category"])[:]
            cand, seen, dup = [], set(), 0
            for i, (p, hc) in enumerate(zip(d["prompt"], d["harm_category"])):
                if not p or not norm(p):
                    continue
                if norm(p) in seen:
                    dup += 1; continue
                seen.add(norm(p)); cand.append((i, p, hc))
            dev = sorted([c for c in cand if hmod(norm(c[1]), 5) == 0], key=lambda c: okey(f"art-{lg}", c[0]))
            exm = sorted([c for c in cand if hmod(norm(c[1]), 5) != 0], key=lambda c: okey(f"art-{lg}", c[0]))
            art[lg] = {"rows_in_split": len(d["prompt"]), "duplicates_in_split": dup, "dev_side": len(dev), "exam_side": len(exm), "exam": exm, "dev": dev}

    # ---------- B. aya_dataset harmless prompts in the six unseen languages (and E's ten languages later)
    AY = load("CohereLabs/aya_dataset")
    aya_cands = {}
    if AY is not None:
        A_ = AY["train"].select_columns(["language_code", "language", "inputs", "annotation_type"])[:]
        codes, names, inputs, ann = A_["language_code"], A_["language"], A_["inputs"], A_["annotation_type"]
        used_aya_norm = {norm(inputs[i]) for i in used["aya"]}
        by_code = collections.defaultdict(list)
        for i, c in enumerate(codes):
            by_code[c].append(i)

        def aya_pool(code, name):  # round 5's Aya rule: original annotations then re-annotations, 15 to 300 characters, unique, not in an earlier set
            cand, seen = [], set()
            for pref in ("original-annotations", "re-annotations"):
                for i in by_code[code]:
                    t = inputs[i]
                    if names[i] == name and ann[i] == pref and 15 <= len(t) <= 300 and norm(t) not in seen and norm(t) not in earlier_norm \
                            and i not in used["aya"] and norm(t) not in used_aya_norm:
                        seen.add(norm(t)); cand.append((i, t))
            return cand
        for lg, (code, name) in AYB.items():
            aya_cands[("b", lg)] = aya_pool(code, name)
        for lg, (code, name) in LANG.items():
            aya_cands[("e", lg)] = aya_pool(code, name)

    # ---------- development rows first (exam rows are then checked against them too)
    for lg in UNSEEN:
        if lg in art:
            k = 0
            for i, p, hc in art[lg]["dev"]:
                if k >= N["xd_art"]:
                    break
                k += accept([row(f"xd_art_{lg}", f"art-{lg}-{i}", p, "aya_redteaming", 1, lang=lg, harm_category=hc)], exam=False)
    for lg in AYB:
        if ("b", lg) in aya_cands:
            dev = sorted([c for c in aya_cands[("b", lg)] if hmod(norm(c[1]), 5) == 0], key=lambda c: okey(f"ayab-{lg}", c[0]))
            k = 0
            for i, t in dev:
                if k >= N["xd_ayab"]:
                    break
                k += accept([row(f"xd_ayab_{lg}", f"ayab-{lg}-{i}", t, "aya", 0, lang=lg)], exam=False)
    dev_rows = list(new)
    for r in dev_rows:
        REF.add(f"r6dev:{r['set']}|{r['id']}", r["prompt"])
    dev_near_earlier = sum(bool([h for h in REF.query(r["prompt"]) if not h.startswith("r6dev:")]) for r in dev_rows)

    # ---------- A exam
    if art:
        for lg in RT.values():
            target, k = (N["xb_art"] if lg in UNSEEN else N["xb_art_ref"]), 0
            for i, p, hc in art[lg]["exam"]:
                if k >= target:
                    break
                k += accept([row(f"xb_art_{lg}", f"art-{lg}-{i}", p, "aya_redteaming", 1, lang=lg, harm_category=hc)], exam=True)
        cats = {}
        for r in new:
            if r["src"] == "aya_redteaming":
                try:
                    labels = json.loads(r["harm_category"])
                    labels = labels if isinstance(labels, list) else [str(labels)]
                except (ValueError, TypeError):
                    labels = [str(r["harm_category"])]
                c = cats.setdefault(r["set"], collections.Counter())
                c.update(labels)
        parts["A_aya_redteaming"] = {
            "dataset": "CohereLabs/aya_redteaming", "revision": rev["CohereLabs/aya_redteaming"],
            "rule": "each language split's prompts (duplicates inside a split kept once) split by sha1(normalised text) mod 5: 0 is development, the rest exam. "
                    "Development xd_art_<lg>: 60 per unseen language (fil, fr, hi, ru, sr, es). Exam xb_art_<lg>: 150 per unseen language, 100 each for ar and en (languages the read has seen; reference). "
                    "Candidates in the order sha1('r6|art-<lg>|<row index>'); id art-<lg>-<row index in the split>; harm_category kept as stored (a JSON list as text).",
            "per_language": {lg: {k: v for k, v in a.items() if k not in ("exam", "dev")} | {
                "dev_taken": sum(r["set"] == f"xd_art_{lg}" for r in new), "exam_taken": sum(r["set"] == f"xb_art_{lg}" for r in new),
                "dev_side_left_unused": a["dev_side"] - sum(r["set"] == f"xd_art_{lg}" for r in new), "exam_side_left_unused_or_refused": a["exam_side"] - sum(r["set"] == f"xb_art_{lg}" for r in new)}
                for lg, a in art.items()},
            "harm_category_counts": {s: dict(sorted(c.items())) for s, c in sorted(cats.items())},
            "note": "a prompt can carry several harm categories, so the category counts can sum to more than the set size; the development side of ar and en is left unused"}

    # ---------- B exam
    if aya_cands:
        for lg in AYB:
            ex = sorted([c for c in aya_cands[("b", lg)] if hmod(norm(c[1]), 5) != 0], key=lambda c: okey(f"ayab-{lg}", c[0]))
            k = 0
            for i, t in ex:
                if k >= N["xb_ayab"]:
                    break
                k += accept([row(f"xb_ayab_{lg}", f"ayab-{lg}-{i}", t, "aya", 0, lang=lg)], exam=True)
        parts["B_aya_unseen_languages"] = {
            "dataset": "CohereLabs/aya_dataset (train)", "revision": rev["CohereLabs/aya_dataset"],
            "rule": "round 5's Aya rule (language code and name match, original annotations then re-annotations, 15 to 300 characters, unique normalised text, not equal to an earlier row, "
                    "Aya rows used earlier excluded by index and text); split by sha1(normalised text) mod 5 == 0 development (40 per language), the rest exam (60); "
                    "candidates in the order sha1('r6|ayab-<lg>|<row index>'); id ayab-<lg>-<row index in the train split>",
            "languages": {lg: list(v) for lg, v in AYB.items()},
            "per_language": {lg: {"candidates": len(aya_cands[("b", lg)]), "dev_side": sum(hmod(norm(c[1]), 5) == 0 for c in aya_cands[("b", lg)]),
                                  "exam_side": sum(hmod(norm(c[1]), 5) != 0 for c in aya_cands[("b", lg)]),
                                  "dev_taken": sum(r["set"] == f"xd_ayab_{lg}" for r in new), "exam_taken": sum(r["set"] == f"xb_ayab_{lg}" for r in new)} for lg in AYB}}

    # ---------- C. WildJailbreak
    WE = load("allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False)
    WT = load("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False) if WE is not None else None
    if WE is not None and WT is not None:
        WE = WE["train"][:]
        pool_h = [i for i, t in enumerate(WE["data_type"]) if t == "adversarial_harmful" and i not in used["wj_eval"]]
        k = 0
        for i in sorted(pool_h, key=lambda i: okey("wjh", i)):
            if k >= N["xb_wj_h"]:
                break
            k += accept([row("xb_wj_h", f"wj-{i}", WE["adversarial"][i], "wildjailbreak", 1)], exam=True)
        W = WT["train"].select_columns(["data_type", "vanilla", "adversarial"])[:]
        dt, van, adv = W["data_type"], W["vanilla"], W["adversarial"]
        vh = lambda i: hsh(norm(van[i]))[:12]  # round 4's vanilla_hash
        used_vanilla_hash |= {vh(i) for i in used["wj_train"] | used["wj_vanilla_row"]}
        eval_norm = {norm(t) for t in WE["adversarial"]}
        part = lambda v: hmod(norm(v), 10)  # round 4 partition of vanilla requests: 0-5 fit, 6-7 development, 8-9 frozen
        pool_b = [i for i, t in enumerate(dt) if t == "adversarial_benign" and i not in used["wj_train"] and part(van[i]) >= 8 and vh(i) not in used_vanilla_hash
                  and norm(adv[i]) not in eval_norm]
        r5_rule_pool = sum(1 for i, t in enumerate(dt) if t == "adversarial_benign" and i not in used["wj_train"] and part(van[i]) >= 8)
        k, van_new = 0, set()
        for i in sorted(pool_b, key=lambda i: okey("wjb", i)):
            if k >= N["xb_wj_b"]:
                break
            if vh(i) in van_new:
                log["xb_wj_b"]["vanilla_already_behind_a_new_row"] += 1; continue
            ok = accept([row("xb_wj_b", f"wjtr-{i}", adv[i], "wildjailbreak", 0, vanilla_hash=vh(i))], exam=True)
            if ok:
                van_new.add(vh(i))
            k += ok
        parts["C_wildjailbreak"] = {
            "dataset": "allenai/wildjailbreak", "revision": rev["allenai/wildjailbreak"],
            "xb_wj_h": {"rule": "eval-split adversarial_harmful rows whose index no earlier set used (rounds 3 and 4 dev and frozen, round 5 wj5_h_frz); order sha1('r6|wjh|<index>'); id wj-<eval index>",
                        "pool_unused_before_round6": len(pool_h), "taken": sum(r["set"] == "xb_wj_h" for r in new),
                        "left_unused": len(pool_h) - sum(r["set"] == "xb_wj_h" for r in new), "round5_reported_pool": man5["parts"]["fresh"]["wj_eval_adversarial_harmful_unused"]},
            "xb_wj_b": {"rule": "round 5's wj5_b_frz rule (train-split adversarial_benign, vanilla request in round 4's frozen partition sha1(norm(vanilla)) mod 10 >= 8, index not used before, "
                                "text not equal to any earlier row nor to an eval-split row, at most 1350 tokens) plus two additions: the row's vanilla request (round 4's vanilla_hash) is behind no earlier row "
                                "(round 4 rows' vanilla_hash, and the vanilla of every earlier wjtr- and aug-wjv- row), and no two new rows share a vanilla request. Order sha1('r6|wjb|<index>'); id wjtr-<train index>",
                        "why_round5_rule_still_has_candidates": "round 5 scanned a seeded sample of 80,000 train rows and stopped at 350 accepted rows; the rule's pool was not exhausted",
                        "round5_rule_pool_now": r5_rule_pool, "pool_after_vanilla_rule": len(pool_b), "taken": sum(r["set"] == "xb_wj_b" for r in new),
                        "left_unused": len(pool_b) - sum(r["set"] == "xb_wj_b" for r in new)}}
        del W, dt, van, adv

    # ---------- Alpaca pool (D's harmless instructions and F's xb_alp), one fixed order
    AL = load("tatsu-lab/alpaca")
    alp_order = []
    if AL is not None:
        al = AL["train"].select_columns(["instruction", "input"])[:]
        used_al_norm = {norm(al["instruction"][i]) for i in used["alpaca"]}
        seen = set()
        for i in sorted(range(len(al["instruction"])), key=lambda i: okey("alp", i)):
            t = al["instruction"][i]
            if al["input"][i] or not (20 <= len(t) <= 200) or i in used["alpaca"] or norm(t) in used_al_norm or norm(t) in earlier_norm or norm(t) in seen:
                continue
            seen.add(norm(t)); alp_order.append((i, t))
    alp_consumed = set()

    # ---------- D. in-the-wild templates
    IT = load("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25")
    FQ = load("TrustAIRLab/forbidden_question_set")
    if IT is not None and FQ is not None and AL is not None:
        I = IT["train"].select_columns(["prompt", "community_id", "platform"])[:]
        itw = [{"prompt": I["prompt"][i], "community_id": I["community_id"][i], "platform": I["platform"][i]} for i in range(len(I["prompt"]))]
        F = FQ["train"][:]
        fq = {(int(F["content_policy_id"][i]), int(F["q_id"][i])): (F["question"][i], F["content_policy_name"][i]) for i in range(len(F["question"]))}
        assert all(q in fq and fq[q][1] in CORE for q in frz_q), "a round 5 frozen in-the-wild question is not a core forbidden question"
        tpl, seen = [], set()  # as in r5_sets.py
        for i in range(len(itw)):
            t = itw[i]["prompt"]
            if not t or norm(t) in seen:
                continue
            seen.add(norm(t))
            c = itw[i].get("community_id")
            group = f"c{int(c)}" if c is not None and str(c) not in ("", "nan", "None") else f"solo{hsh(norm(t))[:10]}"
            tpl.append({"i": i, "t": t, "group": group, "ntok": ntok(t), "platform": itw[i]["platform"]})
        elig = [x for x in tpl if 40 <= x["ntok"] <= 900]
        side5 = lambda x: "dev" if hmod(x["group"], 5) == 0 else "frz"  # round 5's community split, recorded only
        TPL = Near([(f"itw-{i}", itw[i]["prompt"]) for i in sorted(used["itw_template"])])
        cand = sorted([x for x in elig if x["i"] not in used["itw_template"]], key=lambda x: okey("itw", x["i"]))
        per, exam_t, sys_t, tlog, near_tpl = collections.Counter(), [], [], collections.Counter(), []
        alp_d = (x for x in alp_order if len(x[1]) <= 140)  # round 5's in-the-wild Alpaca rule: 20 to 140 characters
        nxt_al, examined = None, 0
        for x in cand:
            if len(exam_t) >= N["itw_templates"] and len(sys_t) >= N["itw_sys"]:
                break
            examined += 1
            if per[x["group"]] >= N["itw_cap"]:
                tlog["community_cap"] += 1; continue
            if norm(x["t"]) in earlier_norm:
                tlog["template_text_equal_to_an_earlier_row"] += 1; continue
            h = TPL.query(x["t"])
            if h:
                tlog["near_duplicate_of_a_used_or_picked_template"] += 1; near_tpl.append([f"itw-{x['i']}", h[0]]); continue
            if len(exam_t) < N["itw_templates"]:
                while nxt_al is None:  # the next fresh instruction that is not itself a near-duplicate of an earlier row
                    a = next(alp_d, None)
                    assert a is not None, "the Alpaca pool ran out"
                    if REF.query(a[1]):
                        tlog["alpaca_instruction_near_duplicate"] += 1; continue
                    nxt_al = a
                qs = sorted(frz_q, key=lambda q: okey(f"itwq-{x['i']}", f"{q[0]}-{q[1]}"))[:2]
                unit = []
                for pid, qid in qs:
                    p, mode = fill(x["t"], fq[(pid, qid)][0])
                    unit.append(row("xb_itw_h", f"itw-{x['i']}::q{pid}-{qid}", p, "itw", 1, template=x["i"], group=x["group"], mode=mode))
                p, mode = fill(x["t"], nxt_al[1])
                unit.append(row("xb_itw_b", f"itw-{x['i']}::a{nxt_al[0]}", p, "itw", 0, template=x["i"], group=x["group"], mode=mode))
                if not accept(unit, exam=True):
                    tlog["rows_refused_(token_text_or_near)"] += 1; continue
                alp_consumed.add(nxt_al[0]); nxt_al = None
                exam_t.append(x)
            else:
                sys_t.append(x)
            per[x["group"]] += 1
            TPL.add(f"itw-{x['i']}", x["t"])
        modes = collections.Counter((r["set"], r["mode"]) for r in new if r["src"] == "itw")
        sys_rows = [{"id": f"itw-{x['i']}", "template": x["i"], "group": x["group"], "ntok": x["ntok"], "platform": x["platform"], "text": x["t"]} for x in sys_t]
        comm = lambda xs: {"from_communities": sum(x["group"].startswith("c") for x in xs), "without_community": sum(x["group"].startswith("solo") for x in xs),
                           "round5_community_side": dict(sorted(collections.Counter(side5(x) for x in xs if x["group"].startswith("c")).items())),
                           "communities_also_used_in_round5": sum(1 for g in {x["group"] for x in xs} if g in {y["group"] for y in tpl if y["i"] in used["itw_template"]})}
        parts["D_in_the_wild"] = {
            "datasets": {"templates": "TrustAIRLab/in-the-wild-jailbreak-prompts jailbreak_2023_12_25", "questions": "TrustAIRLab/forbidden_question_set", "harmless": "tatsu-lab/alpaca"},
            "rule": "round 5's template rule (unique normalised templates, 40 to 900 tokens, community id as the group, templates without one grouped by their own hash, at most 12 templates per group; "
                    "the cap counts the 150 exam and the 40 system templates together). Excluded: any template an earlier row used (by index, including the template round 5 dropped) and any "
                    "template whose 3-gram Jaccard with a used template or with an already picked new template is 0.85 or more (treated as the same template). Not split (all exam). "
                    "Order sha1('r6|itw|<index>'). Each exam template: 2 harmful questions = the first 2 of round 5's frozen in-the-wild questions in the order sha1('r6|itwq-<template>|<pid>-<qid>') "
                    "(question reuse is intended: those questions were never in a fit), and 1 fresh Alpaca instruction (round 5's in-the-wild rule: no input, 20 to 140 characters; never used before; "
                    "each template gets a different one). Fill as in round 5 (placeholder if the template has one, else appended). A template is refused whole if any of its 3 rows fails a check. "
                    "Ids itw-<template>::q<pid>-<qid> and itw-<template>::a<alpaca index>.",
            "question_reuse": {"source": "questions of round 5's frozen in-the-wild slice (itw_h_frz), never in a fit", "n_questions": len(frz_q),
                               "questions_used": len({r["id"].split("::q")[1] for r in new if r["set"] == "xb_itw_h"})},
            "templates_unique": len(tpl), "eligible_40_to_900_tokens": len(elig), "used_before": len(used["itw_template"]),
            "eligible_unused": len(cand), "exam_templates": len(exam_t), "system_templates": len(sys_t),
            "refused_templates": dict(sorted(tlog.items())), "near_duplicate_templates_refused": near_tpl,
            "exam_template_groups": comm(exam_t), "system_template_groups": comm(sys_t),
            "eligible_unused_examined_in_order": examined, "eligible_unused_never_examined": len(cand) - examined,
            "fill_modes": {f"{a}|{b}": n for (a, b), n in sorted(modes.items())},
            "exam_template_ids": [x["i"] for x in sorted(exam_t, key=lambda x: x["i"])],
            "system_templates": {"file": "runs_private/round6/itw_system_templates.jsonl", "ids": [f"itw-{x['i']}" for x in sorted(sys_t, key=lambda x: x["i"])],
                                 "note": "not prompts and not part of exam B; 40 further unused eligible templates, disjoint from the 150 and not near-duplicates of them, for use as system prompts"}}
    else:
        sys_rows = None

    # ---------- E. MultiJail requests never used, all 10 languages; harmless Aya in the same languages
    MJ = load("DAMO-NLP-SG/MultiJail")
    if MJ is not None:
        M = MJ["train"][:]
        mj = {int(M["id"][i]): {lg: M[lg][i] for lg in LANG} for i in range(len(M["id"]))}
        rest = sorted(set(mj) - used["mj_request"])
        mj_log = collections.Counter()
        for q in rest:
            if any(not (mj[q][lg] or "").strip() for lg in LANG):
                mj_log["empty_text_in_a_language"] += 1; continue
            unit = [row(f"xb_mj_{lg}", f"mj-{q}-{lg}", mj[q][lg], "multijail", 1, lang=lg, request=q) for lg in LANG]
            mj_log["taken" if accept(unit, exam=True, allow_internal_dup=True) else "refused"] += 1
        parts["E_multijail"] = {
            "dataset": "DAMO-NLP-SG/MultiJail", "revision": rev["DAMO-NLP-SG/MultiJail"], "requests": len(mj), "used_before": len(used["mj_request"] & set(mj)),
            "remaining": len(rest), "requests_taken": mj_log["taken"], "requests_refused": mj_log["refused"], "empty": mj_log["empty_text_in_a_language"],
            "rule": "every request whose id no earlier row used, in all 10 languages; a request is refused whole (all 10 languages) if any of its language rows equals an earlier or new row "
                    "or near-duplicates an earlier row, so every xb_mj_<lg> set holds the same requests; two languages of one request may share a text. Id mj-<request>-<lg>",
            "languages": list(LANG)}
    if aya_cands:
        for lg in LANG:
            ex = sorted([c for c in aya_cands[("e", lg)] if hmod(norm(c[1]), 5) != 0], key=lambda c: okey(f"aya-{lg}", c[0]))
            k = 0
            for i, t in ex:
                if k >= N["xb_aya"]:
                    break
                k += accept([row(f"xb_aya_{lg}", f"aya-{i}", t, "aya", 0, lang=lg)], exam=True)
        parts["E_aya_multijail_languages"] = {
            "dataset": "CohereLabs/aya_dataset (train)", "revision": rev["CohereLabs/aya_dataset"],
            "rule": "round 5's Aya rule and language names (see B); only the exam side of round 5's split (sha1(normalised text) mod 5 != 0) so the development side stays for later; "
                    "Aya rows used earlier excluded by index and text; 40 per language in the order sha1('r6|aya-<lg>|<row index>'); id aya-<row index> as in round 5",
            "per_language": {lg: {"candidates_unused": len(aya_cands[("e", lg)]), "exam_side_unused": sum(hmod(norm(c[1]), 5) != 0 for c in aya_cands[("e", lg)]),
                                  "taken": sum(r["set"] == f"xb_aya_{lg}" for r in new)} for lg in LANG}}

    # ---------- F. OR-Bench-hard and Alpaca
    OB = load("bench-llm/or-bench", "or-bench-hard-1k")
    if OB is not None:
        ob = OB["train"][:]["prompt"]
        seen, pool = set(), []
        for i, p in enumerate(ob):
            if p and i not in used["orbench"] and norm(p) not in earlier_norm and norm(p) not in seen:
                seen.add(norm(p)); pool.append((i, p))
        k = 0
        for i, p in sorted(pool, key=lambda c: okey("orb", c[0])):
            if k >= N["xb_orb"]:
                break
            k += accept([row("xb_orb", f"orb-{i}", p, "orbench", 0)], exam=True)
        parts["F_orbench"] = {"dataset": "bench-llm/or-bench or-bench-hard-1k", "revision": rev["bench-llm/or-bench"], "rows": len(ob), "fresh_pool": len(pool), "taken": k, "left_unused": len(pool) - k,
                              "rule": "rows whose index no earlier row used (orb- and orb-fz- ids) and whose normalised text equals no earlier row (round 5's rule), unique; order sha1('r6|orb|<index>'); id orb-<index>"}
    if AL is not None:
        k = 0
        for i, t in alp_order:
            if k >= N["xb_alp"]:
                break
            if i in alp_consumed:
                continue
            k += accept([row("xb_alp", f"alp-{i}", t, "alpaca", 0)], exam=True)
        parts["F_alpaca"] = {"dataset": "tatsu-lab/alpaca", "revision": rev["tatsu-lab/alpaca"], "fresh_pool": len(alp_order), "used_by_xb_itw_b": len(alp_consumed), "taken": k,
                             "left_unused": len(alp_order) - len(alp_consumed) - k,
                             "rule": "round 5's alp5_frz rule (no input field, 20 to 200 characters, normalised text equal to no earlier row); Alpaca rows used earlier (alp-, alp-fz-, sfx-al and in-the-wild "
                                     "a-ids) excluded by index and text; one fixed order sha1('r6|alp|<index>') shared with D, whose instructions are skipped here; id alp-<index>"}

    return dict(r5_bytes=r5_bytes, r5=r5, earlier=earlier, new=new, dev_rows=dev_rows, REF=REF, used=used, log=log, near_refused=near_refused, parts=parts, rev=rev,
                tok_rev=tok_rev, tok=tok, ntok=ntok, skipped=skipped, sys_rows=sys_rows, dev_near_earlier=dev_near_earlier, man5=man5, r4_frozen_n=len(r4_frozen),
                earlier_norm=earlier_norm)


def format_requests(r5, new):
    """G. Request lists for the format tests (rendered later by another script)."""
    out, info = [], {}
    by = collections.defaultdict(list)
    for r in new:
        by[r["set"]].append(r)
    plan = [("xb_wj_h", 1), ("xb_mj_en", 1), ("xb_orb", 0), ("xb_alp", 0), ("xb_wj_b" if by.get("xb_wj_b") else "xb_alp", 0)]
    taken = set()
    for s_, harm in plan:
        pool = sorted([r for r in by.get(s_, []) if (r["set"], r["id"]) not in taken], key=lambda r: okey(f"fmt-{s_}", r["id"]))[:N["fmt_exam"]]
        for r in sorted(pool, key=lambda r: r["id"]):
            taken.add((r["set"], r["id"]))
            out.append({"id": r["id"], "set": r["set"], "prompt": r["prompt"], "harm": harm, "side": "exam"})
    suites = json.loads((R5OUT / "robust/suites.json").read_text())
    harm_of = {g[0]: g[2] for g in suites["groups"]}
    r5_by_key = {f"{r['set']}|{r['id']}": r for r in r5}
    dev_keys = suites["ids"]
    for k in dev_keys:
        r = r5_by_key[k]
        out.append({"id": r["id"], "set": r["set"], "prompt": r["prompt"], "harm": harm_of[r["set"]], "side": "dev"})
    dev_set = set(dev_keys)
    for c, plan_ in FIT_FMT.items():
        for s_, n in plan_:
            pool = sorted([r for r in r5 if r["set"] == s_ and f"{s_}|{r['id']}" not in dev_set], key=lambda r: okey(f"fmtfit-{s_}", r["id"]))[:n]
            for r in sorted(pool, key=lambda r: r["id"]):
                out.append({"id": r["id"], "set": r["set"], "prompt": r["prompt"], "harm": 1 if c == "h" else 0, "side": "fit"})
    cnt = collections.Counter((r["side"], r["set"], r["harm"]) for r in out)
    info = {"file": "runs_private/round6/format_requests.jsonl", "rows": len(out),
            "counts": {f"{a}|{b}|harm{c}": n for (a, b, c), n in sorted(cnt.items())},
            "rule": "exam: 40 each from xb_wj_h and xb_mj_en (harm 1), xb_orb, xb_alp and xb_wj_b (harm 0; 40 more xb_alp if xb_wj_b were missing), first in the order sha1('r6|fmt-<set>|<id>'). "
                    "dev: the 200 requests of round 5's format tests (results/round5/robust/suites.json, ids), listed again so the file is complete; harm from that file's groups. "
                    "fit: 100 harmful (50 trfit_h, 50 harmful_fit) and 100 harmless (34 trfit_b, 33 benign_fit, 33 orbench_hard_fit) round 5 fit rows, first in the order sha1('r6|fmtfit-<set>|<id>'), "
                    "none of them among the 200 dev requests.",
            "ids": {side: [f"{r['set']}|{r['id']}" for r in out if r["side"] == side] for side in ("exam", "dev", "fit")}}
    return out, info


def main():
    t0 = time.time()
    b = build()
    r5, new, earlier, REF, used, ntok, tok = b["r5"], b["new"], b["earlier"], b["REF"], b["used"], b["ntok"], b["tok"]
    exam = [r for r in new if r["set"].startswith("xb_")]
    dev = [r for r in new if r["set"].startswith("xd_")]
    assert len(exam) + len(dev) == len(new)

    # ---------- checks on the finished rows (all must be zero; anything found is dropped and recorded)
    earlier_norm = b["earlier_norm"]
    keys_new = [(r["set"], r["id"]) for r in new]
    assert len(set(keys_new)) == len(keys_new), "duplicate set|id among new rows"
    exact_vs_earlier = sum(norm(r["prompt"]) in earlier_norm for r in new)
    units = collections.defaultdict(set)  # normalised text -> units holding it (two languages of one MultiJail request may share a text)
    for r in new:
        units[norm(r["prompt"])].add(("mj", r["request"]) if r["src"] == "multijail" else (r["set"], r["id"]))
    exact_inside_new = sum(len(u) - 1 for u in units.values())
    over_limit = [f"{r['set']}|{r['id']}" for r in new if ntok(r["prompt"]) > MAXTOK]
    tpl_clash = [f"{r['set']}|{r['id']}" for r in exam if r.get("src") == "itw" and r["template"] in used["itw_template"]]
    mj_clash = [f"{r['set']}|{r['id']}" for r in exam if r.get("src") == "multijail" and r["request"] in used["mj_request"]]
    idx_clash = sum(1 for r in exam for k, p in (("wj_eval", r"^wj-(\d+)$"), ("wj_train", r"^wjtr-(\d+)$"), ("aya", r"^aya-(\d+)$"), ("orbench", r"^orb-(\d+)$"), ("alpaca", r"^alp-(\d+)$|::a(\d+)$"))
                    if (m := re.search(p, r["id"])) and int(next(g for g in m.groups() if g)) in used[k])
    near_post = {}
    for r in exam:
        h = REF.query(r["prompt"])
        if h:
            near_post[f"{r['set']}|{r['id']}"] = h[0]
    drop = set(near_post) | set(over_limit) | set(tpl_clash) | set(mj_clash)
    drop_tpl = {int(k.split("|", 1)[1].split("::")[0][4:]) for k in drop if k.split("|", 1)[1].startswith("itw-")}  # a template is dropped whole
    drop_req = {int(re.match(r"mj-(\d+)-", k.split("|", 1)[1]).group(1)) for k in drop if k.split("|", 1)[1].startswith("mj-")}  # a request is dropped whole
    keep = [r for r in new if f"{r['set']}|{r['id']}" not in drop and not (r["set"].startswith("xb_itw") and r["template"] in drop_tpl)
            and not (r["set"].startswith("xb_mj_") and r["request"] in drop_req)]
    post_dropped = sorted({f"{r['set']}|{r['id']}" for r in new} - {f"{r['set']}|{r['id']}" for r in keep})
    assert exact_vs_earlier == 0 and exact_inside_new == 0 and idx_clash == 0, (exact_vs_earlier, exact_inside_new, idx_clash)
    new = keep
    exam = [r for r in new if r["set"].startswith("xb_")]
    dev = [r for r in new if r["set"].startswith("xd_")]
    chat_len = lambda t: ntok(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False))
    longest = max(new, key=lambda r: ntok(r["prompt"]))
    max_chat = chat_len(longest["prompt"])
    chat_all = [chat_len(r["prompt"]) for r in new]
    chat_info = {"max_tokens_with_chat_template": max(chat_all), "rows_over_1536_with_chat_template": sum(c > 1536 for c in chat_all),
                 "chat_template_overhead_tokens": chat_len("hello") - ntok("hello"),
                 "note": "r4lib renders every prompt with apply_chat_template (no system turn) and a 1536-token limit; this template adds about 365 tokens, not the 60 that round 5's comment assumes, "
                         "so a prompt near 1350 tokens would not fit; no new row comes near that. A system prompt added later (in-the-wild system templates are 40 to 900 tokens) adds to this."}

    # ---------- format requests (G) and the system templates
    fmt, fmt_info = format_requests(r5, new)

    # ---------- refuse to clobber anything that is not ours
    RAW.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    tgt = RAW / "sets.jsonl"
    body = b["r5_bytes"] if b["r5_bytes"].endswith(b"\n") else b["r5_bytes"] + b"\n"
    body += "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new).encode()
    if tgt.exists():
        old = tgt.read_bytes()
        old_rows = [json.loads(l) for l in old.splitlines() if l.strip()]
        r5_keys = {(r["set"], r["id"]) for r in r5}
        foreign = [r for r in old_rows if (r["set"], r["id"]) not in r5_keys and not r["set"].startswith(("xb_", "xd_"))]
        assert not foreign, f"runs_private/round6/sets.jsonl holds {len(foreign)} rows that are neither round 5's nor xb_/xd_: refusing to overwrite"
        gen = RAW / "gen"
        cells = [p.name for p in gen.glob("*.jsonl") if re.search(r"xb6|xd6", p.name)] if gen.exists() else []
        assert old == body or not cells, f"exam B cells already exist ({len(cells)} files in runs_private/round6/gen) and the rebuild differs: refusing to change the sealed exam"
        rebuilt = "identical to the existing file" if old == body else ("replaced a file that held " + str(sum(r["set"].startswith(("xb_", "xd_")) for r in old_rows)) + " xb_/xd_ rows")
    else:
        rebuilt = "new file"
    tmp = tgt.with_suffix(".jsonl.tmp"); tmp.write_bytes(body); tmp.replace(tgt)
    with open(RAW / "format_requests.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in fmt)
    if b["sys_rows"] is not None:
        with open(RAW / "itw_system_templates.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in sorted(b["sys_rows"], key=lambda r: r["template"]))

    sizes = collections.Counter(r["set"] for r in new)
    targets = {}
    for s_ in sizes:
        lg = s_.rsplit("_", 1)[-1]
        targets[s_] = (N["xd_art"] if s_.startswith("xd_art") else N["xb_art"] if s_.startswith("xb_art") and lg in UNSEEN else N["xb_art_ref"] if s_.startswith("xb_art")
                       else N["xd_ayab"] if s_.startswith("xd_ayab") else N["xb_ayab"] if s_.startswith("xb_ayab") else N["xb_aya"] if s_.startswith("xb_aya_")
                       else 2 * N["itw_templates"] if s_ == "xb_itw_h" else N["itw_templates"] if s_ == "xb_itw_b" else None if s_.startswith("xb_mj_")
                       else N.get(s_))
    below = {s_: {"n": sizes[s_], "target": t} for s_, t in sorted(targets.items()) if t is not None and sizes[s_] < t}
    for s_ in ["xb_wj_h", "xb_wj_b", "xb_itw_h", "xb_itw_b", "xb_orb", "xb_alp"] + [f"xd_art_{l}" for l in UNSEEN] + [f"xb_art_{l}" for l in RT.values()] + \
              [f"xd_ayab_{l}" for l in AYB] + [f"xb_ayab_{l}" for l in AYB] + [f"xb_aya_{l}" for l in LANG]:
        if s_ not in sizes:
            below[s_] = {"n": 0, "target": None, "why": "part skipped (see skipped)"}
    man = {
        "written_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "script": "scripts/r6_sets.py",
        "round5_rows_carried": len(r5), "round4_frozen_rows_also_treated_as_earlier": b["r4_frozen_n"], "new_rows": len(new),
        "new_sizes": dict(sorted(sizes.items())), "below_target": below, "token_limit": MAXTOK,
        "token_rule": "prompt tokens of CohereLabs/tiny-aya-global's tokenizer without special tokens, at most 1350 (round 5's rule; the chat template adds about 60 and the run-time limit is 1536); candidates over the limit are refused before selection",
        "longest_new_row": {"id": f"{longest['set']}|{longest['id']}", "prompt_tokens": ntok(longest["prompt"]), "tokens_with_chat_template": max_chat},
        "chat_template_lengths": chat_info,
        "dropped_over_token_limit": {s_: c["over_token_limit"] for s_, c in sorted(b["log"].items()) if c.get("over_token_limit")},
        "refused_candidates_by_set": {s_: dict(sorted(c.items())) for s_, c in sorted(b["log"].items())},
        "sides": {"exam_b": len(exam), "development": len(dev)},
        "exam_b_n": len(exam), "exam_b_ids_sha256": ids_sha(exam), "exam_b_ids_and_text_sha256": ids_sha(exam, text=True),
        "exam_b_hash_definition": "sha256 of the newline-joined sorted '<set>|<id>' keys of all xb_ rows (round 5's frozen_ids_sha256 definition); the _and_text variant appends '|sha1(text)'",
        "dev_n": len(dev), "dev_ids_sha256": ids_sha(dev),
        "sets_jsonl_sha256": hashlib.sha256(body).hexdigest(), "sets_jsonl_rows": len(r5) + len(new), "sets_jsonl_write": rebuilt,
        "overlap_checks": {
            "earlier_rows": f"{len(r5)} rows of runs_private/round5/sets.jsonl (fit, development and frozen) + {b['r4_frozen_n']} round 4 frozen rows + {len(b['man5'].get('near_duplicate_drop', {}).get('dropped', []))} ids round 5 dropped before generation",
            "new_rows_with_text_equal_to_an_earlier_row": exact_vs_earlier, "new_rows_with_text_equal_to_another_new_row": exact_inside_new,
            "exam_rows_reusing_an_earlier_source_index": idx_clash, "exam_rows_reusing_an_earlier_itw_template": len(tpl_clash), "exam_rows_reusing_an_earlier_multijail_request": len(mj_clash),
            "question_reuse": "xb_itw_h reuses the harmful questions of round 5's frozen in-the-wild slice by design (never in a fit); every template is new",
            "near_duplicate_rule": "character 3-gram Jaccard of normalised text >= 0.85, computed exactly (round 5's r5_sets.py rule; CuratorKIT's threshold and n-gram without the MinHash estimate), against every earlier row and every new development row",
            "near_duplicate_exam_candidates_refused": sum(len(v) for v in b["near_refused"].values()),
            "near_duplicate_exam_candidates_refused_by_set": {s_: len(v) for s_, v in sorted(b["near_refused"].items())},
            "near_duplicate_refused_ids": {s_: v for s_, v in sorted(b["near_refused"].items())},
            "near_duplicate_exam_rows_after_selection": len(near_post), "dropped_after_selection": post_dropped,
            "development_rows_near_an_earlier_row_(kept, informational)": b["dev_near_earlier"],
            "how": "every candidate is checked before it is accepted (refused candidates are listed with the first earlier row they match), then every finished exam row is checked again; "
                   "rows found at the second check are dropped (an in-the-wild template or a MultiJail request as a whole)"},
        "dataset_revisions": b["rev"], "tokenizer_revision": {TOKENIZER: b["tok_rev"]}, "skipped": b["skipped"],
        "inputs_sha256": {p: sha256_file(ROOT / p) for p in ("runs_private/round5/sets.jsonl", "runs_private/round4/sets.jsonl", "results/round5/sets_manifest.json", "results/round5/robust/suites.json")},
        "round5_frozen_ids_sha256_checked": b["man5"]["frozen_ids_sha256"],
        "used_before_counts": {k: len(v) for k, v in sorted(used.items())},
        "parts": b["parts"], "format_requests": fmt_info,
        "ids": {s_: [r["id"] for r in new if r["set"] == s_] for s_ in sorted(sizes)}}
    (OUT / "sets_manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))

    print(f"\n{'set':<18}{'n':>6}{'target':>8}")
    for s_ in sorted(sizes, key=lambda s: (not s.startswith("xd_"), s)):
        t = targets.get(s_)
        print(f"{s_:<18}{sizes[s_]:>6}{(t if t is not None else 'all'):>8}{'  < target' if t is not None and sizes[s_] < t else ''}")
    print(f"\nexam_b_n {len(exam)}  dev_n {len(dev)}  format_requests {len(fmt)}  itw_system_templates {len(b['sys_rows'] or [])}")
    print(f"exam_b_ids_sha256 {man['exam_b_ids_sha256']}")
    print(f"sets.jsonl sha256 {man['sets_jsonl_sha256']} ({man['sets_jsonl_rows']} rows; {rebuilt})")
    print(f"near-duplicate candidates refused {man['overlap_checks']['near_duplicate_exam_candidates_refused']}; after selection {len(near_post)}; skipped parts {len(b['skipped'])}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
