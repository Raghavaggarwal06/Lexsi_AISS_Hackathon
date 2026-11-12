"""Round 7: side-by-side behaviour evaluation of three layer-22 edits from wsp run real-01, on the live GPU worker.

  A  AyaDial as it stands: the starting knob ("current" in the request schema), runs/real-01/case/start_edit, hash 652305b8341870ab
  B  the editor's interim candidate accepted at turn 96 (edits/history/t096_*), hash a0b53cb078d5352a, units 6666 and 1896 only
  C  the editor's final edit (edits/current), hash 2f196f8de6de1ae6: A on 6666/1896 plus an add-on pair on 9173/6376
  Base  the original model ("original" arm), added to every B job

One worker job carries one edit ("edited" arms) plus the starting knob ("current" arms), so every (format, prompt selection)
is run twice: a B job with arms [current s=1, edited s=1, original] and a C job with arms [current s=1, edited s=1].

  python wsp/eval_edits.py --plan        print the jobs (sets, ids, formats, sizes) and check them; submit nothing
  python wsp/eval_edits.py --run         submit the jobs one at a time through passes.RemoteBackend, wait, validate, store
  python wsp/eval_edits.py --analyze     build summary.json and summary.md from the stored job summaries

--run is resumable: a job whose summary exists in results/round7/fable_edits/jobs/ is skipped; a job already queued or claimed
under its k is waited for, not sent again. A failed job is never resubmitted under the same k (Drive-sync quirk): the ledger
records the failure and the job is retried under a fresh k from 600 upward.

Nothing written here holds prompt or reply text: job summaries carry numbers, set ids, prompt ids and labels only. The worker
keeps prompts and replies in its private folder. Picks use the catalogue's fixed order (passes.fixed_order), the order ids index.
"Never read" is strict: a prompt counts as read if the editor read the same underlying prompt (runs_private/round6/sets.jsonl
set|id) in any read pass under any catalogue set id (state.json "seen" mapped through the catalogue). Only set and id fields of
the private file are used.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

WSP = Path(__file__).resolve().parent
BUILD = WSP.parent
sys.path.insert(0, str(WSP))
import edits as E  # noqa: E402
import passes as P  # noqa: E402
from worker import boot_ci  # noqa: E402  (the worker's own bootstrap: B = 2000, seed 0, percentile)

RUN = WSP / "runs" / "real-01"
OUT = BUILD / "results" / "round7" / "fable_edits"
JOBS = OUT / "jobs"
LEDGER = JOBS / "ledger.json"
LOG = OUT / "eval_log.txt"
EVAL_RUN_ID = "r7_fable_edits"
HASH = {"A": "652305b8341870ab", "B": "a0b53cb078d5352a", "C": "2f196f8de6de1ae6"}
FOLD_UNITS = {6666, 1896}
MAX_NEW = 128
ARMS = {"B": [{"weights": "current", "scale": 1.0}, {"weights": "edited", "scale": 1.0}, {"weights": "original", "scale": 0.0}],
        "C": [{"weights": "current", "scale": 1.0}, {"weights": "edited", "scale": 1.0}]}

FMT_NAME = {"default": "default (no system prompt)", "sys_bank": "short one-line system prompt", "sys_persona": "persona system prompt",
            "sys_car": "car-assistant system prompt", "sys_de": "German system prompt", "sys_long": "long system prompt (house style)",
            "sys_law": "long system prompt (law office)", "mt_benign": "after two harmless turns", "mt_refused": "after a refused turn",
            "mt3": "after three harmless turns", "mt_sys": "system prompt plus two turns"}
BIG_FORMATS = ["sys_bank", "mt_sys", "mt_benign"]                       # format pool + 412 + 412 never-read adversarial ids
SMALL_FORMATS = ["sys_persona", "sys_car", "sys_de", "sys_long", "sys_law", "mt_refused", "mt3"]   # format pool only
JBB = ["jbb_GCG_pool", "jbb_GCG_dev", "jbb_PAIR_pool", "jbb_PAIR_dev", "jbb_RS_pool", "jbb_RS_dev", "jbb_DSN_pool", "jbb_DSN_dev"]
MJ = ["mj_en_dev", "mj_zh_dev", "mj_it_dev", "mj_vi_dev", "mj_ar_dev", "mj_ko_dev", "mj_th_dev", "mj_bn_dev", "mj_sw_dev", "mj_jv_dev"]
AYA9 = "harmless prompts, 9 other languages (Aya)"
GROUPS = [("deployment", "Deployment formats (format pool, and never-read adversarial prompts in three formats)"),
          ("default_en", "Default format, English benchmarks"),
          ("languages", "Languages (default format)")]


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def jdump(obj):
    return json.dumps(obj, indent=1, sort_keys=False)


# == inputs ==
def load_state(files):
    st = {}
    for f in files:
        c = E.load_npz(f)
        st[c["name"]] = c
    return st


def load_edits():
    A = load_state(sorted((RUN / "case" / "start_edit").glob("*.npz")))
    B = load_state(sorted((RUN / "edits" / "history").glob("t096_*.npz")))
    C = load_state(sorted((RUN / "edits" / "current").glob("*.npz")))
    st = {"A": A, "B": B, "C": C}
    checks = {}
    for k, s in st.items():
        h = E.state_hash(s)
        if h != HASH[k]:
            raise SystemExit(f"edit {k}: hash {h}, expected {HASH[k]}")
        checks[f"{k}_hash"] = h
        checks[f"{k}_units"] = E.state_units(s)[0]
    if set(E.state_units(B)[1]) != FOLD_UNITS:
        raise SystemExit(f"B touches units {E.state_units(B)[1]}, not only 6666 and 1896")
    same = True
    for name, a in A.items():
        c = C[name]
        for i, u in enumerate(a["units"].tolist()):
            j = c["units"].tolist().index(u)
            same &= bool(np.array_equal(a["install"][i], c["install"][j]) and np.array_equal(a["knob"][i], c["knob"][j]))
    checks["C_equals_A_on_6666_1896_bitwise"] = same
    checks["B_touches_only_6666_1896"] = True
    if not same:
        raise SystemExit("C differs from A on units 6666/1896")
    return st, checks


QUEUE_OVERRIDE, POLL_OVERRIDE = None, None


def run_config():
    cfg = json.loads((RUN / "config.json").read_text())
    cfg["run_id"] = EVAL_RUN_ID
    if QUEUE_OVERRIDE:
        cfg["remote"]["queue_dir"] = QUEUE_OVERRIDE
    if POLL_OVERRIDE:
        cfg["remote"]["poll_s"] = POLL_OVERRIDE
    return cfg


class Catalogue:
    """Catalogue (set, index) -> underlying private key set|id, mirroring worker.Catalogue's ordering. Reads only set and id."""

    def __init__(self):
        self.sets = json.loads((RUN / "case" / "sets.json").read_text())
        self.formats = json.loads((RUN / "case" / "formats.json").read_text())
        self.meta = {s["set"]: s for s in self.sets}
        by, group_key = collections.defaultdict(list), {}
        with open(BUILD / "runs_private" / "round6" / "sets.jsonl") as f:
            for line in f:
                r = json.loads(line)
                k = f"{r['set']}|{r['id']}"
                by[r["set"]].append(k)
                g = r.get("vanilla_hash") or r.get("cluster")
                if g:
                    group_key[k] = g
        dev = {r["row"]: r["sets"] for r in json.loads((BUILD / "results/round5/tables/dev_main.json").read_text())["rows"]}
        s5 = json.loads((BUILD / "results/round5/robust/suites.json").read_text())
        harm_of = {g[0]: int(g[2]) for g in s5["groups"]}
        self.keys, self.labels = {}, {}
        for s in self.sets:
            sid = s["set"]
            if sid == "format_pool":
                ks = list(s5["ids"])
                labs = ["harmful" if harm_of[k.split("|")[0]] else "harmless" for k in ks]
            else:
                mem = [sid] if sid in by else dev[sid]
                ks = [k for m in mem for k in by[m]]
                labs = [s["label"]] * len(ks)
            if len(ks) != s["n"]:
                raise SystemExit(f"catalogue mismatch for {sid}: {len(ks)} vs {s['n']}")
            self.keys[sid], self.labels[sid] = ks, labs
        self.group_key = group_key
        seen = json.loads((RUN / "state.json").read_text())["seen"]
        self.seen_setid = {(s, int(i)) for s, i in seen}
        self.read_keys = {self.keys[s][int(i)] for s, i in seen}
        self.read_groups = {group_key[k] for k in self.read_keys if k in group_key}
        self.behaved = self.run_behave_tests()

    def key(self, s, i):
        return self.keys[s][i]

    def is_read(self, s, i):
        return self.keys[s][i] in self.read_keys

    def seen_expanded(self):
        """Every catalogue (set, index) whose underlying prompt the editor read: what the worker's read/never-read split uses."""
        return sorted((s, i) for s, ks in self.keys.items() for i, k in enumerate(ks) if k in self.read_keys)

    def run_behave_tests(self):
        """{(edit hash or None, format): set of underlying keys} for every behave pass of the run (the editor saw their labels)."""
        out = collections.defaultdict(set)
        for line in open(RUN / "tool_calls.jsonl"):
            d = json.loads(line)
            if d.get("kind") != "behave":
                continue
            r = d["request"]
            uses_edit = any(a["weights"] == "edited" for a in r["arms"])
            for s in r["sets"]:
                ids = r["ids"] if "ids" in r else P.fixed_order(s, self.meta[s]["n"])[: min(r["n"], self.meta[s]["n"])]
                for i in ids:
                    out[(d["edit_hash"] if uses_edit else None, r["format"])].add(self.keys[s][i])
                    out[("any", "any")].add(self.keys[s][i])
        return out


# == the plan ==
def never_read_order(cat, s, exclude_keys):
    """Ids of set s in the catalogue's fixed order that the editor never read, never saw in a run behave pass, and that are not
    in exclude_keys (the format pool's own prompts)."""
    tested = cat.behaved[("any", "any")]
    return [i for i in P.fixed_order(s, cat.meta[s]["n"])
            if not cat.is_read(s, i) and cat.key(s, i) not in tested and cat.key(s, i) not in exclude_keys]


def prefer_unread(cat, s, n):
    order = P.fixed_order(s, cat.meta[s]["n"])
    ids = [i for i in order if not cat.is_read(s, i)] + [i for i in order if cat.is_read(s, i)]
    return sorted(ids[:n])


def build_plan(cat):
    pool_keys = set(cat.keys["format_pool"])
    adv_h = never_read_order(cat, "adversarial_harmful", pool_keys)
    adv_b = never_read_order(cat, "adversarial_benign", pool_keys)
    full = lambda s: list(range(cat.meta[s]["n"]))  # noqa: E731
    sels = []
    for f in BIG_FORMATS:
        sels.append(("deployment", f, f"dep_{f}", [("format_pool", full("format_pool")), ("adversarial_harmful", sorted(adv_h[:412])),
                                                  ("adversarial_benign", sorted(adv_b[:412]))]))
    sels.append(("default_en", "default", "def_jb", [(s, prefer_unread(cat, s, 103) if s == "jbb_RS_pool" else full(s)) for s in JBB]
                 + [("itw_h_dev", full("itw_h_dev")), ("sorry_base_dev", full("sorry_base_dev")), ("harmful", full("harmful"))]))
    sels.append(("default_en", "default", "def_mix", [("adversarial_harmful", sorted(adv_h[:200])), ("adversarial_benign", sorted(adv_b[:179]))]
                 + [(s, full(s)) for s in ("itw_b_dev", "orbench_hard", "xstest_safe", "benign", "pht_harmless_dev")]))
    for f in SMALL_FORMATS:
        sels.append(("deployment", f, f"dep_{f}", [("format_pool", full("format_pool"))]))
    sels.append(("languages", "default", "lang", [(s, full(s)) for s in MJ] + [("aya_en_dev", full("aya_en_dev")), (AYA9, full(AYA9))]))
    # added 11 Nov 23:59 at the owner's request: the two hardest system prompts with the never-read WildJailbreak ids (512 + 512; the setting of the
    # run's own behaviour test of B, on ids it never read), so that the long-system-prompt result can be checked directly
    for f in ("sys_long", "sys_law"):
        sels.append(("deployment", f, f"dep_{f}_adv", [("adversarial_harmful", sorted(adv_h[:512])), ("adversarial_benign", sorted(adv_b[:512]))]))
    jobs = []
    for group, fmt, tag, picks in sels:
        for ed in ("B", "C"):
            jobs.append({"name": f"{tag}_{ed}", "k": 500 + len(jobs), "group": group, "format": fmt, "edit": ed, "picks": picks,
                         "arms": ARMS[ed]})
    return jobs, {"adversarial_harmful_never_read_untested_available": len(adv_h),
                  "adversarial_benign_never_read_untested_available": len(adv_b)}


def check_job(job, cat, cfg):
    """Each (set, ids) through passes.validate_request, then the job-level limits. Returns the normalised request."""
    norm_arms = None
    for s, ids in job["picks"]:
        n1, _ = P.validate_request({"type": "behave", "arms": job["arms"], "set": s, "ids": ids, "format": job["format"],
                                    "max_new_tokens": MAX_NEW}, cat.sets, cat.formats, cfg)
        norm_arms = n1["arms"]
    total = sum(len(i) for _, i in job["picks"])
    if total > cfg["behave_max_prompts_per_arm"] or len(job["picks"]) > 16:
        raise SystemExit(f"{job['name']}: {total} prompts in {len(job['picks'])} sets exceeds the limits")
    keys = [cat.key(s, i) for s, ids in job["picks"] for i in ids]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{job['name']}: the same underlying prompt appears twice")
    return {"type": "behave", "sets": [s for s, _ in job["picks"]], "format": job["format"],
            "selection": "explicit ids per set (see picks)", "arms": norm_arms, "max_new_tokens": MAX_NEW}


def spec_hash(job, norm, seen):
    return hashlib.sha256(json.dumps({"norm": norm, "picks": job["picks"], "edit": HASH[job["edit"]], "start": HASH["A"],
                                      "seen": seen}, sort_keys=True).encode()).hexdigest()[:16]


# == the worker ==
def worker_status(queue):
    try:
        d = json.loads((queue / "worker_info.json").read_text())
    except (OSError, ValueError):
        return None, None, "worker_info.json unreadable"
    hb = d.get("heartbeat")
    age = None
    if hb:
        t = dt.datetime.strptime(hb, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
        age = (dt.datetime.now(dt.timezone.utc) - t).total_seconds()
    return bool(d.get("ready")) and not d.get("stopped"), age, d


def run_jobs(args):
    states, checks = load_edits()
    cat = Catalogue()
    cfg = run_config()
    jobs, _ = build_plan(cat)
    seen = [list(x) for x in cat.seen_expanded()]
    backend = P.RemoteBackend(cfg)
    queue = backend.queue
    JOBS.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    log(f"edits verified: {checks}")
    for job in jobs:
        if args.only and job["name"] not in args.only.split(","):
            continue
        dst = JOBS / f"{job['name']}.json"
        if dst.exists():
            log(f"{job['name']}: summary exists; skipped")
            continue
        norm = check_job(job, cat, cfg)
        sh_ = spec_hash(job, norm, seen)
        ent = ledger.setdefault(job["name"], {"spec_hash": sh_, "attempts": []})
        if ent["spec_hash"] != sh_:
            log(f"{job['name']}: plan changed since the last submission; a fresh k will be used")
            for a in ent["attempts"]:
                a["status"] = a["status"] if a["status"] != "submitted" else "abandoned (plan changed)"
            ent["spec_hash"] = sh_
        used = {a["k"] for e in ledger.values() for a in e["attempts"]}
        live = [a for a in ent["attempts"] if a["status"] == "submitted"]
        if live and (queue / f"fwd_{live[-1]['k']}.failed.json").exists():
            live[-1]["status"] = "failed (found on resume)"
            live = []
        if live:
            k = live[-1]["k"]
        else:
            fails = sum(a["status"].startswith("failed") for a in ent["attempts"])
            if fails >= args.max_retries + 1:
                log(f"{job['name']}: failed {fails} times; giving up on this job")
                continue
            taken = lambda kk: kk in used or any((queue / f"fwd_{kk}{suf}").exists()  # noqa: E731
                                                 for suf in (".json", ".json.claimed", ".done.json", ".failed.json", ".edit.npz"))
            k = job["k"]
            if ent["attempts"] or taken(k):  # a retry, or a primary k with files we did not write: take a fresh k from 600 up
                k = max([599] + [u for u in used if u >= 600]) + 1
                while taken(k):
                    k += 1
            ent["attempts"].append({"k": k, "status": "submitted", "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        LEDGER.write_text(jdump(ledger))
        ok, age, info = worker_status(queue)
        if not ok or age is None or age > args.max_heartbeat_age:
            log(f"STOP: worker not alive before {job['name']} (ready {ok}, heartbeat age {age if age is None else round(age)} s)")
            ent["attempts"][-1]["status"] = "not sent (worker not alive)" if not live else ent["attempts"][-1]["status"]
            LEDGER.write_text(jdump(ledger))
            return 3
        n = sum(len(i) for _, i in job["picks"])
        t0 = time.time()
        key, eh, sh = backend.submit(k, norm, [[s, ids] for s, ids in job["picks"]], states[job["edit"]], states["A"],
                                     [tuple(x) for x in seen])
        if eh != HASH[job["edit"]] or sh != HASH["A"]:
            raise SystemExit(f"hash mismatch at submit: {eh} {sh}")
        ent["attempts"][-1]["job_key"] = key
        LEDGER.write_text(jdump(ledger))
        log(f"{job['name']}: submitted as fwd_{k} ({n} prompts x {len(job['arms'])} arms, format {job['format']}, edit {eh})")
        res, err, last_note = None, None, 0
        while True:
            try:
                res = backend.collect(k, norm, key, eh, sh, JOBS)
            except RuntimeError as e:
                err = str(e)[:500]
                break
            if res is not None:
                break
            el = time.time() - t0
            if el > args.job_timeout:
                err = f"no answer within {args.job_timeout} s"
                break
            ok, age, _ = worker_status(queue)
            if age is not None and age > args.stale_abort and not (queue / f"fwd_{k}.json.claimed").exists():
                log(f"STOP: heartbeat {round(age)} s old and the job is not claimed; worker presumed dead")
                return 3
            if el - last_note > 300:
                claimed = (queue / f"fwd_{k}.json.claimed").exists()
                log(f"{job['name']}: waiting {round(el)} s; claimed {claimed}; heartbeat age {None if age is None else round(age)} s")
                last_note = el
            time.sleep(backend.poll_s)
        if err is not None:
            ent["attempts"][-1].update(status=f"failed: {err}", time_end=time.strftime("%Y-%m-%d %H:%M:%S"))
            LEDGER.write_text(jdump(ledger))
            log(f"{job['name']}: FAILED under fwd_{k}: {err}")
            if "within" in err:
                return 4  # a timeout: stop rather than pile jobs onto a stuck worker
            return run_jobs(args)  # retried under a fresh k (the failed one is never reused)
        meta = backend.last_meta or {}
        if meta.get("edit_hash_applied") != HASH[job["edit"]] or meta.get("start_hash_applied") != HASH["A"]:
            raise SystemExit(f"{job['name']}: the worker applied {meta.get('edit_hash_applied')} / {meta.get('start_hash_applied')}")
        if meta.get("weights_hash_after") != meta.get("weights_hash_original"):
            raise SystemExit(f"{job['name']}: layer 22 not restored on the worker")
        P.validate_response(norm, res)
        worker_meta = {kk: meta.get(kk) for kk in ("k", "job_key", "run_id", "device", "dtype", "version", "edit_hash_applied",
                                                    "start_hash_applied", "start_is_reference_knob", "weights_hash_original",
                                                    "weights_hash_after", "n_prompts", "n_truncated", "seconds", "seconds_generating",
                                                    "generated_tokens", "judge", "judge_seconds", "judge_calls", "judge_unparsed",
                                                    "max_new_tokens", "recovered")}
        rec = {"name": job["name"], "k": k, "group": job["group"], "format": job["format"], "edit": job["edit"],
               "edit_hash": HASH[job["edit"]], "start_hash": HASH["A"], "arms": norm["arms"], "picks": job["picks"],
               "beh_file": f"beh_{k}.jsonl", "result": res, "worker": worker_meta, "wall_seconds": round(time.time() - t0, 1),
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        dst.write_text(jdump(rec))
        ent["attempts"][-1].update(status="done", time_end=rec["finished"])
        LEDGER.write_text(jdump(ledger))
        pr = res["pairs"][0]
        log(f"{job['name']}: done in {rec['wall_seconds']} s (worker {meta.get('seconds')} s); edited minus current: unsafe "
            f"{pr.get('unsafe_diff_b_minus_a')} {pr.get('ci95')}, refusal {pr.get('refusal_diff_b_minus_a')} {pr.get('refusal_ci95')}")
    log("all jobs done")
    return 0


# == analysis ==
def rate(lab, m, j):
    return None if not m.any() else round(float(lab[m, j].mean()) * 100, 1)


def cluster_ci(d, clusters, B=2000, seed=0):
    """Percentile bootstrap of 100 * mean(d) resampling underlying prompts (a prompt seen in several formats moves as one)."""
    d = np.asarray(d, np.float64)
    if len(d) == 0:
        return [None, None]
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    s, c = np.bincount(inv, weights=d), np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(4):  # in blocks, to bound memory
        idx = rng.integers(0, len(uniq), (B // 4, len(uniq)))
        out.append(s[idx].sum(1) / c[idx].sum(1) * 100)
    m = np.concatenate(out)
    return [round(float(x), 1) for x in np.percentile(m, [2.5, 97.5])]


def diff_block(x, y, m, metric):
    """y minus x over mask m: points, 95% bootstrap interval, flips. metric index 0 = refusal, 1 = unsafe."""
    if not m.any():
        return None
    d = y[m, metric] - x[m, metric]
    return {"n": int(m.sum()), "diff": round(float(d.mean()) * 100, 1), "ci95": boot_ci(d), "up": int((d > 0).sum()),
            "down": int((d < 0).sum()), "net_prompts": int(d.sum())}


def analyze(args):
    cat = Catalogue()
    jobs = {}
    for p in sorted(JOBS.glob("*.json")):
        if p.name == "ledger.json":
            continue
        j = json.loads(p.read_text())
        jobs[j["name"]] = j
    pairs = collections.OrderedDict()
    plan, _ = build_plan(cat)
    for job in plan:
        tag = job["name"].rsplit("_", 1)[0]
        pairs.setdefault(tag, {"group": job["group"], "format": job["format"], "B": None, "C": None})
        pairs[tag][job["edit"]] = jobs.get(job["name"])
    tests = {e: {f: cat.behaved.get((HASH[e], f), set()) for f in FMT_NAME} for e in ("B", "C")}
    units, pooled_rows = [], []
    for tag, pr in pairs.items():
        jb, jc = pr["B"], pr["C"]
        if jb is None and jc is None:
            continue
        per = collections.OrderedDict()
        for ed, j in (("B", jb), ("C", jc)):
            if j is None:
                continue
            idx = {("A" if a["weights"] == "current" else ed if a["weights"] == "edited" else "Base"): t for t, a in enumerate(j["arms"])}
            for line in open(JOBS / j["beh_file"]):
                r = json.loads(line)
                e = per.setdefault((r["set"], int(r["id"])), {"label": r["label"]})
                for arm, t in idx.items():
                    e[("A_" + ed) if arm == "A" else arm] = r["labels"][t]
        fmt = pr["format"]
        by_unit = collections.OrderedDict()
        for (s, i), e in per.items():
            by_unit.setdefault((s, e["label"]), []).append((i, e))
        for (s, label), items in by_unit.items():
            ids = [i for i, _ in items]
            keys = np.array([cat.key(s, i) for i in ids])
            nr = np.array([not cat.is_read(s, i) for i in ids])
            nr_setid = np.array([(s, i) not in cat.seen_setid for i in ids])
            arms = {}
            for arm in ("A_B", "A_C", "B", "C", "Base"):
                if all(arm in e for _, e in items):
                    arms[arm] = np.array([e[arm] for _, e in items], np.int64)
            A = arms.get("A_B", arms.get("A_C"))
            metric = 1 if label == "harmful" else 0
            allm = np.ones(len(ids), bool)
            u = {"group": pr["group"], "format": fmt, "set": s, "label": label, "n": len(ids), "n_never_read": int(nr.sum()),
                 "n_never_read_by_set_id": int(nr_setid.sum()),
                 "n_behaviour_tested_in_run_B_same_format": int(sum(k in tests["B"].get(fmt, set()) for k in keys)),
                 "n_behaviour_tested_in_run_C_same_format": int(sum(k in tests["C"].get(fmt, set()) for k in keys)),
                 "rates": {}, "rates_never_read": {}, "main_metric": "unsafe" if metric else "refusal"}
            for name, arm in (("A", A), ("B", arms.get("B")), ("C", arms.get("C")), ("Base", arms.get("Base"))):
                if arm is None:
                    continue
                u["rates"][name] = {"refused": rate(arm, allm, 0), "unsafe": rate(arm, allm, 1), "degenerate": rate(arm, allm, 2)}
                u["rates_never_read"][name] = {"refused": rate(arm, nr, 0), "unsafe": rate(arm, nr, 1), "degenerate": rate(arm, nr, 2)} if nr.any() else None
            if "A_B" in arms and "A_C" in arms:
                u["A_label_changes_between_B_and_C_jobs"] = int((arms["A_B"] != arms["A_C"]).any(1).sum())
                u["rates"]["A_in_C_job"] = {"refused": rate(arms["A_C"], allm, 0), "unsafe": rate(arms["A_C"], allm, 1)}
            for nm, x, y in (("B_minus_A", arms.get("A_B"), arms.get("B")), ("C_minus_A", arms.get("A_C"), arms.get("C")),
                             ("B_minus_C", arms.get("C"), arms.get("B")), ("A_minus_Base", arms.get("Base"), arms.get("A_B"))):
                if x is None or y is None:
                    continue
                u[nm] = {"main": diff_block(x, y, allm, metric), "main_never_read": diff_block(x, y, nr, metric),
                         "other": diff_block(x, y, allm, 1 - metric)}
                pooled_rows.append({"group": pr["group"], "format": fmt, "set": s, "label": label, "cmp": nm,
                                    "d": (y[:, metric] - x[:, metric]).astype(np.float64), "keys": keys, "nr": nr,
                                    "x": x[:, metric], "y": y[:, metric]})
            units.append(u)
    pooled = pool(pooled_rows)
    meta = {"what": "round 7: three layer-22 edits from wsp run real-01, side by side on the live GPU worker",
            "edits": {"A": f"AyaDial as it stands (starting knob, 'current'), hash {HASH['A']}",
                      "B": f"interim candidate accepted at turn 96 (edits/history/t096_*), hash {HASH['B']}, units 6666/1896 only",
                      "C": f"final edit (edits/current), hash {HASH['C']}: A on 6666/1896 plus an add-on pair on 9173/6376",
                      "Base": "original CohereLabs/tiny-aya-global (arm 'original', run in every B job)"},
            "scale": 1.0, "max_new_tokens": MAX_NEW, "judge": "Qwen3Guard-Gen-0.6B (worker); a degenerate reply counts as neither",
            "never_read": "strict: the editor never read the same underlying prompt (private set|id) in any read pass under any set id; "
                          "n_never_read_by_set_id uses state.json 'seen' literally",
            "intervals": "per row: percentile bootstrap over prompts (2000 draws, seed 0, the worker's boot_ci); pooled: the same "
                         "resampling underlying prompts as clusters, so a prompt run in several formats moves as one",
            "A_reference": "B minus A pairs B with the A arm of the B job; C minus A pairs C with the A arm of the C job (same prompts)",
            "jobs": {n: {"k": j["k"], "format": j["format"], "edit": j["edit"], "n_prompts": j["worker"].get("n_prompts"),
                         "arms": [a["weights"] for a in j["arms"]], "worker_seconds": j["worker"].get("seconds"),
                         "judge_unparsed": j["worker"].get("judge_unparsed"), "n_truncated": j["worker"].get("n_truncated")}
                     for n, j in jobs.items()},
            "missing_jobs": [job["name"] for job in plan if job["name"] not in jobs]}
    out = {"meta": meta, "rows": units, "pooled": pooled}
    (OUT / "summary.json").write_text(jdump(out))
    (OUT / "summary.md").write_text(render_md(out))
    log(f"summary written: {len(units)} rows, {len(jobs)} jobs, missing {meta['missing_jobs']}")
    return 0


def pool(rows):
    """Pooled lines: by group (and for deployment, by part and by format), harmful unsafe and harmless refusal, every comparison."""
    out = []
    specs = []
    for g, _ in GROUPS:
        specs.append((g, "all rows", lambda r, g=g: r["group"] == g))
        if g == "deployment":
            specs.append((g, "format pool only (10 formats)", lambda r: r["group"] == "deployment" and r["set"] == "format_pool"))
            specs.append((g, "adversarial sets only (3 formats)", lambda r: r["group"] == "deployment" and r["set"] != "format_pool"))
            specs.append((g, "system-prompt formats", lambda r: r["group"] == "deployment" and r["format"].startswith("sys_")))
            specs.append((g, "multi-turn formats", lambda r: r["group"] == "deployment" and r["format"].startswith("mt")))
            for f in BIG_FORMATS + SMALL_FORMATS:
                specs.append((g, f"format {f}", lambda r, f=f: r["group"] == "deployment" and r["format"] == f))
        if g == "default_en":
            specs.append((g, "JailbreakBench sets", lambda r: r["group"] == "default_en" and r["set"].startswith("jbb_")))
            specs.append((g, "adversarial sets", lambda r: r["group"] == "default_en" and r["set"].startswith("adversarial")))
        if g == "languages":
            specs.append((g, "MultiJail without English", lambda r: r["group"] == "languages" and r["set"].startswith("mj_") and r["set"] != "mj_en_dev"))
    for g, part, sel in specs:
        for label in ("harmful", "harmless"):
            for cmp_ in ("B_minus_A", "C_minus_A", "B_minus_C", "A_minus_Base"):
                rs = [r for r in rows if sel(r) and r["label"] == label and r["cmp"] == cmp_]
                if not rs:
                    continue
                e = {"group": g, "part": part, "label": label, "metric": "unsafe" if label == "harmful" else "refusal", "cmp": cmp_}
                for sub, msk in (("all", None), ("never_read", "nr")):
                    d = np.concatenate([r["d"] if msk is None else r["d"][r["nr"]] for r in rs])
                    k = np.concatenate([r["keys"] if msk is None else r["keys"][r["nr"]] for r in rs])
                    x = np.concatenate([r["x"] if msk is None else r["x"][r["nr"]] for r in rs])
                    y = np.concatenate([r["y"] if msk is None else r["y"][r["nr"]] for r in rs])
                    if len(d) == 0:
                        e[sub] = None
                        continue
                    e[sub] = {"n_rows": int(len(d)), "n_prompts": int(len(set(k.tolist()))), "x_rate": round(float(x.mean()) * 100, 1),
                              "y_rate": round(float(y.mean()) * 100, 1), "diff": round(float(d.mean()) * 100, 1),
                              "ci95": cluster_ci(d, k), "up": int((d > 0).sum()), "down": int((d < 0).sum()), "net_rows": int(d.sum())}
                out.append(e)
    return out


# == markdown ==
def fmt_ci(b):
    if b is None:
        return "–"
    lo, hi = b["ci95"]
    return f"{b['diff']:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def fmt_rates(u, key, which=("A", "B", "C", "Base")):
    vals = []
    for w in which:
        r = u["rates"].get(w)
        vals.append("–" if r is None or r.get(key) is None else f"{r[key]:.1f}")
    return " / ".join(vals)


def set_name(s, label):
    return f"format pool, {label}" if s == "format_pool" else s


def render_md(out):
    rows, pooled, meta = out["rows"], out["pooled"], out["meta"]
    L = ["# Round 7: AyaDial (A) against the editor's two edits (B, C), side by side", ""]
    L.append(f"Edits: A = {meta['edits']['A']}; B = {meta['edits']['B']}; C = {meta['edits']['C']}. Base = the original model. "
             f"All at s = 1, greedy, {MAX_NEW} new tokens, judged by Qwen3Guard-Gen-0.6B on the GPU worker. "
             f"{len(meta['jobs'])} jobs ran" + (f"; missing: {', '.join(meta['missing_jobs'])}" if meta["missing_jobs"] else "") + ". "
             "Numbers are percentages of prompts; differences are in points, with 95% bootstrap intervals over prompts. "
             "'nr' = prompts the editor never read (strict: no read pass touched the same underlying prompt under any set id). "
             "Rates are given as A / B / C / Base. B minus A uses the A arm of the B job; C minus A the A arm of the C job.")
    L.append("")
    for g, title in GROUPS:
        gr = [u for u in rows if u["group"] == g]
        if not gr:
            continue
        L.append(f"## {title}")
        L.append("")
        for label, metric, other in (("harmful", "unsafe", "refused"), ("harmless", "refused", "unsafe")):
            sub = [u for u in gr if u["label"] == label]
            if not sub:
                continue
            L.append(f"**{label.capitalize()} prompts: {'unsafe replies (lower is better)' if label == 'harmful' else 'refusals (lower is better)'}**")
            L.append("")
            L.append(f"| format | set | n | n nr | {metric} % A / B / C / Base | {other} % A / B / C / Base | B − A | C − A | B − A, nr | C − A, nr | tested in run (B, C) |")
            L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for u in sub:
                g1 = lambda nm, k: fmt_ci((u.get(nm) or {}).get(k))  # noqa: E731
                L.append(f"| {u['format']} | {set_name(u['set'], label)} | {u['n']} | {u['n_never_read']} | {fmt_rates(u, metric)} | "
                         f"{fmt_rates(u, other)} | {g1('B_minus_A', 'main')} | {g1('C_minus_A', 'main')} | {g1('B_minus_A', 'main_never_read')} | "
                         f"{g1('C_minus_A', 'main_never_read')} | {u['n_behaviour_tested_in_run_B_same_format']}, "
                         f"{u['n_behaviour_tested_in_run_C_same_format']} |")
            L.append("")
    L.append("## Pooled lines")
    L.append("")
    L.append("Each line pools every (format, prompt) row of the group; the interval resamples underlying prompts. "
             "'rows' counts (format, prompt) pairs and 'prompts' distinct prompts. Rates are x → y for the comparison y − x.")
    L.append("")
    L.append("| group | part | metric | comparison | rows (prompts) | rates | diff [95% CI] | nr rows (prompts) | nr diff [95% CI] |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for e in pooled:
        if e["cmp"] not in ("B_minus_A", "C_minus_A", "B_minus_C"):
            continue
        a, nr = e["all"], e["never_read"]
        cmpn = {"B_minus_A": "B − A", "C_minus_A": "C − A", "B_minus_C": "B − C"}[e["cmp"]]
        L.append(f"| {e['group']} | {e['part']} | {e['label']} {e['metric']} | {cmpn} | {a['n_rows']} ({a['n_prompts']}) | "
                 f"{a['x_rate']:.1f} → {a['y_rate']:.1f} | {fmt_ci(a)} | "
                 + (f"{nr['n_rows']} ({nr['n_prompts']}) | {fmt_ci(nr)} |" if nr else "0 | – |"))
    L.append("")
    L.append("<!-- READING -->")
    L.append("")
    return "\n".join(L)


def plan_report(args):
    states, checks = load_edits()
    cat = Catalogue()
    cfg = run_config()
    jobs, avail = build_plan(cat)
    print(json.dumps(checks, indent=1, default=str))
    print(json.dumps(avail))
    tot = 0
    for job in jobs:
        norm = check_job(job, cat, cfg)
        n = sum(len(i) for _, i in job["picks"])
        nr = sum(not cat.is_read(s, i) for s, ids in job["picks"] for i in ids)
        tot += n * len(job["arms"])
        print(f"k={job['k']} {job['name']:<22} {job['format']:<11} edit {job['edit']} arms {len(norm['arms'])} prompts {n:4d} "
              f"never-read {nr:4d} sets {len(job['picks'])}: " + ", ".join(f"{s}:{len(i)}" for s, i in job["picks"]))
    print(f"{len(jobs)} jobs, {tot} generations in all; expanded seen size {len(cat.seen_expanded())}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--only", help="comma list of job names (with --run)")
    ap.add_argument("--max-heartbeat-age", type=float, default=900, help="seconds; refuse to submit if the heartbeat is older")
    ap.add_argument("--stale-abort", type=float, default=1800, help="seconds; stop waiting for an unclaimed job if the heartbeat is older")
    ap.add_argument("--job-timeout", type=float, default=2700)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--queue", help="override the queue folder (tests only; default: the run's remote.queue_dir)")
    ap.add_argument("--out", help="override the output folder (tests only; default: results/round7/fable_edits)")
    ap.add_argument("--poll", type=float, help="override the poll interval in seconds (tests only)")
    a = ap.parse_args()
    global OUT, JOBS, LEDGER, LOG, QUEUE_OVERRIDE, POLL_OVERRIDE
    if a.out:
        OUT = Path(a.out).expanduser().resolve()
        JOBS, LEDGER, LOG = OUT / "jobs", OUT / "jobs" / "ledger.json", OUT / "eval_log.txt"
    QUEUE_OVERRIDE, POLL_OVERRIDE = a.queue, a.poll
    if a.plan:
        return plan_report(a)
    if a.run:
        rc = run_jobs(a)
        if rc:
            return rc
    if a.analyze or a.run:
        return analyze(a)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
