"""Drop frozen rows that CuratorKIT found to be near-duplicates (MinHash Jaccard >= 0.85) of a development or fit row.
Runs once, before any generation: refuses if a generated cell already exists. Rewrites runs_private/round5/sets.jsonl and updates the manifest.
Dropped: the new frozen rows of finding 2 and the in-the-wild frozen templates that near-duplicate a development row.
Kept, with a stated caveat: JailbreakBench JBC and random-search rows, which share one fixed attack template per method with development by construction."""
import hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, read_jsonl, write_jsonl

OUT, RAW = ROOT / "results/round5", ROOT / "runs_private/round5"
assert not any((RAW / "gen").glob("*.jsonl")) if (RAW / "gen").exists() else True, "cells were already generated: the frozen slice can no longer change"
rep = json.loads((OUT / "pipeline/curate_report.json").read_text())
man = json.loads((OUT / "sets_manifest.json").read_text())
assert "near_duplicate_drop" not in man, "already applied"
F = rep["findings"]
drop = set(F["2_new_frozen_rows_near_fit_or_development"]["near_ge_0.85"]["frozen_row_ids"])
other = F["other_frozen_families_near_fit_or_development"]["near_ge_0.85"]
tpl = {f"itw-{i}" for i in other["frozen_itw_template_indices"]}
rows = read_jsonl(RAW / "sets.jsonl")
drop |= {f"{r['set']}|{r['id']}" for r in rows if r["set"].startswith("itw_") and r["set"].endswith("_frz") and r["id"].split("::")[0] in tpl}
assert all(k.split("|")[0].endswith("_frz") for k in drop)
keep = [r for r in rows if f"{r['set']}|{r['id']}" not in drop]
assert len(rows) - len(keep) == len(drop)
write_jsonl(RAW / "sets.jsonl", keep)
hsh = lambda t: hashlib.sha1(t.encode()).hexdigest()
frz = [r for r in keep if r["set"].endswith("_frz")]
fk = sorted(f"{r['set']}|{r['id']}" for r in frz)
ft = sorted(f"{r['set']}|{r['id']}|{hsh(r['prompt'])}" for r in frz)
man["near_duplicate_drop"] = {"applied_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "source": "results/round5/pipeline/curate_report.json (CuratorKIT %s, MinHash Jaccard >= 0.85)" % rep["curatorkit_version"],
                              "dropped": sorted(drop), "n_dropped": len(drop), "frozen_hash_before": man["frozen_ids_sha256"], "frozen_n_before": man["frozen_n"],
                              "kept_with_caveat": {"jbb_JBC_frz": other["frozen_rows_by_set"].get("jbb_JBC_frz"), "jbb_same_method_template_frozen_rows": F["jbb_same_method_template_cross_side"]["frozen_rows_involved_at_threshold"],
                                                   "note": "JBC and random search wrap every behaviour in one fixed published template; development holds the same template (other behaviours, and in-the-wild template %s). Held out by behaviour only. Neither template is in any fit row." % other["development_itw_template_indices"]}}
for k in drop:
    s, i = k.split("|", 1)
    man["ids"][s].remove(i)
    man["new_sizes"][s] -= 1
man["new_rows"] -= len(drop)
man["sides"]["frozen"] = len(frz)
man["frozen_n"], man["frozen_ids_sha256"], man["frozen_ids_and_text_sha256"] = len(frz), hashlib.sha256("\n".join(fk).encode()).hexdigest(), hashlib.sha256("\n".join(ft).encode()).hexdigest()
(OUT / "sets_manifest.json").write_text(json.dumps(man, indent=1))
print(json.dumps({k: v for k, v in man["near_duplicate_drop"].items()}, indent=1))
print("frozen", man["frozen_n"], man["frozen_ids_sha256"], man["frozen_ids_and_text_sha256"], "| rows", len(keep))
