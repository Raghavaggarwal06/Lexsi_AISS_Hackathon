"""Build the private bundle for round 7 on the Mac:   python3 scripts/r7_bundle.py
Writes colab/r7_bundle_private.zip (paths relative to build/, unzipped into /content/build by scripts/r7_boot.sh) and prints its size and file list.
It holds private prompt rows (sets.jsonl, the format suites, round 4's base replies, data/harmful.jsonl for the leak check): it goes to the owner's
private Drive folder MyDrive/lexsi-knob/round7/ only, never to anything public. Nothing is uploaded by this script."""
import hashlib, sys, zipfile
from pathlib import Path

B = Path(__file__).resolve().parent.parent
OUTZ = B / "colab" / "r7_bundle_private.zip"
SCRIPTS = ["common.py", "05_score.py", "r4lib.py", "r5lib.py", "r6lib.py", "r4_scoreboard.py", "r5_degcheck.py", "r5_report.py", "r6_report.py",
           "r7lib.py", "r7gpu.py", "r7_safetune.py", "r7_boot.sh", "r7_queue.sh", "r7_pack.py", "r7_bundle.py", "pipeline/check_no_harmful_text.py"]
FORMATS = ["xa_sysA", "xa_sysB", "xa_sysC", "xa_sysD", "xa_mt2", "xa_mtref", "xa_mtsys", "xa_itwsys"]


def files():
    out = [(B / "scripts" / s, f"scripts/{s}") for s in SCRIPTS]
    out += [(B / "data" / f, f"data/{f}") for f in ("harmful.jsonl", "xstest_unsafe.jsonl")]  # the leak check's private sources
    out += [(B / "results/round5" / f, f"results/round5/{f}") for f in ("lab/v3g.json", "frozen_declaration.json", "sets_manifest.json", "tables/frz_tight.json")]
    out += [(p, f"results/round5/labels/{p.name}") for p in sorted((B / "results/round5/labels").glob("*.json"))]
    out += [(B / "results/round6" / f, f"results/round6/{f}") for f in ("sets_manifest.json", "exam_formats.json", "tables/xb_tight.json")]
    out += [(p, f"results/round6/labels/{p.name}") for p in sorted((B / "results/round6/labels").glob("*.json"))]
    out += [(B / "results/round7" / f, f"results/round7/{f}") for f in ("calib.json", "journal.md", "RUNBOOK.md")]
    out += [(B / "runs_private/round6/sets.jsonl", "runs_private/round7/sets.jsonl")]  # round 6's sets file, unchanged (sha256 checked by r7lib)
    out += [(B / "runs_private/round7/suites" / f"xbf_{f}.jsonl", f"runs_private/round7/suites/xbf_{f}.jsonl") for f in FORMATS]
    out += [(B / "runs_private/round7/ref/base_wjx_r4.jsonl", "runs_private/round7/ref/base_wjx_r4.jsonl")]  # round 4's Base replies, for the reproduction check
    return out


def main():
    fl = files()
    missing = [str(s) for s, _ in fl if not s.exists()]
    if missing:
        sys.exit("missing: " + ", ".join(missing))
    OUTZ.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUTZ, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for src, arc in fl:
            z.write(src, arc)
    groups = {}
    for src, arc in fl:
        g = arc.rsplit("/", 1)[0] if arc.count("/") >= 2 and ("labels/" in arc) else arc
        groups.setdefault(g, [0, 0])
        groups[g][0] += 1
        groups[g][1] += src.stat().st_size
    h = hashlib.sha256(OUTZ.read_bytes()).hexdigest()
    print(f"{OUTZ.relative_to(B)}: {OUTZ.stat().st_size / 2**20:.1f} MiB, {len(fl)} files, sha256 {h[:16]}")
    for g, (n, sz) in groups.items():
        print(f"  {g}" + (f"  ({n} files)" if n > 1 else "") + f"  {sz / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
