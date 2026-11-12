#!/bin/bash
# Round 6 start on the runtime: copy (never move) what round 5 produced into the round 6 folders, so that round 5 stays as it was packed.
cd /content/build || exit 1
mkdir -p results/round6/labels results/round6/lax runs_private/round6/gen runs_private/round6/suites cache_r6 specs6
cp -n runs_private/round5/sets.jsonl runs_private/round6/sets.jsonl
cp -n runs_private/round5/gen/*.jsonl runs_private/round6/gen/
cp -rn runs_private/round5/suites/. runs_private/round6/suites/
cp -n runs_private/round5/scores*.jsonl runs_private/round6/ 2>/dev/null
cp -n results/round5/labels/*.json results/round6/labels/
for j in wg q8; do mkdir -p results/round6/labels_$j; cp -n results/round5/labels_$j/*.json results/round6/labels_$j/; done
cp -n results/round5/lax/exposure_*.json results/round6/lax/
cp -n cache_r5/*.npz cache_r6/
sha256sum cache_r6/vec_v3g.npz cache_r6/vec_v1.npz cache_r6/vec_v2.npz | cut -c1-16,65- 
ls runs_private/round6/gen | wc -l; wc -l < runs_private/round6/sets.jsonl; du -sh cache_r6 | cut -f1
echo R6_BOOT DONE
