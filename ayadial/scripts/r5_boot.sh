#!/bin/bash
# Restore the round 5 working tree on a fresh runtime from the private Drive folder. Run once after the SETUP cell (token + Drive mount).
set -e
D=/content/drive/MyDrive/lexsi-knob
B=${1:-round5_bundle_v4.zip}
mkdir -p /content/build && cd /content/build
unzip -qo $D/round4/round4_final.zip 'results/*' 'runs_private/*'
unzip -qo $D/round5/$B   # after round 4, so the Mac's scripts and the round 5 sets win
mkdir -p results/round5 runs_private/round5/gen cache_r5 specs5 $D/round5
# anything a previous round 5 session already mirrored to Drive comes back (resumable runs)
if [ -d $D/round5/results ]; then cp -rn $D/round5/results/. results/round5/; fi
if [ -d $D/round5/runs_private ]; then cp -rn $D/round5/runs_private/. runs_private/round5/; fi
if [ -d $D/round5/cache ]; then cp -n $D/round5/cache/*.npz cache_r5/ 2>/dev/null || true; fi
ls; ls results/round5 | head -20; wc -l runs_private/round5/sets.jsonl; date -u
