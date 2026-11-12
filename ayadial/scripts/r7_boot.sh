#!/bin/bash
# Round 7 on a fresh Colab runtime (Drive mounted at /content/drive, HF_TOKEN set). Idempotent: run it again after a recycled runtime.
# 1. unpack the private bundle into /content/build; 2. bring back whatever an earlier round 7 session mirrored to Drive (labels, replies,
# judge cache, calibration states, pick file), never overwriting a local file; 3. install SafeTune 0.1.13 and the transformers version of
# rounds 5 and 6; 4. re-check the calibration set and the test id lists (stage calib). Usage: bash scripts/r7_boot.sh [bundle name]
set -e
D=/content/drive/MyDrive/lexsi-knob/round7
B=${1:-r7_bundle_private.zip}
mkdir -p /content/build && cd /content/build
unzip -qo "$D/$B"
mkdir -p results/round7 runs_private/round7/logs cache_r7
if [ -d "$D/results" ]; then cp -rn "$D/results/." results/round7/; fi
if [ -d "$D/runs_private" ]; then cp -rn "$D/runs_private/." runs_private/round7/; fi
pip -q install "safetune==0.1.13" "transformers==5.22.1" scikit-learn 2>&1 | grep -v -i "warning" | tail -3 || true
python -c "import torch, transformers, safetune; print('torch', torch.__version__, '| transformers', transformers.__version__, '| safetune', safetune.__version__, '|', torch.cuda.get_device_name(0))"
echo "sets.jsonl $(sha256sum runs_private/round7/sets.jsonl | cut -c1-16) (expected 27a89c7b8b8fc99c)"
python scripts/r7_safetune.py calib
python scripts/r7_safetune.py status | tail -4
date -u
echo R7_BOOT DONE
