#!/bin/bash
# Train + release first (what the model's runs need), then the ground-truth behaviour evals.
#   nohup bash tasks/sycophancy/build_bg.sh [gpu] &
cd "$HOME/wsp-safety"
PY=$HOME/.conda/envs/mech-interp/bin/python
export CUDA_VISIBLE_DEVICES=${1:-0} HF_HOME=/scratch/scholar/pande1/hf_cache TOKENIZERS_PARALLELISM=false
L=/scratch/scholar/pande1/logs
$PY tasks/sycophancy/build.py --stages train,release > $L/wsp_syc_build.log 2>&1 || { echo "BUILD FAILED" >> $L/wsp_syc_build.log; exit 1; }
echo "build done" >> $L/wsp_syc_build.log
$PY tasks/sycophancy/build.py --stages truth > $L/wsp_syc_truth.log 2>&1 && echo "truth done" >> $L/wsp_syc_truth.log || echo "TRUTH FAILED" >> $L/wsp_syc_truth.log
