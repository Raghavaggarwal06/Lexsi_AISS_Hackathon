#!/bin/bash
#SBATCH -p cpu
#SBATCH -A scholar
#SBATCH -c 16
#SBATCH --mem=60G
#SBATCH -t 4:00:00
#SBATCH -o /scratch/scholar/pande1/logs/wsp_%x_%j.log
#SBATCH --signal=B:USR1@600
# Run several WSP conversations in parallel in one CPU job (Codex calls need no GPU; the 270M attention tool runs on
# CPU). Conversations resume from state.json, so 10 min before the 4 h wall limit the job resubmits itself with the
# same commands; finished conversations return at once on resubmission.
#   sbatch -J syc_attn scripts/conv_job.sh "python tasks/sycophancy/run_case.py --round attention_tool --case pair_02df1c" ...
# "python" in the commands is the mech-interp env's interpreter.
cd "$HOME/wsp-safety"
export PATH="$HOME/.conda/envs/mech-interp/bin:$HOME/.local/bin:$PATH" HF_HOME=/scratch/scholar/pande1/hf_cache \
       TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=
resubmit() {
  echo "[$(date)] wall limit near: resubmitting"
  sbatch -J "$SLURM_JOB_NAME" scripts/conv_job.sh "$@"
  kill $(jobs -p) 2>/dev/null
  exit 0
}
trap 'resubmit "$@"' USR1
for c in "$@"; do
  echo "[$(date)] start: $c"
  bash -c "$c" &
done
wait
echo "[$(date)] all done"
