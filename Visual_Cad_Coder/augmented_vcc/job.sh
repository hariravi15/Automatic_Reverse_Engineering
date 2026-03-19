#!/bin/bash
#SBATCH --account==westai0070
#SBATCH --job-name=mv-mask2former
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dc-gpu          # <-- change to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p logs

# --- Activate environment (choose one and edit) ---
# (A) conda
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate myenv
# (B) venv
source ~/.venv/bin/activate

# Optional: reduce tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# Optional: put HF cache on scratch to avoid hammering $HOME
# export HF_HOME=/scratch/$USER/hf
# export TRANSFORMERS_CACHE=/scratch/$USER/hf/transformers

# ---- EDIT THESE PATHS ----
DATA_ROOT="C:/Users/mb01/Desktop/Treburi/Vision/CAD/Sample_dataset_output"
OUT_DIR="${SLURM_SUBMIT_DIR}/runs/mv_mask2former_${SLURM_JOB_ID}"

mkdir -p "${OUT_DIR}"

python -u train_multiview_mask2former.py \
  --data_root "${DATA_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --epochs 20 \
  --batch_size 2 \
  --lr 1e-4 \
  --image_size 512 \
  --test_ratio 0.2 \
  --seed 0 \
  --num_workers "${SLURM_CPUS_PER_TASK}" \
  --device cuda

echo "Training finished."
echo "Results:"
echo "  ${OUT_DIR}/final_metrics.json"
echo "  ${OUT_DIR}/train_metrics.jsonl"
echo "  ${OUT_DIR}/eval_metrics.jsonl"
echo "  ${OUT_DIR}/final/"