#!/bin/bash
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=4
#SBATCH -n 1
#SBATCH -t 1:00:00
#SBATCH -J ORB2_EVAL_TRANSFER
#SBATCH --output=output_script/%x-%j.out
#SBATCH --error=output_script/%x-%j.err

set -euo pipefail

mkdir -p output_script

# ---- Threading hygiene ----
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ---- CUDA toolchain (evaluation still runs on GPU-capable nodes) ----
module purge
module load cuda/12.2u2
NVCC_PATH="$(command -v nvcc)"
CUDA_HOME="${NVCC_PATH%/bin/nvcc}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

nvidia-smi || true

date

# ---- Conda env ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hacknip

# ---- Job params ----
PY_SCRIPT="src/matbench/evaluate_hp_model_transfer.py"
DATA_ROOT="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data"
MLIP="orb2"
MODEL="modnet"
TARGET_SLUGS="ood_split_dedup_w_min_freq,space_group_split_dedup_w_min_freq"
TEST_SPLIT="test"
CUDA_VISIBLE="0"
SEED="42"

# Path to the model saved by opt_hp_modnet_from_supercells.py (update as needed)
MODEL_PATH="${DATA_ROOT}/feat_${MLIP}/results_${MODEL}/best_models/random_split_dedup_w_min_freq/train2test/l11_v3/model.modnet"
METADATA_PATH="$(dirname "${MODEL_PATH}")/metadata.json"

export BENCH_DATA_DIR="${DATA_ROOT}"
export BENCH_MLIP="${MLIP}"
export BENCH_MODEL="${MODEL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}"

echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] TARGET_SLUGS=${TARGET_SLUGS}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] METADATA_PATH=${METADATA_PATH}"
echo "[INFO] TEST_SPLIT=${TEST_SPLIT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE}"
echo "[INFO] PY_SCRIPT=${PY_SCRIPT}"

PY_BIN="$(command -v python)"
CMD=(
  "${PY_BIN}" "${PY_SCRIPT}"
  --model-path "${MODEL_PATH}"
  --metadata-path "${METADATA_PATH}"
  --target-slugs "${TARGET_SLUGS}"
  --test-split "${TEST_SPLIT}"
  --seed "${SEED}"
  --cuda-visible-devices "${CUDA_VISIBLE}"
)

echo "[INFO] PY_BIN=${PY_BIN}"
echo "[INFO] Command: ${CMD[*]}"

"${CMD[@]}"

date
