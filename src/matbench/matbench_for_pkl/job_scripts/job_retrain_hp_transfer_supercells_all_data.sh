#!/bin/bash
#SBATCH -p gpu_short
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH -J ORB2_RETRAIN_TRANSFER_ALL
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

# ---- CUDA toolchain ----
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
PY_SCRIPT="../5_1_all_data_retrain_hp_model_transfer.py"
DATA_ROOT="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data"
MLIP="orb2"
MODEL="modnet"
TARGET_SLUGS="random_split_dedup_w_min_freq"
TRAIN_SPLIT="train"
TEST_SPLIT="test"
CUDA_VISIBLE="0"
SEED="42"
SAVE_MODELS="--save-models"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
# Optional explicit output subdir (relative to results_modnet/...), keeps runs separated by timestamp.
OUTPUT_DIR="hp_transfer_retrain_all_data/run_${TIMESTAMP}"

# Path to metadata produced by retrain_best_modnet_from_summary.py for the best model
METADATA_PATH="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data/feat_orb2/results_modnet/best_models/random_split_dedup_w_min_freq/train2test/l11/metadata.json"

export BENCH_DATA_DIR="${DATA_ROOT}"
export BENCH_MLIP="${MLIP}"
export BENCH_MODEL="${MODEL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}"

echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] TARGET_SLUGS=${TARGET_SLUGS}"
echo "[INFO] METADATA_PATH=${METADATA_PATH}"
echo "[INFO] TRAIN_SPLIT=${TRAIN_SPLIT}, TEST_SPLIT=${TEST_SPLIT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE}"
echo "[INFO] PY_SCRIPT=${PY_SCRIPT}"

PY_BIN="$(command -v python)"
CMD=(
  "${PY_BIN}" "${PY_SCRIPT}"
  --metadata-path "${METADATA_PATH}"
  --target-slugs "${TARGET_SLUGS}"
  --train-split "${TRAIN_SPLIT}"
  --test-split "${TEST_SPLIT}"
  --seed "${SEED}"
  --cuda-visible-devices "${CUDA_VISIBLE}"
  --output-dir "${OUTPUT_DIR}"
  ${SAVE_MODELS}
)

echo "[INFO] PY_BIN=${PY_BIN}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] Command: ${CMD[*]}"

"${CMD[@]}"

date
