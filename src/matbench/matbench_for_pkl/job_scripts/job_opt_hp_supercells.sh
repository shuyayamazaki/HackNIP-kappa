#!/bin/bash
#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=12
#SBATCH -n 1
#SBATCH -t 100:00:00
#SBATCH -J ORB2_TUNE_SUPER
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
ls "$CUDA_HOME/nvvm/libdevice"/libdevice*.bc 2>/dev/null || \
  echo "[WARN] libdevice not found under $CUDA_HOME/nvvm/libdevice"

date

# ---- Conda env ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hacknip

# ---- Job params ----
PY_SCRIPT="src/matbench/opt_hp_modnet_from_supercells.py"
DATA_ROOT="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data"
MLIP="orb2"
MODEL="modnet"
DATASET_SLUG="ood_split_dedup_w_min_freq"
TRAIN_SPLIT="train"
TEST_SPLIT="test"
CUDA_VISIBLE="0"
SEED="42"
N_TRIALS="60"
CV_FOLDS="5"
N_JOBS="1"
SAMPLER="tpe"
EARLY_STOP="20"
SPLIT_TYPE="ood"
TIME="20251027_094124"
LAYER="l8"
# Path to the model produced by train_modnet_from_supercells.py
TRAINED_MODEL_PATH="${DATA_ROOT}/feat_${MLIP}/results_${MODEL}/training_dataset_${SPLIT_TYPE}_split_dedup_w_min_freq_${TIME}/trained_models/layers_${TRAIN_SPLIT}2${TEST_SPLIT}/${SPLIT_TYPE}_split_dedup_w_min_freq_${TRAIN_SPLIT}2${TEST_SPLIT}_XPS_${MLIP}_${LAYER}.modnet"
TARGET_LAYER="8"

export BENCH_DATA_DIR="${DATA_ROOT}"
export BENCH_MLIP="${MLIP}"
export BENCH_MODEL="${MODEL}"
export BENCH_TASKS=""
export BENCH_FEATURE_SLUGS="${DATASET_SLUG}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}"

echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] DATASET_SLUG=${DATASET_SLUG}"
echo "[INFO] TRAIN_SPLIT=${TRAIN_SPLIT}, TEST_SPLIT=${TEST_SPLIT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE}"
echo "[INFO] PY_SCRIPT=${PY_SCRIPT}"
echo "[INFO] N_TRIALS=${N_TRIALS}, CV_FOLDS=${CV_FOLDS}, EARLY_STOP=${EARLY_STOP}"
echo "[INFO] TRAINED_MODEL_PATH=${TRAINED_MODEL_PATH}"
echo "[INFO] TARGET_LAYER=${TARGET_LAYER}"

PY_BIN="$(command -v python)"
CMD=(
  "${PY_BIN}" "${PY_SCRIPT}"
  --feature-slugs "${DATASET_SLUG}"
  --train-split "${TRAIN_SPLIT}"
  --test-split "${TEST_SPLIT}"
  --seed "${SEED}"
  --cuda-visible-devices "${CUDA_VISIBLE}"
  --n-trials "${N_TRIALS}"
  --cv-folds "${CV_FOLDS}"
  --n-jobs "${N_JOBS}"
  --sampler "${SAMPLER}"
  --early-stop-trials "${EARLY_STOP}"
  --trained-model-path "${TRAINED_MODEL_PATH}"
  --layer "${TARGET_LAYER}"
)

echo "[INFO] PY_BIN=${PY_BIN}"
echo "[INFO] Command: ${CMD[*]}"

"${CMD[@]}"

date
