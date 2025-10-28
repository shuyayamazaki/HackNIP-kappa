#!/bin/bash
#SBATCH -p gpu_short
#SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=12
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH -J ORB2_TRAIN_SUPER
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

# ---- Disable TF JIT/XLA (libdevice不要で確実に回す) ----
export TF_DISABLE_JIT=1
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export TF_CPP_MIN_LOG_LEVEL=2           # ログ抑制（任意）
export TF_ENABLE_ONEDNN_OPTS=0          # CPU最適化を無効化（任意）

# ---- CUDA toolchain（後でXLA使う場合にも備えてロード） ----
module purge
module load cuda/12.2u2

# CUDA_HOME を nvcc から推定して PATH/LD を整える
NVCC_PATH="$(command -v nvcc)"
CUDA_HOME="${NVCC_PATH%/bin/nvcc}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# XLA が有効なときのための libdevice パス（いまは JIT OFF なので影響なし）
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

# Sanity check
nvidia-smi || true
ls "$CUDA_HOME/nvvm/libdevice"/libdevice*.bc 2>/dev/null || \
  echo "[WARN] libdevice not found under $CUDA_HOME/nvvm/libdevice (JITはOFFなので続行)"

date

# ---- Conda env ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hacknip

# ---- Job params ----
PY_SCRIPT="src/matbench/train_modnet_from_supercells.py"
DATA_ROOT="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data"
MLIP="orb2"
MODEL="modnet"
DATASET_SLUG="random_split_dedup_w_min_freq"
CUDA_VISIBLE="0"
SEED="42"
TRAIN_SPLIT="train"
TEST_SPLIT="test"
FINAL_MODEL_DIR="${DATA_ROOT}/feat_${MLIP}/results_${MODEL}/trained_models"

if [[ ! -d "${DATA_ROOT}/feat_${MLIP}" ]]; then
  echo "[ERROR] Feature directory not found: ${DATA_ROOT}/feat_${MLIP}" >&2
  exit 1
fi
mkdir -p "${FINAL_MODEL_DIR}"

export BENCH_DATA_DIR="${DATA_ROOT}"
export BENCH_MLIP="${MLIP}"
export BENCH_MODEL="${MODEL}"
export BENCH_TASKS=""
export BENCH_FEATURE_SLUGS="${DATASET_SLUG}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}"

echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] DATASET_SLUG=${DATASET_SLUG}"
echo "[INFO] MLIP=${MLIP}, MODEL=${MODEL}"
echo "[INFO] TRAIN_SPLIT=${TRAIN_SPLIT}, TEST_SPLIT=${TEST_SPLIT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE}"
echo "[INFO] PY_SCRIPT=${PY_SCRIPT}"
echo "[INFO] TF_DISABLE_JIT=${TF_DISABLE_JIT} | XLA_FLAGS=${XLA_FLAGS}"

PY_BIN="$(command -v python)"
CMD=(
  "${PY_BIN}" "${PY_SCRIPT}"
  --feature-slugs "${DATASET_SLUG}"
  --seed "${SEED}"
  --train-split "${TRAIN_SPLIT}"
  --test-split "${TEST_SPLIT}"
  --cuda-visible-devices "${CUDA_VISIBLE}"
  --fast
  --train-final
  --final-model-path "${FINAL_MODEL_DIR}"
)

echo "[INFO] PY_BIN=${PY_BIN}"
echo "[INFO] Command: ${CMD[*]}"

"${CMD[@]}"

date
