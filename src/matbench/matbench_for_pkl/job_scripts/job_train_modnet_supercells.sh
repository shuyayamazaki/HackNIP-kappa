#!/bin/bash
#SBATCH -p gpu_short                     # Queue/partition: short GPU jobs
#SBATCH --gres=gpu:1                     # Request 1 GPU
##SBATCH --cpus-per-task=12              # (optional) CPU cores per task
#SBATCH -n 1                             # 1 task (no MPI)
#SBATCH -t 4:00:00                       # Walltime limit: 4 hours
#SBATCH -J ORB2_TRAIN_SUPER              # Job name
#SBATCH --output=output_script/%x-%j.out # Stdout log path
#SBATCH --error=output_script/%x-%j.err  # Stderr log path

# Exit on error/undef; fail on pipe errors — safer batch execution
set -euo pipefail

# Create log directory if missing
mkdir -p output_script

# ---------------------- Threading hygiene ----------------------
# Prevent BLAS/MKL/NumExpr from oversubscribing CPU threads per process
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1              # Flush Python output immediately

# ---------------- TensorFlow JIT/XLA controls ------------------
# Keep execution stable by disabling TF JIT/XLA (avoids libdevice/XLA quirks)
export TF_DISABLE_JIT=1
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export TF_CPP_MIN_LOG_LEVEL=2          # Reduce TF log verbosity (optional)
export TF_ENABLE_ONEDNN_OPTS=0         # Disable oneDNN CPU optimizations (optional)

# ---------------------- CUDA toolchain -------------------------
# Load a known-good CUDA toolchain from modules (adjust version to your cluster)
module purge
module load cuda/12.2u2

# Derive CUDA_HOME from nvcc and wire up PATH/LD_LIBRARY_PATH
NVCC_PATH="$(command -v nvcc)"
CUDA_HOME="${NVCC_PATH%/bin/nvcc}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Provide XLA with a libdevice path if you later enable it (currently JIT off)
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

# Quick sanity checks (do not fail the job if unavailable)
nvidia-smi || true
ls "$CUDA_HOME/nvvm/libdevice"/libdevice*.bc 2>/dev/null || \
  echo "[WARN] libdevice not found under $CUDA_HOME/nvvm/libdevice (JIT is OFF; continuing)"

date  # Timestamp (start)

# -------------------- Conda environment ------------------------
# Activate the Python environment that has your dependencies
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hacknip

# ---------------------- Job parameters -------------------------
# Main training script
PY_SCRIPT="src/matbench/train_modnet_from_supercells.py"

# Dataset root prepared by your pipeline
DATA_ROOT="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data"

# Feature/model choices
MLIP="orb2"                             # Feature family (e.g., ORB2)
MODEL="modnet"                          # Regressor (MODNet)

# Dataset slug (matches precomputed feature bundle name)
DATASET_SLUG="random_split_dedup_w_min_freq"

# GPU visibility (single GPU index on the node)
CUDA_VISIBLE="0"

# Reproducibility and splits
SEED="42"
TRAIN_SPLIT="train"
TEST_SPLIT="test"

# Where to store the final trained model artifacts
FINAL_MODEL_DIR="${DATA_ROOT}/feat_${MLIP}/results_${MODEL}/trained_models"

# Ensure features exist before launching (fail early if not found)
if [[ ! -d "${DATA_ROOT}/feat_${MLIP}" ]]; then
  echo "[ERROR] Feature directory not found: ${DATA_ROOT}/feat_${MLIP}" >&2
  exit 1
fi
mkdir -p "${FINAL_MODEL_DIR}"

# ----------------- Environment for Python code -----------------
# These env vars are consumed by the training/benchmark scripts
export BENCH_DATA_DIR="${DATA_ROOT}"
export BENCH_MLIP="${MLIP}"
export BENCH_MODEL="${MODEL}"
export BENCH_TASKS=""                   # Empty = all default tasks, or set explicitly
export BENCH_FEATURE_SLUGS="${DATASET_SLUG}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}"

# Echo runtime context for the logs
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] DATASET_SLUG=${DATASET_SLUG}"
echo "[INFO] MLIP=${MLIP}, MODEL=${MODEL}"
echo "[INFO] TRAIN_SPLIT=${TRAIN_SPLIT}, TEST_SPLIT=${TEST_SPLIT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE}"
echo "[INFO] PY_SCRIPT=${PY_SCRIPT}"
echo "[INFO] TF_DISABLE_JIT=${TF_DISABLE_JIT} | XLA_FLAGS=${XLA_FLAGS}"

# ---------------------- Build the command ----------------------
PY_BIN="$(command -v python)"
CMD=(
  "${PY_BIN}" "${PY_SCRIPT}"
  --feature-slugs "${DATASET_SLUG}"     # Which precomputed feature bundle(s) to use
  --seed "${SEED}"                      # Global RNG seed
  --train-split "${TRAIN_SPLIT}"        # Name of the training split to consume
  --test-split "${TEST_SPLIT}"          # Name of the held-out test split
  --cuda-visible-devices "${CUDA_VISIBLE}"  # Propagate GPU index to Python
  --fast                                # Use fast mode (e.g., fewer CV folds/epochs) if implemented
  --train-final                         # Train the final model on the full train split
  --final-model-path "${FINAL_MODEL_DIR}"   # Directory to save trained model artifacts
)

echo "[INFO] PY_BIN=${PY_BIN}"
echo "[INFO] Command: ${CMD[*]}"

# ---------------------- Execute training -----------------------
"${CMD[@]}"

date  # Timestamp (end)
