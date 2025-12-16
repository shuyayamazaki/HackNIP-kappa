#!/bin/bash
#SBATCH -p gpu_short
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH -J ORB2_PRED_SUPER_REG
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

# ---- Disable TF JIT/XLA ----
export TF_DISABLE_JIT=1
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

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
  echo "[WARN] libdevice not found under $CUDA_HOME/nvvm/libdevice (JIT is OFF)"

date

# ---- Conda env ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hacknip

# ---- Job params (edit below) ----
PY_SCRIPT="../6_predict_from_packed_supercell_features_regression.py"
MODEL_PATH="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data/feat_orb2/results_modnet/best_models/random_split_dedup_w_min_freq/train2test/l11/model.modnet"  # regression .modnet saved by 3_train_modnet_from_supercells.py
FEATURE_PATH=""  # packed pickle mode (if using packed features)
NPY_PATH="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data/feat_orb2/npy/lemat_bulk_csx_batch1_all_XPS_l11.npy"  # raw features
META_PICKLE="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data/metadata/lemat_bulk_csx_batch1_meta.pkl"   # optional meta pickle (mp_ids, generation_id, targets) to pair with NPY
MP_IDS_PATH=""  # optional text file with mp_ids (overrides meta mp_ids)
META_ID_KEY="ids"  # key inside META_PICKLE to pull ids from (fallback to mp_ids)
ID_COLUMN_NAME="immutable_id"  # customize output id column header

SPLIT=""    # used for packed pickle or when meta pickle has splits
LAYER=""        # leave empty to infer, or set e.g. "11"
KEY="XPS"
TARGET_NAME=""  # optional override, else inferred or defaults to "g"
OUTPUT_CSV=""   # optional explicit output path
CUDA_VISIBLE="0"

# ---- Validation ----
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[ERROR] Model file not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ -z "${FEATURE_PATH}" && -z "${NPY_PATH}" ]]; then
  echo "[ERROR] Specify either FEATURE_PATH or NPY_PATH." >&2
  exit 1
fi
if [[ -n "${FEATURE_PATH}" && -n "${NPY_PATH}" ]]; then
  echo "[ERROR] Use only one of FEATURE_PATH or NPY_PATH (mutually exclusive for prediction input)." >&2
  exit 1
fi
if [[ -n "${FEATURE_PATH}" && ! -f "${FEATURE_PATH}" ]]; then
  echo "[ERROR] Feature pickle not found: ${FEATURE_PATH}" >&2
  exit 1
fi
if [[ -n "${NPY_PATH}" && ! -f "${NPY_PATH}" ]]; then
  echo "[ERROR] Feature .npy not found: ${NPY_PATH}" >&2
  exit 1
fi
if [[ -n "${META_PICKLE}" && ! -f "${META_PICKLE}" ]]; then
  echo "[ERROR] Meta pickle not found: ${META_PICKLE}" >&2
  exit 1
fi
if [[ -n "${MP_IDS_PATH}" && ! -f "${MP_IDS_PATH}" ]]; then
  echo "[ERROR] mp_ids file not found: ${MP_IDS_PATH}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE}"

echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] FEATURE_PATH=${FEATURE_PATH:-<unused>}"
echo "[INFO] NPY_PATH=${NPY_PATH:-<unused>}"
echo "[INFO] META_PICKLE=${META_PICKLE:-<none>}"
echo "[INFO] MP_IDS_PATH=${MP_IDS_PATH:-<none>}"
echo "[INFO] META_ID_KEY=${META_ID_KEY:-mp_ids}"
echo "[INFO] ID_COLUMN_NAME=${ID_COLUMN_NAME}"
echo "[INFO] SPLIT=${SPLIT} | LAYER=${LAYER:-auto} | KEY=${KEY}"
echo "[INFO] TARGET_NAME=${TARGET_NAME:-auto/default}"
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV:-auto}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE}"
echo "[INFO] PY_SCRIPT=${PY_SCRIPT}"
echo "[INFO] TF_DISABLE_JIT=${TF_DISABLE_JIT} | XLA_FLAGS=${XLA_FLAGS}"

PY_BIN="$(command -v python)"
CMD=(
  "${PY_BIN}" "${PY_SCRIPT}"
  --model "${MODEL_PATH}"
  --key "${KEY}"
)

if [[ -n "${FEATURE_PATH}" ]]; then
  CMD+=("--features" "${FEATURE_PATH}" "--split" "${SPLIT}")
else
  CMD+=("--npy" "${NPY_PATH}")
  if [[ -n "${META_PICKLE}" ]]; then
    CMD+=("--meta-pickle" "${META_PICKLE}" "--split" "${SPLIT}")
    if [[ -n "${META_ID_KEY}" ]]; then
      CMD+=("--meta-id-key" "${META_ID_KEY}")
    fi
  fi
  if [[ -n "${MP_IDS_PATH}" ]]; then
    CMD+=("--mp-ids-path" "${MP_IDS_PATH}")
  fi
fi
if [[ -n "${OUTPUT_CSV}" ]]; then
  CMD+=("--output" "${OUTPUT_CSV}")
fi
if [[ -n "${ID_COLUMN_NAME}" ]]; then
  CMD+=("--id-column" "${ID_COLUMN_NAME}")
fi
if [[ -n "${LAYER}" ]]; then
  CMD+=("--layer" "${LAYER}")
fi
if [[ -n "${TARGET_NAME}" ]]; then
  CMD+=("--target-name" "${TARGET_NAME}")
fi

echo "[INFO] PY_BIN=${PY_BIN}"
echo "[INFO] Command: ${CMD[*]}"

"${CMD[@]}"

date
