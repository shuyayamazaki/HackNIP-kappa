#!/bin/bash
# Full pipeline: build supercells -> featurize -> predict (regression) in one job.
# Edit the parameters in the "User parameters" block for your dataset/model.

#SBATCH -p gpu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH -n 1
#SBATCH -t 72:00:00
#SBATCH -J SC_PIPELINE_REG
#SBATCH --output=output_script/%x-%j.out
#SBATCH --error=output_script/%x-%j.err

set -euo pipefail

# Thread control
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p output_script

# CUDA toolchain (needed for ORB2 featurization/prediction)
module purge
module load cuda/12.2u2
nvidia-smi || true

date

# Conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hacknip

# ===== User parameters =====
# Stage 1: build supercells
INPUT_PATH="/work/y-tomiya/ntu/Dataset_thermoconductivity_pred/other_datasets/lemat/lemat_bulk_csx_batch1.pkl"
ID_COLUMN="immutable_id"
TARGET_LENGTH="10.0"
DATASET_SLUG="lemat_bulk_csx_batch1"
DATA_ROOT="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data"
OUTPUT_STRUCT_DIR="${DATA_ROOT}/structures"
SKIP_BASE_TRAJ=""   # set to "--skip-base-traj" to skip writing *_XP.traj

# Stage 2: featurize supercells
MLIP="orb2"
LAYERS="11"         # comma-separated
DEVICE="auto"       # auto | cpu | cuda:0, etc.
OVERWRITE=""        # set to "--overwrite" to recompute features

# Stage 3: predict
MODEL_PATH="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data/feat_orb2/results_modnet/best_models/random_split_dedup_w_min_freq/train2test/l11/model.modnet"
KEY="XPS"
SPLIT="test"
TARGET_NAME=""      # optional override; else inferred/default "g"
META_PICKLE=""      # optional override; default auto-loads <data_root>/metadata/<slug>_meta.pkl
META_ID_KEY="ids"
ID_COLUMN_NAME="immutable_id"
PREDICTION_COLUMN="predicted_log_klat(hacknip)"
MP_IDS_PATH=""      # optional mp_ids text file (one per line)
SKIP_TARGET="false"
SKIP_GENERATION="false"
LAYERS_LABEL="${LAYERS//,/_}"
OUTPUT_CSV="${DATA_ROOT}/feat_${MLIP}/npy/${DATASET_SLUG}_pred_l${LAYERS_LABEL}.csv"

# ===== Validation & derived paths =====
if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "[ERROR] INPUT_PATH not found: ${INPUT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}" >&2
  exit 1
fi
mkdir -p "${DATA_ROOT}/metadata" "${DATA_ROOT}/structures" "${DATA_ROOT}/feat_${MLIP}/npy"

TRAJ_PATH="${DATA_ROOT}/structures/${DATASET_SLUG}_all_${KEY}.traj"
DEFAULT_META="${DATA_ROOT}/metadata/${DATASET_SLUG}_meta.pkl"

echo "[INFO] INPUT_PATH=${INPUT_PATH}"
echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] DATASET_SLUG=${DATASET_SLUG}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] LAYERS=${LAYERS}"
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV}"

# ===== Stage 1: build supercells =====
BUILD_PY="../1_1_build_supercelss_from_pkl_arbital.py"
BUILD_CMD=( python "${BUILD_PY}" --input-path "${INPUT_PATH}" --target-length "${TARGET_LENGTH}" )
if [[ -n "${ID_COLUMN}" ]]; then
  BUILD_CMD+=( --id-column "${ID_COLUMN}" )
fi
if [[ -n "${OUTPUT_STRUCT_DIR}" ]]; then
  BUILD_CMD+=( --output-dir "${OUTPUT_STRUCT_DIR}" )
fi
if [[ -n "${DATASET_SLUG}" ]]; then
  BUILD_CMD+=( --dataset-slug "${DATASET_SLUG}" )
fi
if [[ -n "${SKIP_BASE_TRAJ}" ]]; then
  BUILD_CMD+=( ${SKIP_BASE_TRAJ} )
fi

echo "[INFO] Stage 1 command: ${BUILD_CMD[*]}"
srun --cpu-bind=cores "${BUILD_CMD[@]}"

if [[ ! -f "${TRAJ_PATH}" ]]; then
  echo "[ERROR] Expected trajectory not found after stage 1: ${TRAJ_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DEFAULT_META}" ]]; then
  echo "[ERROR] Expected meta pickle not found after stage 1: ${DEFAULT_META}" >&2
  exit 1
fi

# ===== Stage 2: featurize =====
FEAT_PY="../2_1_featurize_construct_from_supercells_arbital.py"
FEAT_CMD=( python "${FEAT_PY}" --slug "${DATASET_SLUG}" --data-root "${DATA_ROOT}" --mlip "${MLIP}" --layers "${LAYERS}" --device "${DEVICE}" )
if [[ -n "${OVERWRITE}" ]]; then
  FEAT_CMD+=( --overwrite )
fi

echo "[INFO] Stage 2 command: ${FEAT_CMD[*]}"
srun --cpu-bind=cores "${FEAT_CMD[@]}"

# ===== Stage 3: predict =====
PREDICT_PY="../6_predict_from_packed_supercell_features_regression.py"
PRED_CMD=( python "${PREDICT_PY}" --model "${MODEL_PATH}" --slug "${DATASET_SLUG}" --layers "${LAYERS}" --mlip "${MLIP}" --data-root "${DATA_ROOT}" --key "${KEY}" )
if [[ -n "${SPLIT}" ]]; then
  PRED_CMD+=( --split "${SPLIT}" )
fi
if [[ -n "${META_PICKLE}" ]]; then
  PRED_CMD+=( --meta-pickle "${META_PICKLE}" )
fi
if [[ -n "${META_ID_KEY}" ]]; then
  PRED_CMD+=( --meta-id-key "${META_ID_KEY}" )
fi
if [[ -n "${ID_COLUMN_NAME}" ]]; then
  PRED_CMD+=( --id-column "${ID_COLUMN_NAME}" )
fi
if [[ -n "${TARGET_NAME}" ]]; then
  PRED_CMD+=( --target-name "${TARGET_NAME}" )
fi
if [[ -n "${OUTPUT_CSV}" ]]; then
  PRED_CMD+=( --output "${OUTPUT_CSV}" )
fi
if [[ -n "${PREDICTION_COLUMN}" ]]; then
  PRED_CMD+=( --prediction-column "${PREDICTION_COLUMN}" )
fi
if [[ -n "${MP_IDS_PATH}" ]]; then
  PRED_CMD+=( --mp-ids-path "${MP_IDS_PATH}" )
fi
if [[ "${SKIP_TARGET}" == "true" ]]; then
  PRED_CMD+=( --skip-target )
fi
if [[ "${SKIP_GENERATION}" == "true" ]]; then
  PRED_CMD+=( --skip-generation )
fi

echo "[INFO] Stage 3 command: ${PRED_CMD[*]}"
srun --cpu-bind=cores "${PRED_CMD[@]}"

date
