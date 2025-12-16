#!/bin/bash
#SBATCH -p cluster_long                 # GPU is recommended for ORB2 inference (switch to a CPU partition if unavailable)
##SBATCH --gres=gpu:1                   # Uncomment if you want to use a GPU
#SBATCH --cpus-per-task=16              # CPUs for I/O and preprocessing
#SBATCH -n 1
#SBATCH -t 100:00:00
#SBATCH -J ORB2_FEAT_ARB                # Job name
#SBATCH --output=output_script/%x-%j.out
#SBATCH --error=output_script/%x-%j.err

set -euo pipefail

# Thread control (avoid excessive parallelism)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Log output directory
mkdir -p output_script

# (Optional) Check GPU info / available CUDA modules
nvidia-smi || true
module avail cuda || true

date

# ==== Activate environment ====
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hacknip

# ==== I/O and parameter settings ====
PY_SCRIPT="../2_1_featurize_construct_from_supercells_arbital.py"
# slug / data root produced by 1_1_build_supercelss_from_pkl_arbital.py
DATA_ROOT="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data"
SLUG="lemat_bulk_csx_batch1"            # points to <slug>_all_XPS.traj and <slug>_meta.pkl
MLIP="orb2"
LAYERS="11"                             # layers to compute (comma-separated)
DEVICE="auto"                           # "auto" | "cpu" | "cuda:0", etc.
OVERWRITE=""                            # set to "--overwrite" if you want to enable overwriting

if [[ ! -f "${DATA_ROOT}/structures/${SLUG}_all_XPS.traj" ]]; then
  echo "[ERROR] Supercell trajectory not found: ${DATA_ROOT}/structures/${SLUG}_all_XPS.traj" >&2
  exit 1
fi
if [[ ! -f "${DATA_ROOT}/metadata/${SLUG}_meta.pkl" ]]; then
  echo "[ERROR] Metadata not found: ${DATA_ROOT}/metadata/${SLUG}_meta.pkl" >&2
  exit 1
fi

echo "[INFO] DATA_ROOT=${DATA_ROOT}"
echo "[INFO] SLUG=${SLUG}"
echo "[INFO] MLIP=${MLIP}"
echo "[INFO] LAYERS=${LAYERS}"
echo "[INFO] DEVICE=${DEVICE}"
echo "[INFO] PY_SCRIPT=${PY_SCRIPT}"

CMD=( python "${PY_SCRIPT}"
  --slug "${SLUG}"
  --data-root "${DATA_ROOT}"
  --mlip "${MLIP}"
  --layers "${LAYERS}"
  --device "${DEVICE}"
)

if [[ -n "${OVERWRITE}" ]]; then
  CMD+=( --overwrite )
fi

echo "[INFO] Command: ${CMD[*]}"

# Run with srun
srun --cpu-bind=cores "${CMD[@]}"

date