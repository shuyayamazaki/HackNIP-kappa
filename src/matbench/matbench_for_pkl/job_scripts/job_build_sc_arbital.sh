#!/bin/bash
#SBATCH -p gpu_long                 # If GPU is not needed, you can switch to a CPU partition (e.g., cpu_short)
##SBATCH --gres=gpu:1                # Uncomment if a GPU is required
#SBATCH --cpus-per-task=52           # CPU-intensive job: CIF parsing and supercell generation
#SBATCH -n 1
#SBATCH -t 15:00:00
#SBATCH -J SUPER_XPS_ARB             # Job name
#SBATCH --output=output_script/%x-%j.out
#SBATCH --error=output_script/%x-%j.err

set -euo pipefail

# Thread control (prevent oversubscription)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Log output directory
mkdir -p output_script

# (Optional) Check GPU info / CUDA modules (harmless even if GPU is unused)
nvidia-smi || true
module avail cuda || true

date

# ==== Activate environment (modify if needed for your setup) ====
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hacknip

# ==== Input / Output paths ====
# CSV or pickle containing CIF strings (must include a 'cif' column)
INPUT_PATH="/work/y-tomiya/ntu/Dataset_thermoconductivity_pred/other_datasets/lemat/lemat_bulk_csx_batch1.pkl"
# Optional: column name containing unique IDs (if omitted, the table index is used)
ID_COLUMN="immutable_id"

# Output root directory (outputs will be placed under metadata/structures).
# If unset, defaults to <input file directory>/benchmark_data/
OUTPUT_DIR="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data/structures"

# Target minimum supercell vector length (Å) for --target-length
TARGET_LENGTH="10.0"

# Optional: explicitly set the dataset name (if unset, inferred from the CSV/pickle name)
DATASET_SLUG="lemat_bulk_csx_batch1"

# ==== Execution command ====
PY_SCRIPT="../1_1_build_supercelss_from_pkl_arbital.py"

CMD=( python "${PY_SCRIPT}"
  --input-path "${INPUT_PATH}"
  --target-length "${TARGET_LENGTH}"
  # --skip-base-traj                # Uncomment to skip output of base structures (_XP.traj)
)

# Append optional arguments only if specified
if [[ -n "${ID_COLUMN}" ]]; then
  CMD+=( --id-column "${ID_COLUMN}" )
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=( --output-dir "${OUTPUT_DIR}" )
fi
if [[ -n "${DATASET_SLUG}" ]]; then
  CMD+=( --dataset-slug "${DATASET_SLUG}" )
fi

echo "[INFO] Command: ${CMD[*]}"
echo "[INFO] Output root (metadata/structures): ${OUTPUT_DIR:-<input_dir>/benchmark_data}"

# Run as a single task with srun
srun --cpu-bind=cores "${CMD[@]}"

date