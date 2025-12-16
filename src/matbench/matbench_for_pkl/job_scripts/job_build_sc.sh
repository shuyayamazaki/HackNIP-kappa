#!/bin/bash
#SBATCH -p gpu_long                 # If GPU is not needed, you can switch to a CPU partition (e.g., cpu_short)
##SBATCH --gres=gpu:1                # Uncomment if a GPU is required
#SBATCH --cpus-per-task=52           # This job is CPU-centric (mainly I/O-bound). Reduce if excessive.
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH -J SUPER_XPS                 # Job name
#SBATCH --output=output_script/%x-%j.out
#SBATCH --error=output_script/%x-%j.err

set -euo pipefail

# Thread control (prevent oversubscription)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1   # Flush Python output immediately for real-time logging

# Log output directory
mkdir -p output_script

# (Optional) Check GPU info / CUDA modules (harmless even if GPU is unused)
nvidia-smi || true
module avail cuda || true

date  # Print start timestamp

# ==== Activate environment (modify if needed for your setup) ====
# When using conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hacknip

# ==== Input / Output paths ====
PICKLE_PATH="/work/y-tomiya/ntu/Dataset_thermoconductivity_pred/processed_splits/dedup_w_apdb_splits/apdb_min_freq_ddp_fc2_0.05_stable_-0.001_v2_cls.pkl.gz"
STRUCTURES_DIR="/work/y-tomiya/ntu/Dataset_thermoconductivity_pred/processed_splits/apdb_min_freq/structures"
OUTPUT_DIR="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data"

# Target minimum supercell vector length (Å) for --target_length
TARGET_LENGTH="10.0"

# Optional: set dataset name explicitly (if empty, inferred automatically from the pickle name)
DATASET_SLUG=""

# Property columns (--property_cols): in this dataset, log_klat etc. are stored as y_train_log_klat, etc.
PROPERTY_COLS=(is_stable)

# ==== Execution command ====
# Path to the Python script (adjust according to repository structure)
PY_SCRIPT="../1_build_supercells_from_pkl.py"

CMD=( python "${PY_SCRIPT}"
  --pickle_path "${PICKLE_PATH}"
  --structures_dir "${STRUCTURES_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --target_length "${TARGET_LENGTH}"
  --property_cols "${PROPERTY_COLS[@]}"
  # --skip_base_traj                 # Uncomment to skip output of base structures (_XP.traj)
)

# Append dataset_slug only if specified
if [[ -n "${DATASET_SLUG}" ]]; then
  CMD+=( --dataset_slug "${DATASET_SLUG}" )
fi

echo "[INFO] Command: ${CMD[*]}"
echo "[INFO] Output root (metadata/structures): ${OUTPUT_DIR}"

# Run as a single task with srun
srun --cpu-bind=cores "${CMD[@]}"

date
