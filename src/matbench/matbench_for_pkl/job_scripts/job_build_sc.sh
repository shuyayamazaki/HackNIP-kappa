#!/bin/bash
#SBATCH -p gpu_short                 # Partition/queue (change to cpu_short if GPU is not required)
##SBATCH --gres=gpu:1                # Uncomment if GPU is needed
#SBATCH --cpus-per-task=52           # Number of CPU cores (I/O-heavy job; reduce if excessive)
#SBATCH -n 1                         # One task (no MPI)
#SBATCH -t 4:00:00                   # Time limit: 4 hours
#SBATCH -J SUPER_XPS                 # Job name
#SBATCH --output=output_script/%x-%j.out  # Standard output log
#SBATCH --error=output_script/%x-%j.err   # Standard error log

# Fail fast: exit on errors, undefined vars, or pipe failures
set -euo pipefail

# ---------------- Threading control ----------------
# Prevent oversubscription in linear-algebra backends
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1   # Flush Python output immediately for real-time logging

# ---------------- Log directory --------------------
mkdir -p output_script

# (Optional) Display GPU info or available CUDA modules
# Safe even when running on CPU nodes
nvidia-smi || true
module avail cuda || true

date  # Print start timestamp

# ================= Environment activation =================
# Activate your Python environment (adjust to your setup)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hacknip

# ================= Input / output paths ===================
PICKLE_PATH="/work/y-tomiya/ntu/Dataset_thermoconductivity_pred/processed_splits/dedup_w_apdb_splits/ood_split_dedup_w_min_freq.pkl.gz"
STRUCTURES_DIR="/work/y-tomiya/ntu/Dataset_thermoconductivity_pred/processed_splits/apdb_min_freq/structures"
OUTPUT_DIR="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data"

# Target minimal supercell vector length (Å)
TARGET_LENGTH="10.0"

# Optional: manually fix dataset slug name (auto-inferred from pickle if empty)
DATASET_SLUG=""

# Property column(s) to include (example: predicting log_klat)
PROPERTY_COLS=(log_klat)

# ================= Python execution command =================
# Adjust path to match your repository structure
PY_SCRIPT="src/matbench/build_supercells_from_pkl.py"

CMD=( python "${PY_SCRIPT}"
  --pickle_path "${PICKLE_PATH}"          # Input pickle file (.pkl or .pkl.gz)
  --structures_dir "${STRUCTURES_DIR}"    # Directory where supercell .traj files are saved
  --output_dir "${OUTPUT_DIR}"            # Root directory for generated data (metadata/structures)
  --target_length "${TARGET_LENGTH}"      # Target supercell vector length
  --property_cols "${PROPERTY_COLS[@]}"   # Target property column(s)
  # --skip_base_traj                      # Uncomment to skip saving base structure (_XP.traj)
)

# Append dataset slug argument only if specified
if [[ -n "${DATASET_SLUG}" ]]; then
  CMD+=( --dataset_slug "${DATASET_SLUG}" )
fi

echo "[INFO] Command: ${CMD[*]}"
echo "[INFO] Output root (metadata/structures): ${OUTPUT_DIR}"

# ================= Launch job =================
# Use srun for a single CPU task (core binding for consistency)
srun --cpu-bind=cores "${CMD[@]}"

date  # Print end timestamp
