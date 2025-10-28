#!/bin/bash
#SBATCH -p gpu_short                 # GPU不要ならCPU系パーティションに変更可（例: cpu_short）
##SBATCH --gres=gpu:1                # 必要ならコメント解除
#SBATCH --cpus-per-task=52           # このジョブはCPU中心（I/O主体）。過剰なら減らしてOK
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH -J SUPER_XPS                 # ジョブ名
#SBATCH --output=output_script/%x-%j.out
#SBATCH --error=output_script/%x-%j.err

set -euo pipefail

# スレッド制御（過剰並列の抑制）
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ログ出力ディレクトリ
mkdir -p output_script

# (任意) GPU情報・CUDAモジュール確認（GPU未使用でも harmless）
nvidia-smi || true
module avail cuda || true

date

# ==== 環境の有効化（あなたの環境に合わせて必要なら修正）====
# conda を使う場合
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hacknip

# ==== 入力/出力パスの設定 ====
PICKLE_PATH="/work/y-tomiya/ntu/Dataset_thermoconductivity_pred/processed_splits/dedup_w_apdb_splits/ood_split_dedup_w_min_freq.pkl.gz"
STRUCTURES_DIR="/work/y-tomiya/ntu/Dataset_thermoconductivity_pred/processed_splits/apdb_min_freq/structures"
OUTPUT_DIR="/work/y-tomiya/ntu/HackNIP_master/HackNIP/benchmark_data"

# supercell の最小ベクトル長(Å)の目標値（--target_length）
TARGET_LENGTH="10.0"

# 任意: データセット名を固定したい場合は設定（未設定ならpickle名から自動推定）
DATASET_SLUG=""

# 物性カラム（--property_cols）: 例では log_klat をターゲットとして使用
PROPERTY_COLS=(log_klat)

# ==== 実行コマンド ====
# Pythonスクリプトのパス（リポジトリ構成に合わせて調整）
PY_SCRIPT="src/matbench/build_supercells_from_pkl.py"

CMD=( python "${PY_SCRIPT}"
  --pickle_path "${PICKLE_PATH}"
  --structures_dir "${STRUCTURES_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --target_length "${TARGET_LENGTH}"
  --property_cols "${PROPERTY_COLS[@]}"
  # --skip_base_traj                 # ベース構造(_XP.traj)を出力しない場合はコメント解除
)

# dataset_slug を使う場合のみ付与
if [[ -n "${DATASET_SLUG}" ]]; then
  CMD+=( --dataset_slug "${DATASET_SLUG}" )
fi

echo "[INFO] Command: ${CMD[*]}"
echo "[INFO] Output root (metadata/structures): ${OUTPUT_DIR}"

# srun で1タスク実行
srun --cpu-bind=cores "${CMD[@]}"

date
