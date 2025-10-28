# matbench_for_pkl Workflow

This folder contains a five-stage pipeline for turning MatBench-style split pickles into MODNet models trained on ORB2 supercell features. Each stage is implemented as a standalone Python entry point (`1_*.py` – `5_*.py`). 

> **Environment note:**  
> Set up the Python environment separately (see the project-level README).  
> The commands below assume the environment already provides ASE, ORB models, MODNet, TensorFlow, PyTorch, and Optuna.  
>  
> The base environment does not include them, so install the required packages manually:
> ```bash
> pip install modnet optuna
> ```

## 0. Prepare inputs
- **Split pickle:** e.g. `ood_split_dedup_w_min_freq.pkl.gz` containing `X_train`, `X_test`, `Y_train`, … blocks with `mp_ids`.
- **Base structures:** directory of `{mpid}.cif` files referenced by the pickle.
- **Output root:** a writable directory (default is `<pickle_dir>/benchmark_data`). The scripts expect the layout created during step 1, so keep all steps pointed at the same root.

## 1. Build supercells (`1_build_supercells_from_pkl.py`)
Generates ASE trajectory files for base and supercell structures plus metadata pickles per split and for the full dataset.

Key options (see `job_build_sc.sh`):
```bash
python 1_build_supercells_from_pkl.py \
  --pickle_path /path/to/ood_split_dedup_w_min_freq.pkl.gz \
  --structures_dir /path/to/structures_mp_cif \
  --output_dir /path/to/benchmark_data \
  --target_length 10.0 \
  --property_cols log_klat
```
- `--dataset_slug` overrides the inferred slug used in filenames.
- Pass `--skip_base_traj` if you only need supercells.
- Outputs land in `<output_dir>/{metadata,structures}` as `<slug>_{split}_XPS.traj` and `<slug>_{split}_meta.pkl`.

## 2. Featurize + assemble pickles (`2_featurize_construct_from_supercells.py`)
Reads the trajectories/metadata from step 1, runs ORB2 to build layer-wise features, and emits MODNet-friendly pickles and NumPy caches.

Example based on `job_featurize_construct_supercells.sh`:
```bash
python 2_featurize_construct_from_supercells.py \
  --slug ood_split_dedup_w_min_freq \
  --data-root /path/to/benchmark_data \
  --mlip orb2 \
  --model modnet \
  --device auto
```
- Uses `/feat_<mlip>/npy` to cache `XPS_l{1..15}.npy`.
- Re-run with `--overwrite` to recompute layers.
- Produces `<slug>_all_XPS_orb2.pkl` and, if per-split metadata exists, `<slug>_{split}_XPS_orb2.pkl`.

## 3. Train baseline MODNet (`3_train_modnet_from_supercells.py`)
Consumes the feature pickles and evaluates each available layer on a fixed train/test split. Optionally retrains the best layer on the full dataset and saves the model.

The SLURM helper `job_train_modnet_supercells.sh` exports several environment variables (`BENCH_DATA_DIR`, `BENCH_MLIP`, `BENCH_MODEL`, `BENCH_FEATURE_SLUGS`). You can either export them yourself or pass explicit flags:
```bash
export BENCH_DATA_DIR=/path/to/benchmark_data
export BENCH_MLIP=orb2
export BENCH_MODEL=modnet

python 3_train_modnet_from_supercells.py \
  --feature-slugs random_split_dedup_w_min_freq \
  --train-split train \
  --test-split test \
  --seed 42 \
  --fast \
  --train-final \
  --final-model-path /path/to/benchmark_data/feat_orb2/results_modnet/trained_models
```
- Use `--layer` to pin a single GNN layer or `--min-layer/--max-layer` to narrow the sweep.
- Set `--cuda-visible-devices` if you need to constrain GPUs.
- Metrics and logs mirror the outputs in `output_script/` from the SLURM run.

## 4. Hyperparameter optimisation (`4_opt_hp_modnet_from_supercells.py`)
Runs Optuna over the MODNet configuration for one or more layers, saving trial logs and best-model metadata. The template `job_opt_hp_supercells.sh` demonstrates a long-running GPU job.

Typical invocation:
```bash
export BENCH_DATA_DIR=/path/to/benchmark_data
export BENCH_MLIP=orb2
export BENCH_MODEL=modnet

python 4_opt_hp_modnet_from_supercells.py \
  --feature-slugs ood_split_dedup_w_min_freq \
  --train-split train \
  --test-split test \
  --seed 42 \
  --cuda-visible-devices 0 \
  --n-trials 60 \
  --cv-folds 5 \
  --sampler tpe \
  --early-stop-trials 20 \
  --trained-model-path /path/to/trained_models/layers_train2test/slug_train2test_XPS_orb2_l8.modnet \
  --layer 8
```
- `--trained-model-path` (or `--model-path`) auto-fills the slug/splits/key/layer and stores trial models next to the training run.
- `--layers`, `--min-layer`, and `--max-layer` let you scan multiple layers in one job.
- Outputs include Optuna study CSV/JSON files and serialized trial models under `<results_modnet>/training_dataset_<slug>_<timestamp>/`.

## 5. Transfer hyperparameters (`5_retrain_hp_model_transfer.py`)
Reuses the best Optuna configuration to retrain on additional dataset splits. `job_retrain_hp_transfer_supercells.sh` shows how to point at the `metadata.json` from step 4 and fan out to new slugs.

Example:
```bash
export BENCH_DATA_DIR=/path/to/benchmark_data
export BENCH_MLIP=orb2
export BENCH_MODEL=modnet

python 5_retrain_hp_model_transfer.py \
  --metadata-path /path/to/trial_models/.../metadata.json \
  --target-slugs random_split_dedup_w_min_freq,space_group_split_dedup_w_min_freq \
  --train-split train \
  --test-split test \
  --seed 42 \
  --cuda-visible-devices 0 \
  --save-models
```
- `--output-dir` overrides the default results folder (`.../hp_transfer_retrain/<metadata>_<timestamp>`).
- When `--save-models` is omitted the script still records metrics in `metrics.csv`.

## Running via SLURM
All job scripts under `HackNIP/job_*.sh` follow the same pattern:
- Load modules, activate the `hacknip` conda environment, and export thread-safety variables.
- Define dataset paths and pass them to one of the Python entry points above.
- Call the script through `srun` so SLURM manages CPU/GPU resources.

To adapt them, clone a job file, edit the path variables (`PICKLE_PATH`, `DATA_ROOT`, `TRAINED_MODEL_PATH`, etc.), then submit with `sbatch job_xxx.sh`.

## Outputs at a glance
- `benchmark_data/structures`: `{slug}_{split}_{XP|XPS}.traj` produced in step 1.
- `benchmark_data/metadata`: `{slug}_{split}_meta.pkl` plus `_data.pkl` after step 2.
- `benchmark_data/feat_orb2/npy`: cached NumPy features per layer from step 2.
- `benchmark_data/feat_orb2/results_modnet`: training logs, best models, Optuna studies, and retrained models from steps 3–5.

Keep the directory layout consistent between steps so downstream commands pick up the expected files.
