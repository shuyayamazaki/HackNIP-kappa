# Benchmark Pipeline

The orchestration script [`run_benchmark.py`](./run_benchmark.py) automates the multi-step workflow across environments.

## Overview

The benchmark workflow is organized into modular steps:

| Step | Script | Environment | Description |
|------|---------|--------------|--------------|
| 1 | `1_retrieve_data.py` | `matbench_env` | Retrieve and cache dataset(s) from MatBench. |
| 2 | `2_build_sc.py` | `hacknip_env` | Build supercells or derived structures for featurization. |
| 3 | `3_featurize_<mlip>.py` | `hacknip_env` | Compute features using the specified MLIP (supported: ORB2). |
| 4 | `4_construct_pkl.py` | `hacknip_env` | Assemble MODData `.pkl` files for training. |
| 5 | `5_train_<model>.py` | `hacknip_env` | Train ML model (supported: MODNet). |
| 6 | `6_opt_hp_<model>.py` | `hacknip_env` | Hyperparameter optimization via Optuna. |
| 7 | `7_get_parity_data.py` | `hacknip_env` | Generate and save parity data for visualization. |

---

## Environment Setup

You’ll need two environments:  
- **`matbench_env`** for MatBench dataset handling  
- **`hacknip_env`** for featurization using MLIPs and ML model training

### Create environments
```bash
conda create -n matbench_env python=3.9
conda activate matbench_env
pip install matbench
```

```bash
conda create -n hacknip_env python=3.11
conda activate hacknip_env
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install "pynanoflann@git+https://github.com/dwastberg/pynanoflann#egg=af434039ae14bedcbb838a7808924d6689274168"
pip install orb-models==0.4.2 ase optuna modnet
```

## Running the Pipeline

### Basic usage
Runs all steps for all tasks using default settings:
* MLIP: orb2
* MODEL: modnet
```bash
python run_benchmark.py
```

Data and logs are automatically placed under
```bash
benchmark_data/
└── logs/
```

### Example: run selected tasks and steps
```bash
python run_benchmark.py --tasks mp_gap,phonons --steps 1-4
```

### Example: specify environments explicitly
```bash
python run_benchmark.py \
  --py-matbench ~/miniconda3/envs/matbench_env/bin/python \
  --py-mlip ~/miniconda3/envs/hacknip_env/bin/python \
  --py-mlmodel ~/miniconda3/envs/hacknip_env/bin/python
  --tasks mp_gap,phonons --steps 1-4
```

### Example: dry run
Lists planned commands without execution:
```bash
python run_benchmark.py --dry-run
```

## Valid Task Slugs
Use one or more of the following:
```bash
dielectric, jdft2d, log_gvrh, log_kvrh,
mp_e_form, mp_gap, perovskites, phonons
```

Example:
```bash
--tasks mp_gap,perovskites
```