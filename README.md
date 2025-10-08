# Leveraging Neural Network Interatomic Potentials for a Foundation Model of Chemistry

This repository contains the code for the paper **"Leveraging neural network interatomic potentials for a foundation model of chemistry"** by Kim et al. (2025).

📄 **Paper**: [arXiv:2506.18497](https://arxiv.org/abs/2506.18497)

## Overview

This codebase implements a comprehensive framework for leveraging neural network interatomic potentials (NNIPs) as foundation models for chemistry. The repository provides tools for:

- **Structure relaxation**: Using pretrained NNIPs (ORB, EquiformerV2, MACE) to relax molecular and crystal structures
- **Feature extraction**: Extracting graph-level features from relaxed structures
- **Property prediction**: Training downstream ML models for various chemical properties
- **Benchmark evaluation**: Standardized evaluation on multiple datasets including MoleculeNet, Materials Project, and Matbench

### Key Features

- Support for multiple datasets:
  - **MoleculeNet**: BACE, BBBP, ClinTox, ESOL, FreeSolv, HIV, Lipophilicity, SIDER, Tox21
  - **Materials Project**: Band gap prediction, trajectory analysis (MPtrj)
  - **Amorphous materials**: Diffusivity prediction
  - **Matbench**: 8 standardized materials science benchmarks
- Three NNIP backends:
  - ORB (Orbital Materials)
  - EquiformerV2
  - MACE (Machine Learning of Atomic Cluster Expansion)
- Automated preprocessing and evaluation pipelines
- Parallel processing support for large-scale experiments

## Repository Structure

```
HackNIP/
├── src/
│   ├── preprocessing_relaxation_*.py  # Data preprocessing scripts
│   ├── train_eval_*.py                # Model training and evaluation
│   ├── run_preprocessing.py           # Batch preprocessing runner
│   ├── run_evaluation.py              # Batch evaluation runner
│   ├── utils.py                       # Utility functions
│   ├── matbench/                      # Matbench benchmark scripts
│   │   ├── 1_retrieve_data.py         # Download Matbench datasets
│   │   ├── 2_build_sc.py              # Build supercells
│   │   ├── 3_featurize_orb2.py        # Extract ORB features
│   │   ├── 4_construct_pkl.py         # Create pickle files
│   │   ├── 5_train_modnet.py          # Train MODNet models
│   │   ├── 6_opt_hp_modnet.py         # Hyperparameter optimization
│   │   └── 7_get_parity_data.py       # Generate parity plots
│   ├── pnas_ce.ipynb                  # PNAS analysis notebook
│   └── visualization.ipynb            # Visualization tools
└── README.md

```

-----------
## Installation
```
conda create -n hacknip python=3.9
conda activate hacknip
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
python -m pip install lightning
pip install hydra-core joblib wandb matplotlib scikit-learn python-dotenv jarvis-tools pymatgen ase rdkit tqdm transformers datasets diffusers fairchem-core
pip install orb-models
pip install "pynanoflann@git+https://github.com/dwastberg/pynanoflann#egg=af434039ae14bedcbb838a7808924d6689274168"
pip3 install auto-sklearn
!pip install git+https://github.com/DavidWalz/diversipy.git
pip install matbench-discovery
pip install matbench
pip install openpyxl
```

```
conda create -n matbench python=3.11
conda activate matbench
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install "pynanoflann@git+https://github.com/dwastberg/pynanoflann#egg=af434039ae14bedcbb838a7808924d6689274168"
pip install matbench-discovery typer
```

## Usage

### 1. Data Preprocessing

The preprocessing scripts convert various data sources into a unified `.pkl` format containing both unrelaxed and NNIP-relaxed structures.

#### Individual Dataset Preprocessing

```bash
# Band gap prediction (Materials Project)
python src/preprocessing_relaxation_bandgap.py \
    --device cuda:0 \
    --data_path MP \
    --property_cols '["Eg(eV)"]'

# Molecular properties (MoleculeNet)
python src/preprocessing_relaxation_moleculenet.py \
    --device cuda:0 \
    --data_path /path/to/ESOL_dataset.csv \
    --property_cols '["measured log solubility in mols per litre"]'

# Diffusivity prediction (amorphous materials)
python src/preprocessing_relaxation_diffusivity.py \
    --device cuda:0 \
    --data_path /path/to/diffusivity_data.parquet \
    --property_cols '["diffusivity"]'

# Materials Project trajectories
python src/preprocessing_relaxation_mptrj.py \
    --device cuda:0 \
    --data_path /path/to/mptrj_data \
    --property_cols '["energy_per_atom"]'
```

#### Batch Preprocessing

For processing multiple datasets in parallel:

```bash
python src/run_preprocessing.py
```

Edit the script to customize `DEVICES`, `DATA_FILES`, and `PROPERTY_COLS` lists.

### 2. Model Training and Evaluation

Train downstream ML models using features extracted from NNIP-relaxed structures.

#### Using ORB Features

```bash
python src/train_eval_orb.py \
    --device cuda:0 \
    --data_path preprocessed_data/ESOL_dataset_relaxed.pkl \
    --task_type regression \
    --split_type random
```

#### Using EquiformerV2 Features

```bash
python src/train_eval_eqV2.py \
    --device cuda:0 \
    --data_path preprocessed_data/BACE_dataset_relaxed.pkl \
    --task_type classification \
    --split_type scaffold
```

#### Using MACE Features

```bash
python src/train_eval_mace.py \
    --device cuda:0 \
    --data_path preprocessed_data/bandgap_relaxed.pkl \
    --task_type regression \
    --split_type random
```

#### Batch Evaluation

```bash
python src/run_evaluation.py
```

### 3. Matbench Benchmark

Follow the sequential pipeline for Matbench evaluation:

```bash
cd src/matbench

# Step 1: Retrieve datasets from Matbench
python 1_retrieve_data.py

# Step 2: Build supercells for materials
python 2_build_sc.py

# Step 3: Extract ORB features from structures
python 3_featurize_orb2.py

# Step 4: Construct pickle files for training
python 4_construct_pkl.py

# Step 5: Train MODNet models
python 5_train_modnet.py

# Step 6: Hyperparameter optimization
python 6_opt_hp_modnet.py

# Step 7: Generate parity plots and analysis
python 7_get_parity_data.py
```


## Supported Tasks

### Regression Tasks
- **ESOL**: Water solubility prediction
- **FreeSolv**: Solvation free energy
- **Lipophilicity**: Octanol/water partition coefficient
- **Band gap**: Electronic band gap of materials
- **Diffusivity**: Ion diffusion in amorphous materials
- **Matbench regression**: Formation energy, elastic moduli, phonon properties

### Classification Tasks
- **BACE**: β-secretase inhibition
- **BBBP**: Blood-brain barrier permeability
- **ClinTox**: Clinical trial toxicity
- **HIV**: HIV inhibition
- **Tox21**: Nuclear receptor signaling toxicity
- **SIDER**: Side effect prediction (27 targets)

## Output Format

All preprocessing scripts generate `.pkl` files containing:
- `X`: List of unrelaxed ASE atoms (JSON-encoded)
- `XR`: List of NNIP-relaxed ASE atoms (JSON-encoded)
- `Y`: Dictionary of property values keyed by property name

Training scripts output results to `results/` directory with performance metrics (MAE, R², ROC-AUC, Accuracy).

## Data Resources

### Amorphous Diffusivity Dataset
- [Materials Project Contribs page](https://contribs.materialsproject.org/projects/amorphous_diffusivity)
- [CSV data download](https://drive.google.com/file/d/1KZn4WD3NLvlD1lr4PGvCBqZ80Syk5Vzr/view?usp=sharing)
- [CSV with crystalline structures](https://drive.google.com/file/d/1-2YsXG4ezZaHTZsnm3l2swgVw0LO7kDI/view?usp=sharing)
- Reference: [The ab initio amorphous materials database: Empowering machine learning to decode diffusivity](https://ar5iv.labs.arxiv.org/html/2402.00177)

### Materials Project
- [Query and download contributed data](https://docs.materialsproject.org/downloading-data/query-and-download-contributed-data)
- Band gap data from Materials Project database
- MPtrj trajectory datasets

### MoleculeNet
Standard molecular property prediction benchmarks available through:
- [MoleculeNet website](http://moleculenet.org/)
- [DeepChem library](https://deepchem.io/)

### Matbench
- [Matbench package](https://github.com/materialsproject/matbench)
- 8 standardized materials property prediction tasks
- Automated data loading via `matbench` Python package

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kim2025leveraging,
  title={Leveraging neural network interatomic potentials for a foundation model of chemistry},
  author={Kim et al.},
  year={2025},
  eprint={2506.18497},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

**Paper**: https://arxiv.org/abs/2506.18497

## Key Dependencies

- **Neural Network Potentials**:
  - `orb-models`: ORB pretrained foundation model
  - `fairchem-core`: EquiformerV2 implementation
  - MACE models

- **Structure manipulation**:
  - `ase`: Atomic Simulation Environment
  - `pymatgen`: Materials analysis
  - `rdkit`: Molecular informatics

- **Machine Learning**:
  - `torch`, `torch_geometric`: Deep learning
  - `sklearn`: Traditional ML models
  - `lightning`: Training framework

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the corresponding author of the paper.
