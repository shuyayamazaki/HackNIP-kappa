# script to construct .pkl

import os, pickle
import numpy as np
from tqdm import tqdm
from ase.io import read
from run_benchmark import parse_task_list
from pathlib import Path

DATA_ROOT = Path(os.environ.get("BENCH_DATA_DIR", Path(__file__).resolve().parent / "benchmark_data")).resolve()
MLIP      = os.environ.get("BENCH_MLIP", "orb2")
MODEL     = os.environ.get("BENCH_MODEL", "modnet")
TASKS     = os.environ.get("BENCH_TASKS")

STRUCTURES_DIR = DATA_ROOT / "structures"
META_DIR       = DATA_ROOT / "metadata"
FEAT_DIR       = DATA_ROOT / f"feat_{MLIP}"
NPY_DIR        = FEAT_DIR / "npy"
RESULTS_DIR    = FEAT_DIR / f"results_{MODEL}"
HP_DIR         = RESULTS_DIR / "hp"
PARITY_DIR     = RESULTS_DIR / "parity"

for p in [STRUCTURES_DIR, META_DIR, FEAT_DIR, NPY_DIR, RESULTS_DIR, HP_DIR, PARITY_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def main():

    KEY = "XPS"
    key = KEY.lower()

    task_slugs = parse_task_list(TASKS)

    for task in task_slugs[:]:
        meta_path = f'{META_DIR}/{task}_meta.pkl'
        data_path = meta_path.replace("_meta.pkl", "_data.pkl")
        sc_atoms_path = f'{STRUCTURES_DIR}/{task}_all_{KEY}.traj'
        sc_atoms = read(sc_atoms_path, index="::")

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        data = meta.copy()
        data[f"{key}_atoms"] = sc_atoms

        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

        feat = data.copy()
        for l in tqdm(range(1,16), desc = f'constructing {task}'):
            npy_path = os.path.join(NPY_DIR, f'{task}_all_{KEY}_l{l}.npy')
            feat[f"{KEY}_l{l}"] = np.load(npy_path)

        feat_path = os.path.join(FEAT_DIR, f'{task}_{KEY}_{MLIP}.pkl')
        with open(feat_path, 'wb') as f:
            pickle.dump(feat, f)

if __name__ == "__main__":
    main()