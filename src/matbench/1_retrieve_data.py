# script to retrieve datasets from matbench

import os
import pickle
from ase.io import write, read
from pymatgen.io.ase import AseAtomsAdaptor
from matbench import MatbenchBenchmark
from run_benchmark import parse_task_list
from pathlib import Path

DATA_ROOT = Path(os.environ.get("BENCH_DATA_DIR", Path(__file__).resolve().parent / "benchmark_data")).resolve()
MLIP      = os.environ.get("BENCH_MLIP", "orb2")
MODEL     = os.environ.get("BENCH_MODEL", "results_modnet")
TASKS     = os.environ.get("BENCH_TASKS")

STRUCTURES_DIR = DATA_ROOT / "structures"
META_DIR       = DATA_ROOT / "metadata"
FEAT_DIR       = DATA_ROOT / f"feat_{MLIP}"
NPY_DIR        = FEAT_DIR / "npy"
RESULTS_DIR    = DATA_ROOT / MODEL
HP_DIR         = RESULTS_DIR / "hp"
PARITY_DIR     = RESULTS_DIR / "parity"

for p in [STRUCTURES_DIR, META_DIR, FEAT_DIR, NPY_DIR, RESULTS_DIR, HP_DIR, PARITY_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def main():

    task_slugs = parse_task_list(TASKS)
    mb = MatbenchBenchmark(subset=[f"matbench_{task}" for task in task_slugs[:]])

    for task in mb.tasks:
        name = task.dataset_name              # e.g., 'matbench_mp_gap'
        slug = name.removeprefix("matbench_") # 'mp_gap'
        task.load()
        df = task.df
        ids = list(df.index)
        targets = df[task.metadata.target].to_numpy()

        meta_path = META_DIR / f"{slug}_meta.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump({"ids": ids, "targets": targets, "dataset": name}, f)

        out_traj = STRUCTURES_DIR / f"{slug}_all_XP.traj"
        atoms = [AseAtomsAdaptor.get_atoms(s) for s in df.structure]
        write(out_traj, atoms)

if __name__ == "__main__":
    main()
