# script to retrieve datasets from matbench
# env: python 3.9, pip install matbench

import os
import pickle
from ase.io import write, read
from pymatgen.io.ase import AseAtomsAdaptor
from matbench import MatbenchBenchmark

# ─── User settings ─────────────────────────────────────────────────────────────
TASKS = [
    'matbench_dielectric', 'matbench_jdft2d', 'matbench_log_gvrh',
    'matbench_log_kvrh', 'matbench_mp_e_form', 'matbench_mp_gap',
    'matbench_perovskites', 'matbench_phonons'
]
STRUCTURES_DIR = '/home/sokim/ion_conductivity/feat/matbench/structures'
META_DIR = '/home/sokim/ion_conductivity/feat/matbench/metadata'
# ────────────────────────────────────────────────────────────────────────────────

os.makedirs(STRUCTURES_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

def main():
    mb = MatbenchBenchmark(subset=TASKS)
    for i, task in enumerate(mb.tasks, start=1):
        task.load()
        # name     = task.dataset_name
        df       = task.df
        ids      = list(df.index)
        targets  = df[task.metadata.target].to_numpy()

        # save metadata
        meta = {'ids': ids, 'targets': targets}
        meta_path = os.path.join(META_DIR, f't{i}_meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(meta, f)
        print(f"[INFO] saved metadata → {meta_path}")

        all_input_path = f"{STRUCTURES_DIR}/t{i}_all_XP.traj"
        all_atoms = [AseAtomsAdaptor.get_atoms(s) for s in task.df.structure]
        write(all_input_path, all_atoms)
        print(f"[INFO] Wrote {len(all_atoms)} structures to {all_input_path}")


if __name__ == "__main__":
    main()
