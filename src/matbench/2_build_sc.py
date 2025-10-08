# script to build supercells

import os
import numpy as np
from ase.build import make_supercell
from ase.io import read
from ase.io.trajectory import Trajectory
from tqdm import tqdm
from run_benchmark import parse_task_list
from pathlib import Path

DATA_ROOT = Path(os.environ.get("BENCH_DATA_DIR", Path(__file__).resolve().parent / "benchmark_data")).resolve()
MLIP      = os.environ.get("BENCH_MLIP", "orb2")
MODEL     = os.environ.get("BENCH_MODEL", "results_modnet")
TASKS     = os.environ.get("BENCH_TASKS")

# common dirs (suggestion: namespace by model)
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

    checkpoint   = 1000
    target_len  = 10.0  # minimum length of cell vector after supercelling, in angstroms: 10 (orb2 and eqV2) or 12 (mace)
    # structures_dir = '/home/sokim/Projects/HackNIP/src/matbench/structures'

    # for i in range(1,9):

    #     print(f"[INFO] Task {i} started")

    #     xp_path = f'/home/sokim/ion_conductivity/feat/matbench/structures/t{i}_all_XP.traj'
    #     output_path = xp_path.replace('_XP.traj', '_XPS.traj')

    task_slugs = parse_task_list(TASKS)

    for task in task_slugs[:]:
        input_path = f"{STRUCTURES_DIR}/{task}_all_XP.traj"
        output_path = f"{STRUCTURES_DIR}/{task}_all_XPS.traj"
        all_atoms = read(input_path, index='::')
        
        print(f"[INFO] _XP.traj read")
        
        # figure out how many are already done
        if os.path.exists(output_path):
            try:
                done = len(read(output_path, index='::', format='traj'))
            except Exception:
                done = 0
            print(f"Resuming: found {done} supercells already in {output_path}")
        else:
            done = 0

        # iterate only over the remaining structures
        buffer = []
        for idx, struct in enumerate(tqdm(all_atoms[done:], 
                                        total=len(all_atoms)-done,
                                        desc="Building supercells"), 
                                    start=done+1):
            a, b, c, _, _, _ = struct.cell.cellpar()
            scale = np.ceil(target_len/np.array([a,b,c])).astype(int)
            sc = make_supercell(struct, np.diag(scale)) if np.any(scale>1) else struct

            buffer.append(sc)

            # checkpoint every `checkpoint` builds
            if idx % checkpoint == 0:
                # use Trajectory in append mode to get the header right
                with Trajectory(output_path, mode='a' if done>0 else 'w') as traj:
                    for atoms in buffer:
                        traj.write(atoms)
                buffer = []
                done += checkpoint
                tqdm.write(f"Checkpoint: wrote up to structure {done}")

        # final flush of any remainder
        if buffer:
            with Trajectory(output_path, mode='a' if done>0 else 'w') as traj:
                for atoms in buffer:
                    traj.write(atoms)
            done += len(buffer)
            tqdm.write(f"Final flush: wrote {len(buffer)} more, total {done}")

        print("All supercells written.")

if __name__ == '__main__':
    main()