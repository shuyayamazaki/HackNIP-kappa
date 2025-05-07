# script to build supercells
# any env with ase

import os
import numpy as np
from ase.build import make_supercell
from ase.io import read
from ase.io.trajectory import Trajectory
from tqdm import tqdm

if __name__ == '__main__':

    checkpoint   = 1000

    for i in [5,6]:

        print(f"[INFO] Task {i} started")

        xp_path = f'/home/sokim/ion_conductivity/feat/matbench/structures/t{i}_all_XP.traj'
        output_path = xp_path.replace('_XP.traj', '_XPS.traj')
        all_atoms = read(xp_path, index='::')
        
        print(f"[INFO] _XP.traj read")
        
    # 1) figure out how many are already done
        if os.path.exists(output_path):
            try:
                done = len(read(output_path, index='::', format='traj'))
            except Exception:
                done = 0
            print(f"Resuming: found {done} supercells already in {output_path}")
        else:
            done = 0

        # 2) We'll buffer builds and flush in checkpoint‐sized batches.
        buffer = []

        # 3) iterate only over the remaining structures
        for idx, struct in enumerate(tqdm(all_atoms[done:], 
                                        total=len(all_atoms)-done,
                                        desc="Building supercells"), 
                                    start=done+1):
            a, b, c, _, _, _ = struct.cell.cellpar()
            scale = np.ceil(10/np.array([a,b,c])).astype(int)
            sc = make_supercell(struct, np.diag(scale)) if np.any(scale>1) else struct

            buffer.append(sc)

            # 4) checkpoint every `checkpoint` builds
            if idx % checkpoint == 0:
                # use Trajectory in append mode to get the header right
                with Trajectory(output_path, mode='a' if done>0 else 'w') as traj:
                    for atoms in buffer:
                        traj.write(atoms)
                buffer = []
                done += checkpoint
                tqdm.write(f"Checkpoint: wrote up to structure {done}")

        # 5) final flush of any remainder
        if buffer:
            with Trajectory(output_path, mode='a' if done>0 else 'w') as traj:
                for atoms in buffer:
                    traj.write(atoms)
            done += len(buffer)
            tqdm.write(f"Final flush: wrote {len(buffer)} more, total {done}")

        print("All supercells written.")