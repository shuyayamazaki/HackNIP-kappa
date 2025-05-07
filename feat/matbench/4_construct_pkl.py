# script to construct .pkl

import os, pickle
import numpy as np
from tqdm import tqdm
from ase.io import read

KEY = "XPS"
key = KEY.lower()
npy_dir = f'/home/sokim/ion_conductivity/feat/matbench/{key}2feat_orb2'

for i in [8]:
    meta_path = f'/home/sokim/ion_conductivity/feat/matbench/metadata/t{i}_meta.pkl'
    data_path = meta_path.replace("_meta.pkl", "_data.pkl")
    all_atoms_path = f'/home/sokim/ion_conductivity/matbench/structures/t{i}_all_{KEY}.traj'
    all_atoms = read(all_atoms_path, index="::")

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    data = meta.copy()
    data[f"{key}_atoms"] = all_atoms

    with open(data_path, 'wb') as f:
        pickle.dump(data, f)

    feat = data.copy()
    for l in tqdm(range(1,16), desc = f'constructing t{i}'):
        npy_path = os.path.join(npy_dir, f't{i}_all_{KEY}_l{l}.npy')
        feat[f"{KEY}_l{l}"] = np.load(npy_path)

    feat_path = os.path.join(npy_dir, f't{i}_{KEY}_orb2.pkl')
    with open(feat_path, 'wb') as f:
        pickle.dump(feat, f)