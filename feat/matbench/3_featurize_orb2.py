# script to featurize using orb2
# env: python 3.9, pip install orb-models, not compatible with matbench

import os, sys, random, torch, pathlib
from tqdm import tqdm
import numpy as np
from ase.io import read
from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.calculator import ORBCalculator

def get_graph_features(atoms, model, layer_until, device):
    """
    Get the graph features from the model until the specified layer.
    
    Args:
    - atoms (dict): Dictionary containing the atomic coordinates and atomic numbers.
    - model (OrbFrozenMLP): The model to extract the features from.
    - layer_until (int): The layer until which the features are extracted. (1-15)
    - device (str): The device to run the model on.
    
    Returns:
    - np.ndarray: The graph features.
    """
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms, device=device)
    graph = model.featurize_edges(graph)
    graph = model.featurize_nodes(graph)
    graph = model._encoder(graph)

    for gnn in model.gnn_stacks[:layer_until]:
        graph = gnn(graph)
    
    graph = model._decoder(graph)
    return graph.node_features['feat'].mean(dim=0).cpu().numpy()

print(f"Command-line arguments: {' '.join(sys.argv)}")

# log the full source into the log file
src_path = pathlib.Path(__file__).resolve()
print(f"--- Begin source: {src_path.name} ---")
print(src_path.read_text())
print(f"--- End source: {src_path.name} ---")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
KEY = "XPS"
key = KEY.lower()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # disables autotuning, useful for reproducibility

data_dir = '/home/sokim/ion_conductivity/feat/matbench/structures'
results_dir = f'/home/sokim/ion_conductivity/feat/matbench/{key}2feat_orb2'
os.makedirs(results_dir, exist_ok=True)

orbff = pretrained.orb_v2(device=device)
model = orbff.model
calc = ORBCalculator(orbff, device=device)

for i in [6]: # range(1,9)

    xps_path = os.path.join(data_dir, f't{i}_all_{KEY}.traj')
    xps_all_atoms = read(xps_path, index = "::")

    for l in range(1,16):
        feat = []
        f_path = os.path.join(results_dir, f't{i}_all_{KEY}_l{l}.npy')

        feat = [get_graph_features(atoms, model, l, device) for atoms in tqdm(xps_all_atoms, desc=f"featuring t{i} l{l}")]
        np.save(f_path, feat)