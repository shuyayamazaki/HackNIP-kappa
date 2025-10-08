# script to featurize using orb2

import os, sys, random, torch, pathlib
from tqdm import tqdm
import numpy as np
from ase.io import read
from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.calculator import ORBCalculator
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

def main():

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

    orbff = pretrained.orb_v2(device=device)
    model = orbff.model
    calc = ORBCalculator(orbff, device=device)

    task_slugs = parse_task_list(TASKS)

    for task in task_slugs[:]:

        sc_path = os.path.join(STRUCTURES_DIR, f'{task}_all_{KEY}.traj')
        sc_atoms = read(sc_path, index = "::")

        for l in range(1,16):
            feat = []
            f_path = os.path.join(NPY_DIR, f'{task}_all_{KEY}_l{l}.npy')

            feat = [get_graph_features(atoms, model, l, device) for atoms in tqdm(sc_atoms, desc=f"featurizing {task} l{l}")]
            np.save(f_path, feat)

if __name__ == '__main__':
    main()