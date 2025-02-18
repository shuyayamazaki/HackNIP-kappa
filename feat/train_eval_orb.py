
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import json

import torch
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.io.jsonio import encode, decode
from ase.optimize import BFGS

from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.calculator import ORBCalculator

from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error, r2_score


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


def random_split(dataset_X, dataset_Y, random_seed, valid_size=0, test_size=0.1):
    import random
    random.seed(random_seed)

    indices = list(range(len(dataset_X)))
    random.shuffle(indices)
    valid_cutoff = int(len(dataset_X) * valid_size)
    test_cutoff = int(len(dataset_X) * test_size)
    
    valid_indices = indices[:valid_cutoff]
    test_indices = indices[valid_cutoff:valid_cutoff + test_cutoff]
    train_indices = indices[valid_cutoff + test_cutoff:]
    
    train_X, train_Y = dataset_X[train_indices], dataset_Y[train_indices]
    valid_X, valid_Y = dataset_X[valid_indices], dataset_Y[valid_indices]
    test_X, test_Y = dataset_X[test_indices], dataset_Y[test_indices]

    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y

def scaffold_split(dataset_X, dataset_Y, smiles_list, random_seed, valid_size=0, test_size=0.1):
    if smiles_list is None:
        raise ValueError("smiles_list is required for scaffold splitting.")
    import random
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    def generate_scaffold(mol, include_chirality=True):
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=include_chirality
        )

    scaffolds = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        scaf = generate_scaffold(mol) if mol is not None else smi
        scaffolds.append(scaf)
    
    scaffold_to_indices = {}
    for idx, scaf in enumerate(scaffolds):
        scaffold_to_indices.setdefault(scaf, []).append(idx)
    
    # Sort by descending frequency
    scaffolds_sorted = sorted(scaffold_to_indices.keys(), key=lambda x: len(scaffold_to_indices[x]), reverse=True)
    
    total_size = len(dataset_X)
    valid_cutoff = int(np.floor(valid_size * total_size))
    test_cutoff = int(np.floor(test_size * total_size))
    
    train_indices, valid_indices, test_indices = [], [], []
    for scaf in scaffolds_sorted:
        idxs = scaffold_to_indices[scaf]
        if len(valid_indices) + len(idxs) <= valid_cutoff:
            valid_indices.extend(idxs)
        elif len(test_indices) + len(idxs) <= test_cutoff:
            test_indices.extend(idxs)
        else:
            train_indices.extend(idxs)
    
    train_X, train_Y = dataset_X[train_indices], dataset_Y[train_indices]
    valid_X, valid_Y = dataset_X[valid_indices], dataset_Y[valid_indices]
    test_X, test_Y = dataset_X[test_indices], dataset_Y[test_indices]

    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y


class LogTransformer:
    """Applies log transformation conditionally based on log_transform flag."""
    
    def __init__(self, log_transform=True):
        self.log_transform = log_transform  # 로그 변환 여부 설정

    def fit(self, y):
        if not self.log_transform:
            return self  # 변환 없이 바로 반환

        self.min_value = np.min(y)  
        if self.min_value <= 0:
            self.offset = abs(self.min_value) + 1  # 음수를 방지하기 위한 오프셋
        else:
            self.offset = 0
        return self

    def transform(self, y):
        if not self.log_transform:
            return y  # 변환 없이 원본 데이터 반환
        return np.log(y + self.offset)

    def inverse_transform(self, y_log):
        if not self.log_transform:
            return y_log  # 변환 없이 원본 데이터 반환
        return np.exp(y_log) - self.offset


def main(args):
    # ----------------------------
    # 1. Basic configuration
    # ----------------------------
    device = args.device
    data_path = args.data_path
    split_type = args.split_type
    task_type = args.task_type
    log_transform = args.log_transform
    # -------------------------------
    # 2. Prepare data and MLIP model
    # -------------------------------
    # Open a file to write outputs
    output_file = f"output/output_{data_path.split('/')[-1].split('.')[0]}_decode.txt"
    with open(output_file, "w") as out:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        orbff = pretrained.orb_v2(device=device)
        orbff.model.eval()
        for layer in range(1, 16):
            out.write(f"Layer: {layer}\n")
            for relax in ['X', 'XR']:
                out.write(f"Relaxation: {relax}\n")
                X = np.array([get_graph_features(decode(json.dumps(atoms_dict)), orbff.model, layer, device) for atoms_dict in data[relax]])
                for prop in data['Y'].keys():
                    out.write(f"Property: {prop}\n")
                    Y = np.array(data['Y'][prop])
                    if task_type == 'classification':
                        result_rocauc = []
                        result_acc = []
                    elif task_type == 'regression':
                        result_mae = []
                        result_r2 = []
                    for seed in range(10):
                        if split_type == 'random':
                            train_X, train_Y, valid_X, valid_Y, test_X, test_Y = random_split(X, Y, random_seed=seed)
                        elif split_type == 'scaffold':
                            train_X, train_Y, valid_X, valid_Y, test_X, test_Y = scaffold_split(X, Y, data['SMILES'], random_seed=seed)
                        
                        scaler = StandardScaler()
                        train_X = scaler.fit_transform(train_X)
                        test_X = scaler.transform(test_X)

                        # Y값 로그 변환 적용 여부에 따라 처리
                        if task_type == 'regression':
                            log_scaler = LogTransformer(log_transform=log_transform)
                            train_Y = log_scaler.fit(train_Y).transform(train_Y)

                        mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=200) if task_type == 'classification' else MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=200)

                        mlp.fit(train_X, train_Y)

                        # Calculate metrics
                        if task_type == 'classification':
                            preds = mlp.predict_proba(test_X)[:, 1]
                            roc_auc = roc_auc_score(test_Y, preds)
                            preds = mlp.predict(test_X)
                            acc = accuracy_score(test_Y, preds)
                            out.write(f"Seed {seed}: {prop} | ROC AUC: {roc_auc} | Accuracy: {acc}\n")
                            result_rocauc.append(roc_auc)
                            result_acc.append(acc)
                        
                        elif task_type == 'regression':
                            preds = mlp.predict(test_X)
                            preds = log_scaler.inverse_transform(preds)
                            mae = mean_absolute_error(test_Y, preds)
                            r2 = r2_score(test_Y, preds)
                            out.write(f"Seed {seed}: {prop} | MAE: {mae} | R2: {r2}\n")
                            result_mae.append(mae)
                            result_r2.append(r2)
                    
                    # calculate mean and std
                    if task_type == 'classification':
                        out.write(f"Property: {prop} | ROC AUC: {np.mean(result_rocauc)} ± {np.std(result_rocauc)} | Accuracy: {np.mean(result_acc)} ± {np.std(result_acc)}\n")
                    elif task_type == 'regression':
                        out.write(f"Property: {prop} | MAE: {np.mean(result_mae)} ± {np.std(result_mae)} | R2: {np.mean(result_r2)} ± {np.std(result_r2)}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on (cpu or cuda:0)')
    parser.add_argument('--data_path', type=str, default='/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/MPContribs_armorphous_diffusivity_relaxed.pkl', help='Path to the preprocessed data file')
    parser.add_argument('--task_type', type=str, default='regression', help='Type of task (classification or regression)')
    parser.add_argument('--split_type', type=str, default='random', help='Type of data split (random or scaffold)')
    parser.add_argument('--log_transform', type=bool, default=True, help='Whether to apply log transformation to the target values')
    args = parser.parse_args()
    main(args)