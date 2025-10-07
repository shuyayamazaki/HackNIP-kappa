
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

from fairchem.core import OCPCalculator
from fairchem.core.datasets import data_list_collater

from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error
from mace.data import AtomicData

from xgboost import XGBClassifier, XGBRegressor
from mace.calculators import MACECalculator
from mace.calculators import mace_mp
from mace.data import AtomicData
from mace.data.utils import Configuration
from mace.tools import AtomicNumberTable
from torch_geometric.data import Batch
from types import SimpleNamespace
from mace.data.utils import config_from_atoms
from mace.calculators.utils import load_checkpoint
from mace.modules.models import MACE
from mace.modules.models import ScaleShiftMACE

def get_graph_features(atoms, calc, layer_until, device):

    features = calc.get_descriptors(atoms, invariants_only=True, num_layers=layer_until)

    return features.mean(axis=0)  

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

    
def main(args):
    # ----------------------------
    # 1. Basic configuration
    # ----------------------------
    device = args.device
    data_path = args.data_path
    split_type = args.split_type
    task_type = args.task_type
    ml_type = args.ml_type
    # -------------------------------
    # 2. Prepare data and MLIP model
    # -------------------------------
    # Open a file to write outputs
    output_file = f"output/output_{data_path.split('/')[-1].split('.')[0]}_mace_{ml_type}.txt"
    with open(output_file, "w") as out:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            
            # checkpoint_path = "path/to/the/downloaded/checkpoint"
            # model = torch.load(checkpoint_path, map_location=device)
            # model = model.to(device)
            # model.eval()
            # num_layers = model.num_interactions.item()
            # calc = MACECalculator(model_paths = checkpoint_path, device=device) 

            calc = mace_mp(model="medium", device=device)
            num_layers = calc.models[0].num_interactions 

        for layer in range(1, num_layers + 1):
            out.write(f"Layer: {layer}\n")
            for relax in ['X']: #, 'XR'
                out.write(f"Relaxation: {relax}\n")
                X = np.array([get_graph_features(decode(json.dumps(atoms_dict)), calc, layer, device) for atoms_dict in data[relax]])

                for prop in data['Y'].keys():
                    out.write(f"Property: {prop}\n")
                    Y = np.array(data['Y'][prop])

                    x_save_path = f"output/X_mace_{prop}_layer{layer}_{relax}.npy"
                    np.save(x_save_path, X)
                
                    # Optionally, print or log where they're saved
                    # out.write(f"Saved X to {x_save_path}\n")
                    # out.write(f"Saved Y to {y_save_path}\n")

                    if task_type == 'classification':
                        result_rocauc = []
                        result_acc = []
                    elif task_type == 'regression':
                        result_mae = []
                    for seed in range(10):
                        if split_type == 'random':
                            train_X, train_Y, valid_X, valid_Y, test_X, test_Y = random_split(X, Y, random_seed=seed)
                        elif split_type == 'scaffold':
                            train_X, train_Y, valid_X, valid_Y, test_X, test_Y = scaffold_split(X, Y, data['SMILES'], random_seed=seed)
                        
                        scaler = StandardScaler()
                        train_X = scaler.fit_transform(train_X)
                        test_X = scaler.transform(test_X)

                        if ml_type == 'mlp':
                            ml = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=200) if task_type == 'classification' else MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=200)
                            ml.fit(train_X, train_Y)
                        elif ml_type == 'xgb':
                            if task_type == 'classification':
                                ml = XGBClassifier(
                                    n_estimators=100,        # number of trees
                                    max_depth=6,             # depth of each tree
                                    learning_rate=0.1,       # step size shrinkage
                                    subsample=0.8,           # fraction of samples used per tree
                                    colsample_bytree=0.8,    # fraction of features used per tree
                                    random_state=42,
                                    use_label_encoder=False, # disable warning
                                    eval_metric='logloss'    # required for newer xgboost versions
                                )
                            else:
                                ml = XGBRegressor(
                                    n_estimators=100,
                                    max_depth=6,
                                    learning_rate=0.1,
                                    subsample=0.8,
                                    colsample_bytree=0.8,
                                    random_state=42
                                )
                            ml.fit(train_X, train_Y)

                        elif ml_type == 'rf':
                            if task_type == 'classification':
                                ml = RandomForestClassifier(
                                    n_estimators=100,      # number of trees
                                    max_depth=6,           # maximum depth of each tree (adjust as needed)
                                    random_state=42
                                )
                            else:
                                ml = RandomForestRegressor(
                                    n_estimators=100,
                                    max_depth=6,
                                    random_state=42
                                )
                            ml.fit(train_X, train_Y)

                        # Calculate metrics
                        if task_type == 'classification':
                            preds = ml.predict_proba(test_X)[:, 1]
                            roc_auc = roc_auc_score(test_Y, preds)
                            preds = ml.predict(test_X)
                            acc = accuracy_score(test_Y, preds)
                            out.write(f"Seed {seed}: {prop} | ROC AUC: {roc_auc} | Accuracy: {acc}\n")
                            result_rocauc.append(roc_auc)
                            result_acc.append(acc)
                        
                        elif task_type == 'regression':
                            preds = ml.predict(test_X)
                            mae = mean_absolute_error(test_Y, preds)
                            out.write(f"Seed {seed}: {prop} | MAE: {mae}\n")
                            result_mae.append(mae)
                    
                    # calculate mean and std
                    if task_type == 'classification':
                        out.write(f"Property: {prop} | ROC AUC: {np.mean(result_rocauc)} ± {np.std(result_rocauc)} | Accuracy: {np.mean(result_acc)} ± {np.std(result_acc)}\n")
                    elif task_type == 'regression':
                        out.write(f"Property: {prop} | MAE: {np.mean(result_mae)} ± {np.std(result_mae)}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on (cpu or cuda:0)')
    parser.add_argument('--data_path', type=str, default='/home/lucky/Projects/ion_conductivity/feat/preprocessed_data/BACE_dataset_relaxed.pkl', help='Path to the preprocessed data file')
    parser.add_argument('--task_type', type=str, default='classification', help='Type of task (classification or regression)')
    parser.add_argument('--split_type', type=str, default='scaffold', help='Type of data split (random or scaffold)')
    parser.add_argument('--ml_type', type=str, default='mlp', help='Type of ML model (mlp, xgb, or rf)')
    args = parser.parse_args()
    main(args)