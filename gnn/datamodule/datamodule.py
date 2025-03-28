import pandas as pd

import torch
from torch.utils.data import Dataset

from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import BFGS

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator


def smiles_to_ase(smiles, conformation=False):
    """
    Convert a SMILES string into a 3D ASE Atoms object using RDKit.
    """
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES string: {smiles}")
    
    # Add hydrogen atoms
    mol = Chem.AddHs(mol)
    
    if conformation:
        # Generate a 3D conformation
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
    else:
        # Generate a 2D conformation
        AllChem.Compute2DCoords(mol)
    
    # Extract atomic positions from the RDKit conformer
    conformer = mol.GetConformer()
    positions = []
    symbols = []
    for atom in mol.GetAtoms():
        pos = conformer.GetAtomPosition(atom.GetIdx())
        positions.append((pos.x, pos.y, pos.z))
        symbols.append(atom.GetSymbol())
    
    # Create an ASE Atoms object
    ase_atoms = Atoms(symbols=symbols, positions=positions)
    return ase_atoms


def atoms_relaxation(atoms, calculator):
    atoms.set_calculator(calculator)
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.01)
    return atoms


class BaseMoleculeDataset(Dataset):
    def __init__(self, csv_path, smiles_col="SMILES", property_cols=None, relaxation=False, task_type="regression"):
        """
        Args:
            csv_path (str): Path to the CSV file containing SMILES and properties.
            smiles_col (str): Name of the CSV column containing SMILES. Default "SMILES".
            property_cols (list): Optional list of columns to treat as property data.
            transform (callable): Optional transform to apply to the (atoms, properties) pair.
        """
        self.data = pd.read_csv(csv_path)
        self.smiles_col = smiles_col
        self.property_cols = property_cols if property_cols is not None else []

        # Get ASE atoms as X and properties as Y
        if relaxation:
            device="cuda" # or device="cuda"
            orbff = pretrained.orb_v2(device=device) # or choose another model using ORB_PRETRAINED_MODELS[model_name]()
            self.calc = ORBCalculator(orbff, device=device)
            self.X = []
            self.smiles_list = []
            self.Y = []
            for i, row in self.data.iterrows():
                try:  # To remove invalid SMILES
                    atoms = atoms_relaxation(smiles_to_ase(row[smiles_col], relaxation), self.calc)
                    self.X.append(atoms)
                    self.smiles_list.append(row[smiles_col])
                    self.Y.append(float(row[property_cols]) if task_type == "regression" else int(row[property_cols]))
                except:
                    pass
        else:
            self.X = []
            self.smiles_list = []
            self.Y = []
            for i, row in self.data.iterrows():
                try:  # To remove invalid SMILES
                    self.X.append(smiles_to_ase(row[smiles_col]))
                    self.smiles_list.append(row[smiles_col])
                    self.Y.append(float(row[property_cols]) if task_type == "regression" else int(row[property_cols]))
                except:
                    pass

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


from orb_models.forcefield import atomic_system
from orb_models.forcefield.base import batch_graphs

def collate_fn(samples):
    """
    :param samples: list of (ASE Atoms, properties) pairs.
    :param device:  device to create the graphs on (cpu or cuda).
    :return: (batched_graph, properties_list)
    """
    # Unpack the samples
    atoms_list, props_list = zip(*samples)  # each is a tuple
    
    # Convert each ASE atoms object to an ORB-model-compatible graph
    graphs = [atomic_system.ase_atoms_to_atom_graphs(atoms) for atoms in atoms_list]
    
    # Batch the list of graphs
    batched_graph = batch_graphs(graphs)

    # Convert properties to a tensor (handle both int and float types)
    # if isinstance(props_list[0], int):
    #     props_tensor = torch.tensor(props_list, dtype=torch.long)
    # elif isinstance(props_list[0], float):
    props_tensor = torch.tensor(props_list, dtype=torch.float)
    # else:
    #     props_tensor = torch.tensor(props_list)  # Default case (fallback)

    return batched_graph, props_tensor


class MoleculeDatasetSplitter:
    def __init__(self, dataset, smiles_list=None, random_seed=42, split_type=None):
        """
        :param dataset: The dataset to be split.
        :param smiles_list: Parallel list of SMILES for each dataset entry (only needed if scaffold splitting).
        :param random_seed: Random seed for splits.
        """
        self.dataset = dataset
        self.smiles_list = smiles_list
        self.random_seed = random_seed
        self.splits = {}  # to store "train", "valid", "test" indices or subsets
        if split_type is not None:
            if split_type == "random":
                self.random_split()
            elif split_type == "scaffold":
                self.scaffold_split()
            else:
                raise ValueError(f"Invalid split_type: {split_type}. Use 'random' or 'scaffold'.")

    def random_split(self, valid_size=0.1, test_size=0.1):
        import random
        random.seed(self.random_seed)

        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        valid_cutoff = int(len(self.dataset) * valid_size)
        test_cutoff = int(len(self.dataset) * test_size)
        
        valid_indices = indices[:valid_cutoff]
        test_indices = indices[valid_cutoff:valid_cutoff + test_cutoff]
        train_indices = indices[valid_cutoff + test_cutoff:]
        
        self.splits["train"] = train_indices
        self.splits["valid"] = valid_indices
        self.splits["test"] = test_indices
    
    def scaffold_split(self, valid_size=0.1, test_size=0.1):
        if self.smiles_list is None:
            raise ValueError("smiles_list is required for scaffold splitting.")
        import random
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        def generate_scaffold(mol, include_chirality=True):
            return MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=include_chirality
            )

        scaffolds = []
        for smi in self.smiles_list:
            mol = Chem.MolFromSmiles(smi)
            scaf = generate_scaffold(mol) if mol is not None else smi
            scaffolds.append(scaf)
        
        scaffold_to_indices = {}
        for idx, scaf in enumerate(scaffolds):
            scaffold_to_indices.setdefault(scaf, []).append(idx)
        
        # Sort by descending frequency
        scaffolds_sorted = sorted(scaffold_to_indices.keys(), key=lambda x: len(scaffold_to_indices[x]), reverse=True)
        
        total_size = len(self.dataset)
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
        
        self.splits["train"] = train_indices
        self.splits["valid"] = valid_indices
        self.splits["test"] = test_indices
    
    def get_dataloader(self, split, batch_size=32, shuffle=True, collate_fn=None):
        """
        Create a DataLoader for the requested split.
        """
        from torch.utils.data import Subset, DataLoader
        
        if split not in self.splits:
            raise ValueError(f"Split '{split}' does not exist in splits dict. Call random_split or scaffold_split first.")
        
        indices = self.splits[split]
        subset = Subset(self.dataset, indices)
        
        return DataLoader(subset, 
                          batch_size=batch_size, 
                          shuffle=shuffle, 
                          collate_fn=collate_fn)

# Example usage:
# dataset = ...
# smiles_list = [...]
# splitter = DatasetSplitter(dataset, smiles_list=smiles_list, random_seed=42)
# splitter.scaffold_split(valid_size=0.1, test_size=0.1)
# train_loader = splitter.get_dataloader('train', batch_size=32, shuffle=True, collate_fn=collate_fn)
# val_loader = splitter.get_dataloader('valid', batch_size=32, shuffle=False, collate_fn=collate_fn)
# test_loader = splitter.get_dataloader('test', batch_size=32, shuffle=False, collate_fn=collate_fn)
