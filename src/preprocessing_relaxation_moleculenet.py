# Purpose: Provide ASE atoms w/wo relaxed positions and properties for the dataset.
import pandas as pd
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

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

from joblib import Parallel, delayed


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
        # Generate a 3D conformation using ETKDG_v3
        AllChem.EmbedMolecule(mol, maxAttempts=10)
        AllChem.UFFOptimizeMolecule(mol, maxIters=100)
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
    atoms.calc = calculator
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.01, steps=100)
    return atoms


def main(args):
    # ----------------------------
    # 1. Basic configuration
    # ----------------------------
    device = args.device
    data_path = args.data_path
    smiles_cols = args.smiles_cols
    property_cols = args.property_cols
    # -------------------------------
    # 2. Prepare data and MLIP model
    # -------------------------------
    data = pd.read_csv(data_path)
    orbff = pretrained.orb_v2(device=device)
    calc = ORBCalculator(orbff, device=device)
    # -------------------------------
    # 3. Convert SMILES to ASE atoms
    # -------------------------------
    X = []
    XUFF = []
    XR = []
    SMILES = []
    Y = {key: [] for key in property_cols}

    # Convert SMILES to ASE atoms with 2D conformation while checking for errors
    for i, row in tqdm(data.iterrows()):
    # for i, row in tqdm(data.iloc[:10].iterrows()):
        try:
            # Contain 2D which have 3D conformation also for fair comparison. Otherwise molecules not failed to get 3D conformation will relatively easier.
            atoms_2D = smiles_to_ase(row[smiles_cols], conformation=False)
            atoms_3D = smiles_to_ase(row[smiles_cols], conformation=True)  
            X.append(json.loads(encode(atoms_2D)))
            XUFF.append(atoms_3D)
            SMILES.append(row[smiles_cols])
            for key in property_cols:
                Y[key].append(float(row[key]))
        except Exception as e:
            print(f"Error processing SMILES {row[smiles_cols]}: {e}")

    # Convert SMILES to ASE atoms with 3D conformation by parallelizing the relaxation process
    def relaxation_tempfunc(atoms, calc):
        # Relax the atoms
        atoms = atoms_relaxation(atoms, calc)
        # Convert to JSON
        atoms_dict = json.loads(encode(atoms))
        return atoms_dict
       
    results = Parallel(n_jobs=8)(delayed(relaxation_tempfunc)(atoms, calc) for atoms in XUFF)
    XR = results
    
    # -------------------------------
    # 4. Save the results
    # -------------------------------
    data_name = data_path.split('/')[-1].split('.')[0]
    with open(f'preprocessed_data/{data_name}_relaxed.pkl', 'wb') as f:
        pickle.dump({
            'X': X,
            'XR': XR,
            'SMILES': SMILES,
            'Y': Y
        }, f)
    # as well as JSON (takes much file storage space)
    # with open(f'{data_name}_relaxed.json', 'w') as f:
    #     json.dump({
    #         'X': X,
    #         'XR': XR,
    #         'SMILES': SMILES,
    #         'Y': Y
    #     }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse input arguments for the script.")

    parser.add_argument("--device", type=str, default='cuda', help="Device to use (e.g., cuda:0, cpu)")
    parser.add_argument("--data_path", type=str, default="/home/lucky/Projects/llacha/data/data/ClinTox_dataset.csv", help="Path to the dataset CSV file")
    parser.add_argument("--smiles_cols", type=str, default='smiles', help="SMILES columns to use")
    parser.add_argument("--property_cols", type=json.loads, default=['FDA_APPROVED', 'CT_TOX'], help="List of property columns to use")

    args = parser.parse_args()
    main(args)
