# Purpose: Provide ASE atoms w/wo relaxed positions and properties for the dataset.
from tqdm import tqdm
import pickle
import argparse
import json

from ase.io.jsonio import encode, decode
from ase.optimize import BFGS
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

# from joblib import Parallel, delayed


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
    property_cols = args.property_cols
    # -------------------------------
    # 2. Prepare data and MLIP model
    # -------------------------------    
    with open(data_path, 'rb') as f:
        mptrj = json.load(f)
    data = [
        (mp_id, graph_id) for mp_id, dct in mptrj.items() for graph_id in dct
    ]
    orbff = pretrained.orb_v2(device=device)
    calc = ORBCalculator(orbff, device=device)
    # -------------------------------
    # 3. Convert SMILES to ASE atoms
    # -------------------------------
    X = []
    XR = []
    Y = {key: [] for key in property_cols}

    # Convert SMILES to ASE atoms with 2D conformation while checking for errors
    for mp_id, graph_id in tqdm(data):
    # for i, row in tqdm(data.iloc[:10].iterrows()):
        try:
            # Contain 2D which have 3D conformation also for fair comparison. Otherwise molecules not failed to get 3D conformation will relatively easier.
            
            pmg_atoms = Structure.from_dict(mptrj[mp_id][graph_id]["structure"])
            # from pmgg_atoms to ase atoms
            atoms = AseAtomsAdaptor.get_atoms(pmg_atoms)
            X.append(json.loads(encode(atoms)))
            for key in property_cols:
                Y[key].append(float(mptrj[mp_id][graph_id][key]))
        except Exception as e:
            print(f"Error processing structure {mp_id}: {e}")

    # Convert SMILES to ASE atoms with 3D conformation by parallelizing the relaxation process
    def relaxation_tempfunc(atoms_dict, calc):
        # decode the atoms
        atoms = decode(json.dumps(atoms_dict))
        # Relax the atoms
        atoms = atoms_relaxation(atoms, calc)
        # Convert to JSON
        atoms_dict = json.loads(encode(atoms))
        return atoms_dict
       
    # XR = Parallel(n_jobs=8)(delayed(relaxation_tempfunc)(atoms_dict, calc) for atoms_dict in X)
    XR = X
        
    # -------------------------------
    # 4. Save the results
    # -------------------------------
    data_name = data_path.split('/')[-1].split('.')[0]
    with open(f'preprocessed_data/{data_name}_relaxed.pkl', 'wb') as f:
        pickle.dump({
            'X': X,
            'XR': XR,
            'Y': Y
        }, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse input arguments for the script.")

    parser.add_argument("--device", type=str, default='cpu', help="Device to use (e.g., cuda:0, cpu)")
    parser.add_argument("--data_path", type=str, default='/home/lucky/Projects/ion_conductivity/feat/data/MPtrj_2022.9_full_sampled.json', help="Path to the dataset CSV file")
    parser.add_argument("--property_cols", type=json.loads, default=["uncorrected_total_energy", "corrected_total_energy", "energy_per_atom", "ef_per_atom", "e_per_atom_relaxed", "ef_per_atom_relaxed"], help="List of property columns to use")

    args = parser.parse_args()
    main(args)
