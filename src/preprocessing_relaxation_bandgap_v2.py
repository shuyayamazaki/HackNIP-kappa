# Purpose: Build a dataset of ASE Atoms (unrelaxed/relaxed) plus properties from a CSV and CIF directory.
import os
import pickle
import argparse
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm
import json

from ase.io import read
from ase.io.jsonio import encode, decode
from ase.optimize import BFGS

from joblib import Parallel, delayed

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator


def atoms_relaxation(atoms, device: str, fmax: float = 0.01, steps: int = 100):
    """Relax ASE Atoms using ORB force field on a given device."""
    # create calculator inside worker (avoid pickling issues)
    orbff = pretrained.orb_v2(device=device)
    calc = ORBCalculator(orbff, device=device)

    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return atoms


def _relaxation_worker(atoms_dict: Dict, device: str):
    atoms = decode(json.dumps(atoms_dict))
    atoms = atoms_relaxation(atoms, device=device)
    return json.loads(encode(atoms))


def main(args):
    # ----------------------------
    # 1) Config
    # ----------------------------
    device: str = args.device
    csv_path = Path(args.csv_path)
    structures_dir = Path(args.structures_dir)
    id_col: str = args.id_col
    property_cols: List[str] = args.property_cols
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 2) Load CSV (properties)
    # ----------------------------
    df = pd.read_csv(csv_path)
    missing_cols = [c for c in property_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Property columns not found in CSV: {missing_cols}")
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in CSV.")

    # ----------------------------
    # 3) Load CIF -> ASE Atoms (unrelaxed)
    # ----------------------------
    X = []
    Y = {key: [] for key in property_cols}
    kept_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Reading CIFs"):
        mpid = str(row[id_col])  # e.g., "mp-12345"
        # if your CIF names are exactly "<mpid>.cif"
        cif_path = structures_dir / f"{mpid}.cif"
        if not cif_path.exists():
            # try fallback naming if your folder uses "mp-{index}.cif"
            # or just skip
            # cif_path = structures_dir / f"mp-{row.name}.cif"
            # if not cif_path.exists():
            print(f"[WARN] CIF not found: {cif_path}")
            continue

        try:
            atoms = read(str(cif_path))
            X.append(json.loads(encode(atoms)))
            for key in property_cols:
                Y[key].append(float(row[key]))
            kept_ids.append(mpid)
        except Exception as e:
            print(f"[ERROR] Failed to read/encode {cif_path}: {e}")

    if len(X) == 0:
        raise RuntimeError("No structures were loaded. Check paths and id/filenames.")

    # ----------------------------
    # 4) Relaxation (parallel, calculator per worker)
    # ----------------------------
    XR = Parallel(n_jobs=args.n_jobs, prefer="processes")(
        delayed(_relaxation_worker)(atoms_dict, device) for atoms_dict in tqdm(X, desc="Relaxation")
    )

    # ----------------------------
    # 5) Save pickle
    # ----------------------------
    out_name = csv_path.stem  # e.g., "MP" if MP.csv
    out_path = out_dir / f"{out_name}_relaxed.pkl"
    payload = {"ids": kept_ids, "X": X, "XR": XR, "Y": Y}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[OK] Saved: {out_path}  (structures={len(X)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make ASE dataset (unrelaxed/relaxed) from CSV + CIFs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g., cuda, cuda:0, or cpu")
    parser.add_argument("--csv_path", type=str, default="MP.csv", help="Path to the CSV with properties")
    parser.add_argument("--structures_dir", type=str, default="./structures", help="Directory containing <mpid>.cif files")
    parser.add_argument("--id_col", type=str, default="mpid", help="Column in CSV that matches CIF basename (e.g., mp-1234)")
    parser.add_argument("--property_cols", nargs="+", default=["y_train_log_klat"], help="Property column names to store in Y")
    parser.add_argument("--out_dir", type=str, default="preprocessed_data", help="Output directory")
    parser.add_argument("--n_jobs", type=int, default=8, help="Parallel workers for relaxation")
    args = parser.parse_args()
    main(args)