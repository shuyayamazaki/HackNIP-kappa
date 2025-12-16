# Build supercells for arbitrary structures supplied in a CSV or pickle file.
#
# The input must contain a "cif" column with CIF text for each structure.
# Optionally provide --id-column to use a specific column as the structure id;
# otherwise the table index (as string) is used. No properties are attached
# because these structures are only for inference, not training.

import argparse
import pickle
from io import StringIO
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ase.build import make_supercell
from ase.io import read
from ase.io.trajectory import Trajectory
from tqdm import tqdm


def build_supercell(atoms, target_length: float):
    """Create a supercell whose minimum cell vector length exceeds target_length."""
    a, b, c, _, _, _ = atoms.cell.cellpar()
    scale = np.ceil(target_length / np.array([a, b, c])).astype(int)
    scale = np.maximum(scale, 1)
    if np.any(scale > 1):
        return make_supercell(atoms, np.diag(scale))
    return atoms.copy()


def write_trajectory(path: Path, atoms_iterable: Iterable) -> None:
    """Write a sequence of ASE Atoms objects to a trajectory file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Trajectory(path, mode="w") as traj:
        for atoms in atoms_iterable:
            traj.write(atoms)


def load_table(path: Path) -> pd.DataFrame:
    """Load a CSV or pickle DataFrame containing a 'cif' column."""
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".csv"):
        return pd.read_csv(path)
    if suffix.endswith(".pkl") or suffix.endswith(".pkl.gz") or suffix.endswith(".pickle") or suffix.endswith(".pickle.gz"):
        return pd.read_pickle(path)
    raise ValueError(
        f"Unsupported input format for {path}. Expected a CSV or pickle with a 'cif' column."
    )


def resolve_ids(df: pd.DataFrame, id_column: Optional[str]) -> List[str]:
    """Return a list of string ids, validating uniqueness."""
    if id_column:
        if id_column not in df.columns:
            raise KeyError(f"Specified id column '{id_column}' not found in input table.")
        ids = df[id_column].astype(str).tolist()
    else:
        ids = df.index.astype(str).tolist()

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ids detected; please ensure ids are unique.")
    return ids


def resolve_output_root(input_path: Path, explicit_output: Optional[Path]) -> Path:
    """Determine the root directory for generated metadata/structures."""
    root = explicit_output if explicit_output is not None else input_path.parent / "benchmark_data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def prepare_output_dirs(root: Path) -> Tuple[Path, Path]:
    """Ensure metadata/structures sub-directories exist under the output root."""
    metadata_dir = root / "metadata"
    structures_dir = root / "structures"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    structures_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir, structures_dir


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build supercells for arbitrary CIF strings stored in a CSV or pickle. "
            "Outputs trajectories + metadata compatible with downstream featurization."
        )
    )
    parser.add_argument(
        "--input-path",
        "--csv-path",
        dest="input_path",
        type=Path,
        required=True,
        help="CSV or pickle file containing a 'cif' column with CIF text.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Optional column name to use as unique structure ids (defaults to table index).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination root directory for generated files. "
        "Metadata is written under <out_dir>/metadata/ and trajectories under "
        "<out_dir>/structures/. Defaults to <input_dir>/benchmark_data/.",
    )
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default=None,
        help="Dataset name to use in output filenames (defaults to input stem).",
    )
    parser.add_argument(
        "--target-length",
        type=float,
        default=10.0,
        help="Minimum cell vector length (Å) after supercelling.",
    )
    parser.add_argument(
        "--skip-base-traj",
        action="store_true",
        help="Only write the supercell trajectory (skip writing the base _XP.traj file).",
    )
    args = parser.parse_args()

    input_path = args.input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = load_table(input_path)
    if "cif" not in df.columns:
        raise KeyError("Input must contain a 'cif' column with CIF text.")

    ids = resolve_ids(df, args.id_column)
    cif_strings: Sequence[str] = df["cif"].tolist()

    slug = args.dataset_slug or input_path.stem
    output_root = resolve_output_root(input_path, args.output_dir)
    metadata_dir, structures_dir = prepare_output_dirs(output_root)

    print(f"[INFO] Structures to process: {len(ids)}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Metadata directory: {metadata_dir}")
    print(f"[INFO] Structures directory: {structures_dir}")

    base_atoms: List[object] = []
    super_atoms: List[object] = []

    for cif_str, struct_id in tqdm(zip(cif_strings, ids), total=len(ids), desc="Building supercells"):
        atoms = read(StringIO(cif_str), format="cif")
        base_atoms.append(atoms)
        super_atoms.append(build_supercell(atoms, target_length=args.target_length))

    if not args.skip_base_traj:
        base_path = structures_dir / f"{slug}_all_XP.traj"
        write_trajectory(base_path, base_atoms)
        print(f"[INFO] Wrote base trajectory: {base_path}")

    super_path = structures_dir / f"{slug}_all_XPS.traj"
    write_trajectory(super_path, super_atoms)
    print(f"[INFO] Wrote super trajectory: {super_path}")

    meta = {
        "ids": ids,
        "dataset": slug,
        "property_columns": [],
        "properties": {},
    }
    meta_path = metadata_dir / f"{slug}_meta.pkl"
    with open(meta_path, "wb") as meta_fp:
        pickle.dump(meta, meta_fp)
    print(f"[INFO] Wrote metadata: {meta_path}")
    print("[DONE] Supercell construction complete.")


if __name__ == "__main__":
    main()
