# Revised supercell builder that consumes split pickle payloads so the
# existing MODNet pipeline can reuse its trajectory-based workflow while
# preserving per-split datasets (train/test/valid) and configurable targets.

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase.build import make_supercell
from ase.io import read
from ase.io.trajectory import Trajectory
from tqdm import tqdm


def load_pickle_gz(path: Path):
    """Load a (possibly gzipped) pickle payload."""
    try:
        with gzip.open(path, "rb") as fp:
            return pickle.load(fp)
    except OSError:
        with open(path, "rb") as fp:
            return pickle.load(fp)


def infer_dataset_slug(pickle_path: Path, override: Optional[str]) -> str:
    if override:
        return override
    name = pickle_path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".pkl"):
        name = name[:-4]
    return name


def detect_splits(data: Dict) -> List[Tuple[str, str, List[str]]]:
    """
    Detect available splits inside the pickle. Returns a list of tuples:
    (raw_split_name, logical_split_name, mp_ids_as_strings)
    """
    discovered: List[Tuple[str, str, List[str]]] = []
    for raw in ("train", "valid", "val", "test"):
        key = f"X_{raw}"
        if key not in data or not isinstance(data[key], dict):
            continue
        mp_ids = data[key].get("mp_ids")
        if mp_ids is None:
            continue
        logical = "valid" if raw == "val" else raw
        discovered.append(
            (raw, logical, [str(mpid) for mpid in mp_ids])
        )
    if not discovered:
        raise KeyError(
            "Could not locate any split data (expected keys like 'X_train[\"mp_ids\"]')."
        )
    return discovered


def ordered_unique(sequence: Sequence[str]) -> List[str]:
    """Return order-preserving unique elements from the provided sequence."""
    seen = set()
    unique: List[str] = []
    for item in sequence:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def resolve_property_columns(dataset: Dict, explicit: Optional[Sequence[str]]) -> List[str]:
    """Determine which property columns should be extracted."""
    if explicit:
        cols = [c.strip() for c in explicit if c and c.strip()]
        if cols:
            return cols
    target_key = dataset.get("target_key")
    if target_key:
        return [str(target_key)]
    for split in ("train", "valid", "val", "test"):
        y_key = f"Y_{split}"
        y_dict = dataset.get(y_key)
        if isinstance(y_dict, dict) and y_dict:
            first = next(iter(y_dict.keys()))
            return [str(first)]
    raise ValueError(
        "Could not infer property columns from the dataset. "
        "Please provide --property-cols explicitly."
    )


def extract_properties_for_split(
    dataset: Dict,
    raw_split: str,
    logical_split: str,
    property_cols: Sequence[str],
) -> Dict[str, List[float]]:
    """Extract property values for a particular split."""
    results: Dict[str, List[float]] = {}
    y_keys = []
    for candidate in {raw_split, logical_split}:
        y_keys.append(f"Y_{candidate}")
    for prop in property_cols:
        values = None
        for y_key in y_keys:
            y_dict = dataset.get(y_key)
            if isinstance(y_dict, dict) and prop in y_dict:
                values = y_dict[prop]
                break
        if values is None:
            prop_dict = dataset.get(prop)
            if isinstance(prop_dict, dict):
                if raw_split in prop_dict:
                    values = prop_dict[raw_split]
                elif logical_split in prop_dict:
                    values = prop_dict[logical_split]
        if values is None:
            candidate_keys = [
                f"{prop}_{raw_split}",
                f"{prop}_{logical_split}",
                f"y_{raw_split}_{prop}",
                f"y_{logical_split}_{prop}",
                prop,
            ]
            for key in candidate_keys:
                if key in dataset:
                    values = dataset[key]
                    break
        if values is None:
            raise KeyError(
                f"Unable to locate property '{prop}' for split '{logical_split}'. "
                f"Tried keys: {y_keys + candidate_keys}."
            )
        results[prop] = [float(v) for v in values]
    return results


def build_meta_for_split(
    slug: str,
    logical_split: str,
    mp_ids: Sequence[str],
    properties: Dict[str, List[float]],
    property_cols: Sequence[str],
) -> Dict:
    """Create a meta payload for an individual split."""
    meta = {
        "ids": list(mp_ids),
        "dataset": slug,
        "split": logical_split,
        "property_columns": list(property_cols),
    }
    if property_cols:
        target_prop = property_cols[0]
        meta["target_property"] = target_prop
        meta["targets"] = properties[target_prop]
    meta["properties"] = {prop: list(values) for prop, values in properties.items()}
    return meta


def write_trajectory(path: Path, atoms_iterable: Iterable):
    """Write a sequence of ASE Atoms objects to a trajectory file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Trajectory(path, mode="w") as traj:
        for atoms in atoms_iterable:
            traj.write(atoms)


def build_supercell(atoms, target_length: float):
    """Create a supercell whose minimum cell vector length exceeds target_length."""
    a, b, c, _, _, _ = atoms.cell.cellpar()
    scale = np.ceil(target_length / np.array([a, b, c])).astype(int)
    scale = np.maximum(scale, 1)
    if np.any(scale > 1):
        return make_supercell(atoms, np.diag(scale))
    return atoms.copy()


def build_meta_payload(
    slug: str,
    mp_ids: Sequence[str],
    property_lookup: Dict[str, Dict[str, float]],
    property_cols: Sequence[str],
) -> Dict:
    """Construct aggregate metadata covering all splits."""
    meta = {
        "ids": list(mp_ids),
        "dataset": slug,
        "property_columns": list(property_cols),
    }
    if property_cols:
        target_prop = property_cols[0]
        targets: List[float] = []
        for mpid in mp_ids:
            try:
                targets.append(float(property_lookup[target_prop][mpid]))
            except KeyError as exc:
                raise KeyError(
                    f"Missing value for property '{target_prop}' (mp-id: {mpid})."
                ) from exc
        meta["target_property"] = target_prop
        meta["targets"] = targets

    properties_block: Dict[str, List[float]] = {}
    for prop in property_cols:
        values: List[float] = []
        lookup = property_lookup.get(prop, {})
        for mpid in mp_ids:
            if mpid not in lookup:
                raise KeyError(
                    f"Missing value for property '{prop}' (mp-id: {mpid})."
                )
            values.append(float(lookup[mpid]))
        properties_block[prop] = values
    if properties_block:
        meta["properties"] = properties_block
    return meta


def prepare_atoms_for_pool(
    mp_ids: Sequence[str],
    base_structures_dir: Path,
    target_length: float,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Preload base structures and corresponding supercells for the mp-id pool."""
    base_atoms: Dict[str, object] = {}
    super_atoms: Dict[str, object] = {}
    for mpid in tqdm(mp_ids, desc="Loading & building supercells"):
        cif_path = base_structures_dir / f"{mpid}.cif"
        if not cif_path.exists():
            raise FileNotFoundError(f"Missing CIF for mp-id {mpid}: {cif_path}")
        atoms = read(str(cif_path))
        base_atoms[mpid] = atoms
        super_atoms[mpid] = build_supercell(atoms, target_length=target_length)
    return base_atoms, super_atoms


def resolve_output_root(pickle_path: Path, explicit_output: Optional[Path]) -> Path:
    """Determine the root directory for generated metadata/structures."""
    if explicit_output is not None:
        root = explicit_output
    else:
        root = pickle_path.parent / "benchmark_data"
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
            "Build supercells for bandgap data specified in a pickle(.gz) split. "
            "Outputs trajectory files compatible with the matbench MODNet pipeline."
        )
    )
    parser.add_argument(
        "--pickle_path",
        type=Path,
        required=True,
        help="Split pickle(.gz) consumed by preprocessing_relaxation_bandgap_pkl.py",
    )
    parser.add_argument(
        "--structures_dir",
        type=Path,
        required=True,
        help="Directory containing <mpid>.cif files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Destination root directory for generated files. "
             "Metadata is written under <out_dir>/metadata/ and trajectories under "
             "<out_dir>/structures/. Defaults to <pickle_dir>/benchmark_data/.",
    )
    parser.add_argument(
        "--dataset_slug",
        type=str,
        default=None,
        help="Optional dataset name to use in output filenames (defaults to pickle stem).",
    )
    parser.add_argument(
        "--property_cols",
        nargs="+",
        default=None,
        help="Property column names to extract (first entry is used as the primary target).",
    )
    parser.add_argument(
        "--target_length",
        type=float,
        default=10.0,
        help="Minimum cell vector length (Å) after supercelling.",
    )
    parser.add_argument(
        "--skip_base_traj",
        action="store_true",
        help="Only write the supercell trajectory (skip writing the base _XP.traj file).",
    )
    args = parser.parse_args()

    dataset = load_pickle_gz(args.pickle_path)
    splits = detect_splits(dataset)
    property_cols = resolve_property_columns(dataset, args.property_cols)

    mpid_pool = ordered_unique(
        mpid for _, _, mpids in splits for mpid in mpids
    )
    if not mpid_pool:
        raise RuntimeError("No mp-ids found in the provided pickle splits.")

    output_root = resolve_output_root(args.pickle_path, args.output_dir)
    metadata_dir, structures_dir = prepare_output_dirs(output_root)

    slug = infer_dataset_slug(args.pickle_path, args.dataset_slug)
    print(f"[INFO] Detected splits: {[logical for _, logical, _ in splits]}")
    print(f"[INFO] Unique structures to process: {len(mpid_pool)}")
    print(f"[INFO] Target properties: {property_cols}")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Metadata directory: {metadata_dir}")
    print(f"[INFO] Structures directory: {structures_dir}")

    base_atoms, super_atoms = prepare_atoms_for_pool(
        mp_ids=mpid_pool,
        base_structures_dir=args.structures_dir,
        target_length=args.target_length,
    )

    property_lookup: Dict[str, Dict[str, float]] = {prop: {} for prop in property_cols}

    for raw_split, logical_split, mp_ids in splits:
        print(f"[INFO] Writing split '{logical_split}' ({len(mp_ids)} structures)")
        properties = extract_properties_for_split(dataset, raw_split, logical_split, property_cols)
        for prop, values in properties.items():
            if len(values) != len(mp_ids):
                raise ValueError(
                    f"Property '{prop}' length ({len(values)}) does not match "
                    f"number of mp_ids ({len(mp_ids)}) for split '{logical_split}'."
                )
            lookup = property_lookup.setdefault(prop, {})
            for mpid, value in zip(mp_ids, values):
                lookup[str(mpid)] = float(value)

        base_seq = [base_atoms[str(mpid)] for mpid in mp_ids] if not args.skip_base_traj else []
        super_seq = [super_atoms[str(mpid)] for mpid in mp_ids]

        if not args.skip_base_traj:
            split_base_path = structures_dir / f"{slug}_{logical_split}_XP.traj"
            write_trajectory(split_base_path, base_seq)
            print(f"[INFO] Wrote base trajectory: {split_base_path}")

        split_super_path = structures_dir / f"{slug}_{logical_split}_XPS.traj"
        write_trajectory(split_super_path, super_seq)
        print(f"[INFO] Wrote super trajectory: {split_super_path}")

        split_meta = build_meta_for_split(slug, logical_split, mp_ids, properties, property_cols)
        split_meta_path = metadata_dir / f"{slug}_{logical_split}_meta.pkl"
        with open(split_meta_path, "wb") as meta_fp:
            pickle.dump(split_meta, meta_fp)
        print(f"[INFO] Wrote split metadata: {split_meta_path}")

    if property_cols:
        missing = [
            (prop, mpid)
            for prop, lookup in property_lookup.items()
            for mpid in mpid_pool
            if mpid not in lookup
        ]
        if missing:
            raise KeyError(
                "Missing property values after split processing: "
                + ", ".join(f"{prop}:{mpid}" for prop, mpid in missing[:10])
                + ("..." if len(missing) > 10 else "")
            )

    if not args.skip_base_traj:
        all_base_path = structures_dir / f"{slug}_all_XP.traj"
        write_trajectory(all_base_path, [base_atoms[mpid] for mpid in mpid_pool])
        print(f"[INFO] Wrote aggregate base trajectory: {all_base_path}")

    all_super_path = structures_dir / f"{slug}_all_XPS.traj"
    write_trajectory(all_super_path, [super_atoms[mpid] for mpid in mpid_pool])
    print(f"[INFO] Wrote aggregate super trajectory: {all_super_path}")

    meta_payload = build_meta_payload(slug, mpid_pool, property_lookup, property_cols)
    meta_path = metadata_dir / f"{slug}_meta.pkl"
    with open(meta_path, "wb") as meta_fp:
        pickle.dump(meta_payload, meta_fp)
    print(f"[INFO] Wrote aggregate metadata: {meta_path}")

    print("[DONE] Supercell construction complete.")


if __name__ == "__main__":
    main()
