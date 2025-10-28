"""Combined featurization + pickle construction starting from pre-built supercells.

This script mirrors the behaviour of:
  * 3_featurize_orb2.py — generate ORB2 graph features layer-by-layer.
  * 4_construct_pkl.py — assemble metadata + features into a single pickle.

It assumes supercells were created via build_supercells_from_pkl.py and stored as
<slug>_all_XPS.traj (with an accompanying <slug>_meta.pkl) under the benchmark data root.
If per-split files such as <slug>_train_meta.pkl / <slug>_train_XPS.traj exist, the
script keeps those splits separate while also building the aggregate bundle.
"""

import argparse
import pickle
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from ase.io import read
from tqdm import tqdm

from orb_models.forcefield import atomic_system, pretrained


def default_data_root() -> Path:
    return Path(__file__).resolve().parent / "benchmark_data"


def prepare_dirs(data_root: Path, mlip: str, model: str) -> Dict[str, Path]:
    structures_dir = data_root / "structures"
    meta_dir = data_root / "metadata"
    feat_dir = data_root / f"feat_{mlip}"
    npy_dir = feat_dir / "npy"
    results_dir = feat_dir / f"results_{model}"
    hp_dir = results_dir / "hp"
    parity_dir = results_dir / "parity"

    for p in [structures_dir, meta_dir, feat_dir, npy_dir, results_dir, hp_dir, parity_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "structures": structures_dir,
        "meta": meta_dir,
        "feat": feat_dir,
        "npy": npy_dir,
    }


def discover_split_meta(meta_dir: Path, slug: str) -> Dict[str, Path]:
    """Return mapping of split name -> meta path (excluding aggregate meta)."""
    split_paths: Dict[str, Path] = {}
    for meta_path in sorted(meta_dir.glob(f"{slug}_*_meta.pkl")):
        if meta_path.name == f"{slug}_meta.pkl":
            continue
        stem = meta_path.stem  # e.g. "<slug>_train_meta"
        suffix = stem[len(slug) + 1 :]  # remove "<slug>_"
        if not suffix.endswith("_meta"):
            continue
        split = suffix[: -len("_meta")]
        split_paths[split] = meta_path
    return split_paths


def get_graph_features(atoms, model, layer_until: int, device: str) -> np.ndarray:
    graph = atomic_system.ase_atoms_to_atom_graphs(atoms, device=device)
    graph = model.featurize_edges(graph)
    graph = model.featurize_nodes(graph)
    graph = model._encoder(graph)

    for gnn in model.gnn_stacks[:layer_until]:
        graph = gnn(graph)

    graph = model._decoder(graph)
    return graph.node_features["feat"].mean(dim=0).cpu().numpy()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_features(
    slug: str,
    atoms_list: List,
    model,
    device: str,
    npy_dir: Path,
    split_indices: Dict[str, np.ndarray],
    overwrite: bool = False,
) -> None:
    key = "XPS"
    splits = list(split_indices.keys())

    def layer_cache_complete(layer: int) -> bool:
        all_path = npy_dir / f"{slug}_all_{key}_l{layer}.npy"
        split_paths = [
            npy_dir / f"{slug}_{split}_{key}_l{layer}.npy"
            for split in splits
        ]
        return all(path.exists() for path in [all_path, *split_paths])

    for layer in range(1, 16):
        npy_path = npy_dir / f"{slug}_all_{key}_l{layer}.npy"
        if not overwrite:
            if splits and layer_cache_complete(layer):
                print(f"[INFO] Skip layer {layer}: cached all/split feature arrays.")
                continue
            if not splits and npy_path.exists():
                print(f"[INFO] Skip layer {layer}: {npy_path.name} already exists.")
                continue

        layer_features = [
            get_graph_features(atoms, model, layer_until=layer, device=device)
            for atoms in tqdm(atoms_list, desc=f"[{slug}] featurizing layer {layer:02d}")
        ]
        layer_array = np.asarray(layer_features, dtype=np.float32)
        np.save(npy_path, layer_array)
        print(f"[OK] Saved features: {npy_path}")

        for split, indices in split_indices.items():
            split_path = npy_dir / f"{slug}_{split}_{key}_l{layer}.npy"
            split_features = layer_array[indices]
            np.save(split_path, split_features)
            print(f"[OK] Saved split features: {split_path}")

        # Free intermediate array before next layer
        del layer_features
        del layer_array
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_pickles(
    slug: str,
    meta_all: Dict,
    atoms_all: List,
    mlip: str,
    feat_dir: Path,
    npy_dir: Path,
    meta_dir: Path,
    split_infos: Dict[str, Dict[str, object]],
) -> None:
    key = "XPS"
    key_lower = key.lower()

    data_payload = dict(meta_all)
    data_payload[f"{key_lower}_atoms"] = atoms_all

    meta_path = meta_dir / f"{slug}_meta.pkl"
    data_path = meta_path.with_name(meta_path.name.replace("_meta.pkl", "_data.pkl"))

    with open(data_path, "wb") as dp:
        pickle.dump(data_payload, dp)
    print(f"[OK] Saved data payload: {data_path}")

    feat_payload = dict(data_payload)
    for layer in range(1, 16):
        npy_path = npy_dir / f"{slug}_all_{key}_l{layer}.npy"
        feat_payload[f"{key}_{layer}"] = np.load(npy_path)

    feat_path = feat_dir / f"{slug}_{key}_{mlip}.pkl"
    with open(feat_path, "wb") as fp:
        pickle.dump(feat_payload, fp)
    print(f"[OK] Saved feature bundle: {feat_path}")

    for split, info in split_infos.items():
        split_meta_path = info["meta_path"]
        split_meta = dict(info["meta"])
        split_atoms = info["atoms"]
        split_data = dict(split_meta)
        split_data[f"{key_lower}_atoms"] = split_atoms

        split_data_path = split_meta_path.with_name(
            split_meta_path.name.replace("_meta.pkl", "_data.pkl")
        )
        with open(split_data_path, "wb") as dp:
            pickle.dump(split_data, dp)
        print(f"[OK] Saved split data payload: {split_data_path}")

        split_feat = dict(split_data)
        for layer in range(1, 16):
            split_npy = npy_dir / f"{slug}_{split}_{key}_l{layer}.npy"
            split_feat[f"{key}_{layer}"] = np.load(split_npy)

        split_feat_path = feat_dir / f"{slug}_{split}_{key}_{mlip}.pkl"
        with open(split_feat_path, "wb") as fp:
            pickle.dump(split_feat, fp)
        print(f"[OK] Saved split feature bundle: {split_feat_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Featurize ORB supercells and assemble MODNet-ready pickle bundles."
    )
    parser.add_argument("--slug", required=True, help="Dataset identifier, e.g., mp_gap.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="Benchmark data root (default: ./benchmark_data next to this script).",
    )
    parser.add_argument("--mlip", default="orb2", help="MLIP tag used for directory naming.")
    parser.add_argument("--model", default="modnet", help="Downstream model tag for directory naming.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for ORB inference (auto|cpu|cuda:0 ...).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute features even if npy files already exist.",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    dirs = prepare_dirs(data_root, args.mlip, args.model)

    structures_all_path = dirs["structures"] / f"{args.slug}_all_XPS.traj"
    meta_all_path = dirs["meta"] / f"{args.slug}_meta.pkl"
    if not structures_all_path.exists():
        raise FileNotFoundError(f"Supercell trajectory not found: {structures_all_path}")
    if not meta_all_path.exists():
        raise FileNotFoundError(f"Metadata pickle not found: {meta_all_path}")

    split_meta_paths = discover_split_meta(dirs["meta"], args.slug)
    if split_meta_paths:
        print(f"[INFO] Detected split metadata: {list(split_meta_paths.keys())}")

    with open(meta_all_path, "rb") as mf:
        meta_all = pickle.load(mf)

    print(f"[INFO] Loading supercells from {structures_all_path}")
    atoms_list = list(read(str(structures_all_path), index="::"))
    print(f"[INFO] Loaded {len(atoms_list)} structures.")

    ids_all = [str(mpid) for mpid in meta_all.get("ids", [])]
    if len(ids_all) != len(atoms_list):
        raise ValueError(
            f"Mismatch between metadata ids ({len(ids_all)}) and structures ({len(atoms_list)})."
        )
    id_to_index = {mpid: idx for idx, mpid in enumerate(ids_all)}

    split_infos: Dict[str, Dict[str, object]] = {}
    for split, meta_path in split_meta_paths.items():
        with open(meta_path, "rb") as sf:
            split_meta = pickle.load(sf)
        split_ids = [str(mpid) for mpid in split_meta.get("ids", [])]
        indices = []
        for mpid in split_ids:
            if mpid not in id_to_index:
                raise KeyError(
                    f"mp-id '{mpid}' from split '{split}' not found in aggregate metadata."
                )
            indices.append(id_to_index[mpid])
        index_array = np.asarray(indices, dtype=np.int64)
        atoms_subset = [atoms_list[idx].copy() for idx in index_array]
        split_infos[split] = {
            "meta_path": meta_path,
            "meta": split_meta,
            "indices": index_array,
            "atoms": atoms_subset,
        }

    set_seed()

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device={device}")

    orbff = pretrained.orb_v2(device=device)
    orbff.model.eval()

    compute_features(
        slug=args.slug,
        atoms_list=atoms_list,
        model=orbff.model,
        device=device,
        npy_dir=dirs["npy"],
        split_indices={split: info["indices"] for split, info in split_infos.items()},
        overwrite=args.overwrite,
    )

    build_pickles(
        slug=args.slug,
        meta_all=meta_all,
        atoms_all=atoms_list,
        mlip=args.mlip,
        feat_dir=dirs["feat"],
        npy_dir=dirs["npy"],
        meta_dir=dirs["meta"],
        split_infos=split_infos,
    )

    print("[DONE] Featurization and pickle construction complete.")


if __name__ == "__main__":
    main()
