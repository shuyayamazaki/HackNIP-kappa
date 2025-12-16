#!/usr/bin/env python3
"""
Predict with a trained MODNet regression model using packed supercell features (.pkl.gz)
or raw feature .npy files, with optional metadata (mp_ids, generation_id, targets).
"""

from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from modnet.models import MODNetModel
from modnet.preprocessing import MODData


def flatten_regression_pred(pred: np.ndarray) -> np.ndarray:
    """Average committee/extra dimensions to 1-D regression scores."""
    arr = np.asarray(pred)
    while arr.ndim > 1:
        arr = arr.mean(axis=-1)
    return arr.reshape(-1)


def load_model(path: Path) -> MODNetModel:
    """Load a MODNet model saved by MODNetModel.save (falls back to pickle)."""
    if hasattr(MODNetModel, "load"):
        return MODNetModel.load(str(path))
    with open(path, "rb") as fp:
        return pickle.load(fp)


def read_pickle(path: Path) -> Dict:
    """Load a pickle or gzip pickle payload."""
    try:
        with gzip.open(path, "rb") as fp:
            return pickle.load(fp)
    except OSError:
        with open(path, "rb") as fp:
            return pickle.load(fp)


def infer_target_name(payload: Dict, explicit: Optional[str], default: str = "g") -> str:
    """Pick target name from CLI, payload target_key, property_columns, else default."""
    if explicit:
        return explicit
    if payload.get("target_key"):
        return str(payload["target_key"])
    props = payload.get("property_columns") or []
    if props:
        return str(props[0])
    return default


def infer_layer(payload: Dict, key: str, explicit: Optional[int]) -> int:
    """Use CLI layer if given, else payload['layer'], else inspect X_<split> keys."""
    if explicit is not None:
        return explicit
    if "layer" in payload:
        return int(payload["layer"])
    for value in payload.values():
        if isinstance(value, dict):
            for feat_key in value.keys():
                if feat_key.startswith(f"{key}_l") and feat_key[len(key) + 2 :].isdigit():
                    return int(feat_key[len(key) + 2 :])
    raise ValueError("Could not infer layer; please pass --layer explicitly.")


def build_moddata(matrix: np.ndarray, target_name: str) -> MODData:
    """Wrap features in MODData with dummy targets to satisfy MODNet API."""
    df = pd.DataFrame(matrix)
    dummy_targets = pd.Series(np.zeros(len(df)), name=target_name)
    md = MODData(df_featurized=df, targets=dummy_targets, target_names=[target_name])
    md.optimal_features = list(df.columns)
    return md


def load_mp_ids(mp_ids_path: Optional[Path], count: int) -> List[str]:
    """Load mp_ids from a text file (one per line) or generate sequential ids."""
    if mp_ids_path is None:
        return [str(i) for i in range(count)]
    with open(mp_ids_path, "r", encoding="utf-8") as fp:
        ids = [line.strip() for line in fp if line.strip()]
    if len(ids) != count:
        raise ValueError(
            f"mp_ids count mismatch: {len(ids)} ids vs {count} feature rows "
            f"(path={mp_ids_path})"
        )
    return ids


def infer_layer_from_npy(path: Path, key: str, explicit: Optional[int]) -> int:
    """Infer layer index from CLI or filename suffix like '<slug>_..._<key>_l11.npy'."""
    if explicit is not None:
        return explicit
    stem = path.stem
    marker = f"{key}_l"
    if marker in stem:
        try:
            return int(stem.split(marker)[-1])
        except ValueError:
            pass
    raise ValueError("Could not infer layer from npy filename; please pass --layer.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict with a trained MODNet regression model using packed supercell features (.pkl.gz) or raw .npy."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the saved MODNet model (.modnet from 3_train_modnet_from_supercells.py).",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--features",
        type=Path,
        help="Packed feature pickle (.pkl.gz) produced by 6_pack_supercell_features_to_pkl_gz.py.",
    )
    source_group.add_argument(
        "--npy",
        type=Path,
        help="Raw feature .npy (e.g., *_all_XPS_l11.npy) to predict without packing.",
    )
    parser.add_argument(
        "--meta-pickle",
        type=Path,
        default=None,
        help=(
            "Optional packed feature pickle to source metadata (mp_ids, generation_id, targets) "
            "when predicting from --npy."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which split inside the packed pickle to use (e.g., train, valid, test). Ignored if meta has no splits.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index to use. If omitted, tries payload['layer'] or infers from feature keys.",
    )
    parser.add_argument(
        "--key",
        default="XPS",
        help="Feature key prefix used when packing (default: XPS).",
    )
    parser.add_argument(
        "--target-name",
        default=None,
        help="Target name for MODData. Defaults to payload target_key/property_columns or 'g'.",
    )
    parser.add_argument(
        "--mp-ids-path",
        type=Path,
        default=None,
        help="Text file with mp_ids (one per line) for npy mode. Default: sequential ids.",
    )
    parser.add_argument(
        "--meta-id-key",
        default="mp_ids",
        help="Key name inside meta pickle to read ids from (default: mp_ids).",
    )
    parser.add_argument(
        "--id-column",
        default="mp_id",
        help="Name of the id column in the output CSV (default: mp_id).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path for predictions (defaults next to features file).",
    )
    args = parser.parse_args()

    model_path = args.model.resolve()

    payload_meta: Optional[Dict] = None
    if args.meta_pickle is not None:
        meta_path = args.meta_pickle.resolve()
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta pickle not found: {meta_path}")
        payload_meta = read_pickle(meta_path)

    if args.features is not None:
        feature_path = args.features.resolve()
        # Allow a .npy to be passed via --features for convenience.
        if feature_path.suffix in {".npy", ".npz"}:
            args.npy = feature_path
        else:
            try:
                payload = read_pickle(feature_path)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Failed to load feature pickle '{feature_path}'. "
                    "If this is a .npy, pass it via --npy."
                ) from exc

    generation_ids: Optional[List[str]] = None

    if args.npy is not None:
        npy_path = args.npy.resolve()
        if not npy_path.exists():
            raise FileNotFoundError(f"Feature .npy not found: {npy_path}")
        layer = infer_layer_from_npy(npy_path, args.key, args.layer)
        X = np.load(npy_path)
        target_name = args.target_name
        if target_name is None and payload_meta is not None:
            target_name = infer_target_name(payload_meta, target_name)
        target_name = target_name or "g"

        mp_ids = load_mp_ids(args.mp_ids_path, len(X))
        y_block: Dict = {}
        if payload_meta is not None:
            requested_split_key = f"X_{args.split}"
            meta_split_key = requested_split_key
            meta_split = payload_meta.get(requested_split_key)

            available_splits = [
                k for k, v in payload_meta.items() if k.startswith("X_") and isinstance(v, dict)
            ]
            if not isinstance(meta_split, dict):
                if len(available_splits) == 1:
                    meta_split_key = available_splits[0]
                    meta_split = payload_meta[meta_split_key]
                elif len(available_splits) == 0:
                    meta_split_key = None
                    meta_split = payload_meta
                else:
                    raise KeyError(
                        f"Split '{args.split}' not found in meta pickle."
                        f" Available splits: {available_splits or '<none>'}"
                    )

            logical_meta_split = meta_split_key.split("X_", 1)[-1] if meta_split_key else None
            meta_id_key = (args.meta_id_key or "mp_ids").strip() or "mp_ids"
            meta_mp_ids = meta_split.get(meta_id_key)
            if meta_mp_ids is None and meta_id_key != "mp_ids":
                meta_mp_ids = meta_split.get("mp_ids")
            if meta_mp_ids is not None and args.mp_ids_path is None:
                if len(meta_mp_ids) != len(X):
                    raise ValueError(
                        f"meta mp_ids length mismatch: {len(meta_mp_ids)} vs {len(X)} feature rows "
                        f"(split={args.split})"
                    )
                mp_ids = [str(mpid) for mpid in meta_mp_ids]

            gen_ids_raw = None
            if "generation_id" in meta_split:
                gen_ids_raw = meta_split.get("generation_id")
            elif "generation_ids" in meta_split:
                gen_ids_raw = meta_split.get("generation_ids")
            if gen_ids_raw is not None:
                if len(gen_ids_raw) != len(X):
                    raise ValueError(
                        f"generation_id length mismatch in meta: {len(gen_ids_raw)} vs {len(X)} feature rows "
                        f"(split={args.split})"
                    )
                generation_ids = [str(gid) for gid in gen_ids_raw]

            if logical_meta_split:
                y_split_key = f"Y_{logical_meta_split}"
                y_block = payload_meta.get(y_split_key, {})
            else:
                y_candidates = [
                    k for k, v in payload_meta.items() if k.startswith("Y_") and isinstance(v, dict)
                ]
                if len(y_candidates) == 1:
                    y_block = payload_meta[y_candidates[0]]

        default_out = npy_path.with_name(npy_path.stem + f"_pred_l{layer}.csv")
        mode_desc = f"npy ({npy_path.name})"
    else:
        payload = read_pickle(feature_path)  # type: ignore[arg-type]
        target_name = infer_target_name(payload, args.target_name)
        layer = infer_layer(payload, args.key, args.layer)

        split_key = f"X_{args.split}"
        if split_key not in payload or not isinstance(payload[split_key], dict):
            raise KeyError(f"Split '{args.split}' not found in {feature_path.name}.")

        split_block = payload[split_key]
        feat_key = f"{args.key}_l{layer}"
        if feat_key not in split_block:
            raise KeyError(
                f"Feature key '{feat_key}' missing in split '{args.split}'. "
                f"Available keys: {list(split_block.keys())}"
            )

        X = np.asarray(split_block[feat_key], dtype=np.float32)
        mp_ids: List[str] = [str(mpid) for mpid in split_block.get("mp_ids", range(len(X)))]
        gen_ids_raw = None
        if "generation_id" in split_block:
            gen_ids_raw = split_block.get("generation_id")
        elif "generation_ids" in split_block:
            gen_ids_raw = split_block.get("generation_ids")
        if gen_ids_raw is not None:
            if len(gen_ids_raw) != len(X):
                raise ValueError(
                    f"generation_id length mismatch: {len(gen_ids_raw)} vs {len(X)} feature rows "
                    f"(split={args.split}, layer={layer})"
                )
            generation_ids = [str(gid) for gid in gen_ids_raw]
        y_split_key = f"Y_{args.split}"
        y_block = payload.get(y_split_key, {})
        default_out = feature_path.with_name(
            feature_path.stem + f"_pred_split-{args.split}_l{layer}.csv"
        )
        mode_desc = f"packed pickle ({feature_path.name})"

    model = load_model(model_path)
    md = build_moddata(X, target_name=target_name)

    raw_pred = model.predict(md, remap_out_of_bounds=False)
    pred_values = flatten_regression_pred(raw_pred)

    id_column = (args.id_column or "mp_id").strip()
    if not id_column:
        raise ValueError("id column name cannot be empty.")

    output_payload = {id_column: mp_ids}
    if generation_ids is not None:
        output_payload["generation_id"] = generation_ids
    output_payload["prediction"] = pred_values
    output = pd.DataFrame(output_payload)

    # Attach ground truth if present.
    if isinstance(y_block, dict):
        if target_name in y_block:
            output["target"] = y_block[target_name]
        elif len(y_block) == 1:
            only_key = next(iter(y_block.keys()))
            output["target"] = y_block[only_key]

    out_path = args.output.resolve() if args.output is not None else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    print(f"[INFO] Model:   {model_path}")
    print(f"[INFO] Source:  {mode_desc}")
    print(f"[INFO] Split:   {args.split} | Layer: {layer} | Key: {args.key}")
    print(f"[INFO] ID column: {id_column}")
    if payload_meta is not None:
        print(f"[INFO] Meta id key: {(args.meta_id_key or 'mp_ids').strip() or 'mp_ids'}")
    print(f"[INFO] Saved predictions: {out_path} ({len(output)} rows)")


if __name__ == "__main__":
    main()
