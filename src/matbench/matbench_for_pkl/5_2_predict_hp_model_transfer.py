#!/usr/bin/env python3
"""
Predict train and test splits for given dataset slugs using a pre-trained MODNet model.

This script mirrors the feature/key/layer handling used by 5_retrain_hp_model_transfer.py
but skips retraining: it loads an existing MODNet model, runs predictions on the requested
slugs' train/test feature bundles, and saves the outputs as CSV files.
"""

import argparse
import csv
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from modnet.models import MODNetModel
from modnet.preprocessing import MODData


DATA_ROOT = Path(
    os.environ.get("BENCH_DATA_DIR", Path(__file__).resolve().parent / "benchmark_data")
).resolve()
MLIP = os.environ.get("BENCH_MLIP", "orb2")
MODEL = os.environ.get("BENCH_MODEL", "modnet")

FEAT_DIR = DATA_ROOT / f"feat_{MLIP}"
RESULTS_DIR = FEAT_DIR / f"results_{MODEL}"

for directory in [FEAT_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_TEST_SPLIT = "test"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_TARGET_NAME = "g"


def configure_devices(cuda_visible: Optional[str]) -> None:
    """Configure CUDA visibility and enable TF memory growth."""
    if cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] PyTorch device: {device}")
    print(f"[INFO] TensorFlow GPUs: {tf.config.list_logical_devices('GPU')}")


def parse_slug_list(spec: Optional[str]) -> List[str]:
    """Parse comma-separated dataset slugs."""
    if spec is None:
        return []
    slugs: List[str] = []
    seen = set()
    for token in spec.split(","):
        slug = token.strip()
        if not slug or slug in seen:
            continue
        slugs.append(slug)
        seen.add(slug)
    return slugs


def load_metadata(metadata_path: Path) -> Dict[str, object]:
    """Load hyperparameter metadata from JSON."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as meta_fp:
        return json.load(meta_fp)


def load_model(path: Path) -> MODNetModel:
    """Load a MODNet model saved by MODNetModel.save (falls back to pickle)."""
    if hasattr(MODNetModel, "load"):
        return MODNetModel.load(str(path))
    with open(path, "rb") as fp:
        return pickle.load(fp)


def load_feature_bundle(slug: str, key: str, split: str) -> Tuple[Dict, Path]:
    """Load the feature bundle for a given slug, split, and key."""
    filename = f"{slug}_{split}_{key}_{MLIP}.pkl"
    path = FEAT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Feature bundle not found: {path}")
    with open(path, "rb") as fp:
        bundle = pickle.load(fp)  # type: ignore[assignment]
    return bundle, path


def detect_available_layers(bundle: Dict, key: str) -> List[int]:
    """Detect layer indices available inside the bundle (handles XPS_lN and XPS_N)."""
    prefix_with_l = f"{key}_l"
    prefix_without = f"{key}_"
    layers: List[int] = []
    for feat_key in bundle.keys():
        suffix = None
        if feat_key.startswith(prefix_with_l):
            suffix = feat_key[len(prefix_with_l) :]
        elif feat_key.startswith(prefix_without):
            suffix = feat_key[len(prefix_without) :]
        if suffix is None or not suffix.isdigit():
            continue
        idx = int(suffix)
        if idx not in layers:
            layers.append(idx)
    return sorted(layers)


def get_layer_features(bundle: Dict, key: str, layer: int) -> np.ndarray:
    """Return feature matrix for the requested layer."""
    candidates = [f"{key}_l{layer}", f"{key}_{layer}"]
    for candidate in candidates:
        if candidate in bundle:
            return np.asarray(bundle[candidate], dtype=np.float32)
    raise KeyError(f"Layer {layer} not found for key '{key}'.")


def build_moddata_for_prediction(matrix: np.ndarray, target_name: str) -> MODData:
    """Wrap numpy arrays into MODData with dummy targets for prediction."""
    df = pd.DataFrame(matrix)
    dummy_targets = pd.Series(np.zeros(len(df)), name=target_name)
    moddata = MODData(df_featurized=df, targets=dummy_targets, target_names=[target_name])
    moddata.optimal_features = list(df.columns)
    return moddata


def flatten_predictions(pred: np.ndarray) -> np.ndarray:
    """Ensure predictions are 1-D arrays."""
    arr = np.asarray(pred)
    while arr.ndim > 1:
        arr = arr.mean(axis=-1)
    return arr.reshape(-1)


def determine_n_features(metadata: Dict[str, object], model: MODNetModel) -> int:
    """Pick feature count from metadata hyperparameters, falling back to the model."""
    hyperparams = metadata.get("hyperparameters")
    if isinstance(hyperparams, dict) and "n_features" in hyperparams:
        return int(hyperparams["n_features"])
    for attr in ("n_feat", "num_features"):
        if hasattr(model, attr):
            val = getattr(model, attr)
            if val is not None:
                return int(val)
    raise ValueError("Unable to determine n_features from metadata or model.")


def extract_ids(bundle: Dict, expected: int) -> List[str]:
    """Return mp_ids/ids if present, else a sequential index."""
    ids = bundle.get("mp_ids") or bundle.get("ids")
    if ids is None:
        return [str(i) for i in range(expected)]
    ids = [str(i) for i in ids]
    if len(ids) != expected:
        raise ValueError(f"id length ({len(ids)}) does not match feature rows ({expected}).")
    return ids


def extract_generation_ids(bundle: Dict, expected: int) -> Optional[List[str]]:
    """Return generation_id if available and aligned."""
    gen_ids = bundle.get("generation_id") or bundle.get("generation_ids")
    if gen_ids is None:
        return None
    gen_ids = [str(gid) for gid in gen_ids]
    if len(gen_ids) != expected:
        raise ValueError(
            f"generation_id length ({len(gen_ids)}) does not match feature rows ({expected})."
        )
    return gen_ids


def predict_split(
    model: MODNetModel,
    bundle: Dict,
    slug: str,
    split: str,
    feature_key: str,
    layer: int,
    n_features: int,
    target_name: str,
) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
    """Run prediction on one split and return the dataframe and optional metrics."""
    available_layers = detect_available_layers(bundle, feature_key)
    if layer not in available_layers:
        raise ValueError(
            f"Layer {layer} unavailable for slug '{slug}' split '{split}'. "
            f"Available: {available_layers}"
        )

    X_full = get_layer_features(bundle, feature_key, layer)
    if X_full.shape[1] < n_features:
        raise ValueError(
            f"Slug '{slug}' split '{split}' layer {layer} has only {X_full.shape[1]} features, "
            f"requires >= {n_features}."
        )

    X = X_full[:, :n_features]
    md = build_moddata_for_prediction(X, target_name=target_name)
    raw_pred = model.predict(md, remap_out_of_bounds=False)
    pred_values = flatten_predictions(raw_pred)

    if len(pred_values) != len(X):
        raise ValueError(
            f"Prediction length ({len(pred_values)}) does not match feature rows ({len(X)})."
        )

    ids = extract_ids(bundle, len(pred_values))
    gen_ids = extract_generation_ids(bundle, len(pred_values))

    df_payload = {"mp_id": ids, "prediction": pred_values}
    if gen_ids is not None:
        df_payload["generation_id"] = gen_ids

    metrics: Optional[Dict[str, float]] = None
    if "targets" in bundle:
        y_true = np.asarray(bundle["targets"], dtype=np.float32).reshape(-1)
        if len(y_true) != len(pred_values):
            raise ValueError(
                f"Target length ({len(y_true)}) does not match feature rows ({len(X)})."
            )
        df_payload["target"] = y_true
        metrics = {
            "mae": float(mean_absolute_error(y_true, pred_values)),
            "r2": float(r2_score(y_true, pred_values)),
        }

    return pd.DataFrame(df_payload), metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict train/test splits for slugs using a trained MODNet model."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="Path to metadata.json produced by opt_hp_modnet_from_supercells.py.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a trained MODNet model (.modnet or pickle).",
    )
    parser.add_argument(
        "--target-slugs",
        default="",
        help="Comma-separated dataset slugs to predict on.",
    )
    parser.add_argument(
        "--train-split",
        default=DEFAULT_TRAIN_SPLIT,
        help="Train split name to use (default: train).",
    )
    parser.add_argument(
        "--test-split",
        default=DEFAULT_TEST_SPLIT,
        help="Test split name to use (default: test).",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="CUDA_VISIBLE_DEVICES value to set before prediction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where prediction CSVs will be stored "
        "(default: <results_modnet>/hp_transfer_predictions/<meta>_<model>_<timestamp>).",
    )
    args = parser.parse_args()

    metadata_path = args.metadata_path.expanduser().resolve()
    metadata = load_metadata(metadata_path)
    if not isinstance(metadata.get("hyperparameters"), dict):
        raise ValueError("Metadata must include 'hyperparameters' dictionary.")

    layer = metadata.get("layer")
    if layer is None:
        raise ValueError("Metadata must include 'layer'.")
    layer = int(layer)

    feature_key = str(metadata.get("feature_key") or metadata.get("key") or "XPS").upper()
    target_name = str(metadata.get("target_property") or DEFAULT_TARGET_NAME)

    target_slugs = parse_slug_list(args.target_slugs)
    if not target_slugs:
        raise ValueError("Please provide at least one slug via --target-slugs.")

    configure_devices(args.cuda_visible_devices)

    model_path = args.model_path.expanduser().resolve()
    model = load_model(model_path)

    n_features = determine_n_features(metadata, model)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_stem = metadata_path.stem
    model_stem = model_path.stem
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = RESULTS_DIR / "hp_transfer_predictions" / f"{meta_stem}_{model_stem}_{timestamp}"
    elif not output_dir.is_absolute():
        output_dir = (RESULTS_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    metrics_rows: List[Dict[str, object]] = []

    for slug in target_slugs:
        print(f"[INFO] Predicting slug: {slug}")
        train_bundle, train_path = load_feature_bundle(slug, feature_key, args.train_split)
        test_bundle, test_path = load_feature_bundle(slug, feature_key, args.test_split)

        train_df, train_metrics = predict_split(
            model=model,
            bundle=train_bundle,
            slug=slug,
            split=args.train_split,
            feature_key=feature_key,
            layer=layer,
            n_features=n_features,
            target_name=target_name,
        )
        test_df, test_metrics = predict_split(
            model=model,
            bundle=test_bundle,
            slug=slug,
            split=args.test_split,
            feature_key=feature_key,
            layer=layer,
            n_features=n_features,
            target_name=target_name,
        )

        train_out = output_dir / f"{slug}_{args.train_split}_{feature_key}_{MLIP}_l{layer}_predictions.csv"
        test_out = output_dir / f"{slug}_{args.test_split}_{feature_key}_{MLIP}_l{layer}_predictions.csv"
        train_df.to_csv(train_out, index=False)
        test_df.to_csv(test_out, index=False)

        print(f"[INFO] Saved train predictions: {train_out} ({len(train_df)} rows)")
        print(f"[INFO] Saved test predictions:  {test_out} ({len(test_df)} rows)")

        if train_metrics is not None:
            metrics_rows.append(
                {
                    "metadata_stem": meta_stem,
                    "model_stem": model_stem,
                    "target_slug": slug,
                    "split": args.train_split,
                    "feature_key": feature_key,
                    "layer": layer,
                    "n_features": n_features,
                    "num_samples": len(train_df),
                    "mae": train_metrics["mae"],
                    "r2": train_metrics["r2"],
                    "source_path": str(train_path),
                    "timestamp": timestamp,
                }
            )
        if test_metrics is not None:
            metrics_rows.append(
                {
                    "metadata_stem": meta_stem,
                    "model_stem": model_stem,
                    "target_slug": slug,
                    "split": args.test_split,
                    "feature_key": feature_key,
                    "layer": layer,
                    "n_features": n_features,
                    "num_samples": len(test_df),
                    "mae": test_metrics["mae"],
                    "r2": test_metrics["r2"],
                    "source_path": str(test_path),
                    "timestamp": timestamp,
                }
            )

    if metrics_rows:
        metrics_path = output_dir / "prediction_metrics.csv"
        write_header = not metrics_path.exists()
        with open(metrics_path, "a", encoding="utf-8", newline="") as csv_fp:
            fieldnames = list(metrics_rows[0].keys())
            writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"[INFO] Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
