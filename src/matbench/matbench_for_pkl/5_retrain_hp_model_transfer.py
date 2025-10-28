#!/usr/bin/env python3
"""
Retrain a MODNet model using best hyperparameters from Optuna on alternate dataset slugs.

Given a metadata.json produced by opt_hp_modnet_from_supercells.py (alongside a saved trial
model), this script reuses the stored hyperparameters and layer selection to train new MODNet
models on other dataset slugs' train splits and evaluates them on their corresponding test splits.
"""

import argparse
import csv
import json
import os
import pickle
import random
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


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, Torch, and TensorFlow RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)


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


def make_moddata(matrix: np.ndarray, targets: np.ndarray, target_name: str) -> MODData:
    """Wrap numpy arrays into MODData with consistent feature naming."""
    df = pd.DataFrame(matrix)
    series = pd.Series(targets)
    moddata = MODData(df_featurized=df, targets=series, target_names=[target_name])
    moddata.optimal_features = list(df.columns)
    return moddata


def build_model(
    num_neurons: Tuple[Tuple[int, ...], ...],
    n_feat: int,
    target_name: str,
    out_act: str,
) -> MODNetModel:
    """Construct a MODNet model for regression."""
    return MODNetModel(
        targets=[[target_name]],
        weights={target_name: 1.0},
        num_neurons=num_neurons,
        n_feat=n_feat,
        num_classes={target_name: 0},
        out_act=out_act,
    )


def flatten_predictions(pred: np.ndarray) -> np.ndarray:
    """Ensure predictions are 1-D arrays."""
    arr = np.asarray(pred)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return arr.reshape(-1)


def load_metadata(metadata_path: Path) -> Dict[str, object]:
    """Load hyperparameter metadata from JSON."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as meta_fp:
        return json.load(meta_fp)


def normalise_num_neurons(value: object) -> Tuple[Tuple[int, ...], ...]:
    """Convert serialized num_neurons to tuple-of-tuples."""
    if isinstance(value, tuple):
        return tuple(tuple(int(x) for x in block) for block in value)
    if isinstance(value, list):
        return tuple(tuple(int(x) for x in block) for block in value)
    raise ValueError("Unsupported num_neurons structure in metadata.")


def derive_num_neurons(params: Dict[str, object]) -> Tuple[Tuple[int, ...], ...]:
    """Derive the MODNet num_neurons tuple from metadata."""
    if "num_neurons" in params:
        return normalise_num_neurons(params["num_neurons"])

    depth_val = params.get("depth")
    width_val = params.get("width")
    if depth_val is None or width_val is None:
        raise ValueError(
            "Metadata must include either 'num_neurons' or both 'depth' and 'width'."
        )

    depth = int(depth_val)
    width = int(width_val)
    if depth <= 0 or width <= 0:
        raise ValueError(
            f"Invalid depth/width in metadata hyperparameters: depth={depth}, width={width}"
        )

    depth = max(0, min(depth, 4))
    return tuple(([width] if idx < depth else []) for idx in range(4))


def train_with_hyperparams(
    params: Dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_name: str,
) -> Tuple[MODNetModel, Dict[str, float]]:
    """Train a MODNet model using stored hyperparameters and evaluate on train/test splits."""
    batch_size = int(params["batch_size"])
    learning_rate = float(params["learning_rate"])
    loss = str(params.get("loss", "mae"))
    out_act = str(params.get("out_act", "linear"))
    n_features = int(params["n_features"])
    num_neurons = derive_num_neurons(params)

    md_train = make_moddata(X_train[:, :n_features], y_train, target_name)
    md_test = make_moddata(X_test[:, :n_features], y_test, target_name)

    model = build_model(
        num_neurons=num_neurons,
        n_feat=n_features,
        target_name=target_name,
        out_act=out_act,
    )
    model.fit(
        md_train,
        batch_size=batch_size,
        lr=learning_rate,
        loss=loss,
    )

    train_pred = flatten_predictions(model.predict(md_train, remap_out_of_bounds=False))
    test_pred = flatten_predictions(model.predict(md_test, remap_out_of_bounds=False))

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain MODNet using best hyperparameters from Optuna on alternate slugs."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="Path to metadata.json produced by opt_hp_modnet_from_supercells.py.",
    )
    parser.add_argument(
        "--target-slugs",
        default="",
        help="Comma-separated dataset slugs to retrain on.",
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
        help="CUDA_VISIBLE_DEVICES value to set before training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where retrained models/metrics will be stored "
        "(default: <results_modnet>/hp_transfer_retrain/<meta_stem>_<timestamp>).",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="If set, save retrained models in the output directory.",
    )
    args = parser.parse_args()

    metadata_path = args.metadata_path.expanduser().resolve()
    metadata = load_metadata(metadata_path)

    hyperparams = metadata.get("hyperparameters")
    if not isinstance(hyperparams, dict):
        raise ValueError("Metadata must include 'hyperparameters' dictionary.")

    required_keys = {"batch_size", "learning_rate", "n_features"}
    missing = [key for key in required_keys if key not in hyperparams]
    if missing:
        raise ValueError(f"Metadata hyperparameters missing keys: {missing}")

    if "num_neurons" not in hyperparams and not {"depth", "width"} <= hyperparams.keys():
        raise ValueError(
            "Metadata hyperparameters must include 'num_neurons' or both 'depth' and 'width'."
        )

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
    set_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_stem = metadata_path.stem
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = RESULTS_DIR / "hp_transfer_retrain" / f"{meta_stem}_{timestamp}"
    elif not output_dir.is_absolute():
        output_dir = (RESULTS_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")

    models_dir = output_dir / "models"
    if args.save_models:
        models_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, object]] = []

    for slug in target_slugs:
        print(f"[INFO] Processing slug: {slug}")
        train_bundle, train_path = load_feature_bundle(slug, feature_key, args.train_split)
        test_bundle, test_path = load_feature_bundle(slug, feature_key, args.test_split)

        if "targets" not in train_bundle or "targets" not in test_bundle:
            raise KeyError(f"Feature bundle for slug '{slug}' missing 'targets'.")

        available_layers_train = detect_available_layers(train_bundle, feature_key)
        available_layers_test = detect_available_layers(test_bundle, feature_key)
        if layer not in available_layers_train or layer not in available_layers_test:
            raise ValueError(
                f"Layer {layer} unavailable for slug '{slug}'. "
                f"Train layers: {available_layers_train} | Test layers: {available_layers_test}"
            )

        X_train_full = get_layer_features(train_bundle, feature_key, layer)
        X_test_full = get_layer_features(test_bundle, feature_key, layer)
        n_features = int(hyperparams["n_features"])
        if X_train_full.shape[1] < n_features or X_test_full.shape[1] < n_features:
            raise ValueError(
                f"Slug '{slug}' layer {layer} has insufficient features "
                f"(train={X_train_full.shape[1]}, test={X_test_full.shape[1]}), "
                f"requires >= {n_features}."
            )

        y_train = np.asarray(train_bundle["targets"], dtype=np.float32)
        y_test = np.asarray(test_bundle["targets"], dtype=np.float32)

        model, metrics = train_with_hyperparams(
            params=hyperparams,
            X_train=X_train_full,
            y_train=y_train,
            X_test=X_test_full,
            y_test=y_test,
            target_name=target_name,
        )

        metrics_row = {
            "metadata_stem": meta_stem,
            "origin_slug": metadata.get("dataset"),
            "target_slug": slug,
            "feature_key": feature_key,
            "layer": layer,
            "n_features": n_features,
            "depth": int(hyperparams.get("depth", 0)),
            "width": int(hyperparams.get("width", 0)),
            "train_split": args.train_split,
            "test_split": args.test_split,
            "num_train_samples": len(y_train),
            "num_test_samples": len(y_test),
            "test_mae": metrics["test_mae"],
            "test_r2": metrics["test_r2"],
            "train_mae": metrics["train_mae"],
            "train_r2": metrics["train_r2"],
            "timestamp": timestamp,
        }
        metrics_rows.append(metrics_row)

        print(
            f"[RESULT] Slug {slug} | Train MAE {metrics['train_mae']:.6f} | "
            f"Train R2 {metrics['train_r2']:.6f} | "
            f"Test MAE {metrics['test_mae']:.6f} | Test R2 {metrics['test_r2']:.6f}"
        )

        if args.save_models:
            model_path = models_dir / f"{slug}_{feature_key}_{MLIP}_l{layer}.modnet"
            model.save(str(model_path))
            print(f"[INFO] Saved retrained model: {model_path}")

    if metrics_rows:
        metrics_path = output_dir / "transfer_retrain_metrics.csv"
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
