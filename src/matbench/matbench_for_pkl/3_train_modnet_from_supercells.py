#!/usr/bin/env python3
"""
Train MODNet models using feature bundles generated from supercell trajectories.

This script mirrors the behaviour of 5_train_modnet.py but operates on the
pickles produced by build_supercells_from_pkl.py and
featurize_construct_from_supercells.py. It respects the train/test splits
produced by those scripts, evaluates each requested layer on that fixed split,
and reports MAE/R2 scores while selecting the best-performing layer.
"""

import argparse
import csv
import os
import pickle
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from modnet.models import MODNetModel
from modnet.preprocessing import MODData
from run_benchmark import parse_task_list

DATA_ROOT = Path(
    os.environ.get("BENCH_DATA_DIR", Path(__file__).resolve().parent / "benchmark_data")
).resolve()
MLIP = os.environ.get("BENCH_MLIP", "orb2")
MODEL = os.environ.get("BENCH_MODEL", "modnet")
TASKS_ENV = os.environ.get("BENCH_TASKS", "")

STRUCTURES_DIR = DATA_ROOT / "structures"
META_DIR = DATA_ROOT / "metadata"
FEAT_DIR = DATA_ROOT / f"feat_{MLIP}"
NPY_DIR = FEAT_DIR / "npy"
RESULTS_DIR = FEAT_DIR / f"results_{MODEL}"
HP_DIR = RESULTS_DIR / "hp"
PARITY_DIR = RESULTS_DIR / "parity"

for directory in [STRUCTURES_DIR, META_DIR, FEAT_DIR, NPY_DIR, RESULTS_DIR, HP_DIR, PARITY_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_KEY = "XPS"
DEFAULT_TARGET_NAME = "g"
DEFAULT_LAYERS = range(1, 16)


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
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    tf.random.set_seed(seed)


def detect_available_layers(feat: Dict, key: str) -> List[int]:
    """Return sorted layers present inside the feature bundle (handles _lN and _N)."""
    with_l_prefix = f"{key}_l"
    without_l_prefix = f"{key}_"
    layers: List[int] = []
    for feat_key in feat.keys():
        suffix: Optional[str] = None
        if feat_key.startswith(with_l_prefix):
            suffix = feat_key[len(with_l_prefix) :]
        elif feat_key.startswith(without_l_prefix):
            suffix = feat_key[len(without_l_prefix) :]
        if suffix is None or not suffix.isdigit():
            continue
        layer_idx = int(suffix)
        if layer_idx not in layers:
            layers.append(layer_idx)
    return sorted(layers)


def get_layer_features(feat: Dict, key: str, layer: int) -> np.ndarray:
    """Extract layer features supporting both XPS_lN and XPS_N naming."""
    candidates = [f"{key}_l{layer}", f"{key}_{layer}"]
    for candidate in candidates:
        if candidate in feat:
            return np.asarray(feat[candidate], dtype=np.float32)
    raise KeyError(
        f"Could not find features for layer {layer}. Tried keys: {', '.join(candidates)}"
    )


def flatten_predictions(pred: np.ndarray) -> np.ndarray:
    """Convert MODNet predictions to a 1-D vector."""
    arr = np.asarray(pred)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return arr.reshape(-1)


def make_moddata(matrix: np.ndarray, targets: np.ndarray, target_name: str) -> MODData:
    """Wrap numpy arrays into MODData with consistent feature naming."""
    df = pd.DataFrame(matrix)
    series = pd.Series(targets)
    moddata = MODData(df_featurized=df, targets=series, target_names=[target_name])
    moddata.optimal_features = list(df.columns)
    return moddata


def train_and_evaluate_layer(
    layer: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_name: str,
    fast_mode: bool,
    log_path: Path,
) -> Tuple[Dict[str, float], MODNetModel]:
    """Train on the provided train split and evaluate on the held-out test split."""
    mod_train = make_moddata(X_train, y_train, target_name)
    mod_test = make_moddata(X_test, y_test, target_name)

    model = MODNetModel(
        targets=[[target_name]],
        weights={target_name: 1.0},
        num_neurons=([256], [256], [256], [256]),
        num_classes={target_name: 0},
    )
    model.fit(mod_train)

    train_pred = flatten_predictions(model.predict(mod_train, remap_out_of_bounds=False))
    test_pred = flatten_predictions(model.predict(mod_test, remap_out_of_bounds=False))
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    train_mae = mean_absolute_error(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    with open(log_path, "a", encoding="utf-8") as log_fp:
        log_fp.write(
            f"Layer: {layer} | Train MAE: {train_mae:.6f} | Train R2: {train_r2:.6f} | "
            f"Test MAE: {test_mae:.6f} | Test R2: {test_r2:.6f}\n"
        )

    return (
        {
            "layer": layer,
            "train_mae": float(train_mae),
            "train_r2": float(train_r2),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2),
        },
        model,
    )


def fit_full_model(
    X: np.ndarray,
    y: np.ndarray,
    target_name: str,
    fast_mode: bool,
) -> MODNetModel:
    """Train a MODNet model on the supplied dataset."""
    mod_full = make_moddata(X, y, target_name)
    model = MODNetModel(
        targets=[[target_name]],
        weights={target_name: 1.0},
        num_neurons=([256], [256], [256], [256]),
        num_classes={target_name: 0},
    )
    model.fit(mod_full)
    return model


def save_model(model: MODNetModel, path: Path) -> None:
    """Persist the trained model if a save method exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save"):
        model.save(str(path))
    else:
        with open(path, "wb") as fp:
            pickle.dump(model, fp)


def append_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    """Append summary rows to a CSV, creating it when needed."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as csv_fp:
        writer = csv.DictWriter(csv_fp, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def format_dataset_label(slugs: List[str]) -> str:
    """Build a safe label for a dataset list suitable for directory names."""
    if not slugs:
        return "unknown"
    clean_slugs: List[str] = []
    for slug in slugs:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug.strip())
        cleaned = cleaned.strip("_") or "dataset"
        clean_slugs.append(cleaned)
    return "__".join(clean_slugs)


def parse_slug_list(spec: Optional[str]) -> List[str]:
    """Parse comma-separated dataset slugs (keeps order, removes duplicates)."""
    if spec is None:
        return []
    entries = []
    seen = set()
    for token in spec.split(","):
        slug = token.strip()
        if not slug or slug in seen:
            continue
        entries.append(slug)
        seen.add(slug)
    return entries


def load_feature_bundle(slug: str, key: str, split: Optional[str]) -> Tuple[Dict, Path]:
    """Load a feature bundle for the specified slug and split."""
    if split:
        filename = f"{slug}_{split}_{key}_{MLIP}.pkl"
    else:
        filename = f"{slug}_{key}_{MLIP}.pkl"
    path = FEAT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Feature bundle not found: {path}")
    with open(path, "rb") as fp:
        bundle = pickle.load(fp)
    return bundle, path


def resolve_target_property(
    feature_bundle: Dict,
    fallback: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """Determine the target property name and property columns."""
    target_property = feature_bundle.get("target_property") or fallback or DEFAULT_TARGET_NAME
    property_columns = feature_bundle.get("property_columns")
    if property_columns is None:
        property_columns = [target_property] if target_property is not None else []
    return str(target_property), list(property_columns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MODNet on supercell feature bundles."
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task slugs. Defaults to BENCH_TASKS or all tasks.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Specific layer to train. If omitted, all available layers are evaluated.",
    )
    parser.add_argument(
        "--min-layer",
        type=int,
        default=min(DEFAULT_LAYERS),
        help="Lower bound when scanning layers (inclusive).",
    )
    parser.add_argument(
        "--max-layer",
        type=int,
        default=max(DEFAULT_LAYERS),
        help="Upper bound when scanning layers (inclusive).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--target-name",
        default=None,
        help=f"Name of the MODNet target (default: '{DEFAULT_TARGET_NAME}').",
    )
    parser.add_argument(
        "--key",
        default=DEFAULT_KEY,
        help="Feature key prefix (default: XPS).",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Split name used for training (default: 'train').",
    )
    parser.add_argument(
        "--test-split",
        default="test",
        help="Split name used for evaluation (default: 'test').",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable MODNet fast mode (early stopping).",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Value assigned to CUDA_VISIBLE_DEVICES before training.",
    )
    parser.add_argument(
        "--train-final",
        action="store_true",
        help="Retrain the best layer on the full dataset after CV.",
    )
    parser.add_argument(
        "--final-model-path",
        type=Path,
        default=None,
        help="Optional path to save the final model when --train-final is used.",
    )
    parser.add_argument(
        "--layer-model-dir",
        type=Path,
        default=None,
        help="Directory where per-layer models will be stored "
             "(default: <results_modnet>/trained_models/layers_<train>2<test>).",
    )
    parser.add_argument(
        "--feature-slugs",
        default=None,
        help="Comma-separated dataset slugs (e.g. 'random_split_dedup_w_min_freq'). "
             "Overrides --tasks when provided.",
    )
    args = parser.parse_args()

    feature_spec = args.feature_slugs or os.environ.get("BENCH_FEATURE_SLUGS")
    explicit_slugs = parse_slug_list(feature_spec)
    if explicit_slugs:
        target_slugs = explicit_slugs
    else:
        task_spec = args.tasks if args.tasks is not None else TASKS_ENV
        target_slugs = parse_task_list(task_spec)

    configure_devices(args.cuda_visible_devices)
    set_seed(args.seed)

    dataset_label = format_dataset_label(target_slugs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"training_dataset_{dataset_label}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Run directory: {run_dir}")

    layer_model_dir = args.layer_model_dir
    default_layer_model_dir = RESULTS_DIR / "trained_models" / f"layers_{args.train_split}2{args.test_split}"
    if layer_model_dir is None or layer_model_dir == default_layer_model_dir:
        layer_model_dir = run_dir / "trained_models" / f"layers_{args.train_split}2{args.test_split}"
    elif not layer_model_dir.is_absolute():
        layer_model_dir = run_dir / layer_model_dir
    layer_model_dir.mkdir(parents=True, exist_ok=True)

    default_final_dir = RESULTS_DIR / "trained_models"
    final_model_output: Optional[Path] = args.final_model_path
    if final_model_output is None or final_model_output == default_final_dir:
        final_model_output = run_dir / "trained_models"
        final_model_output.mkdir(parents=True, exist_ok=True)
    elif not final_model_output.is_absolute():
        final_model_output = run_dir / final_model_output

    key = args.key.upper()
    summary_rows: List[Dict[str, object]] = []

    for slug in target_slugs:
        print(f"[INFO] Processing dataset: {slug}")

        try:
            train_feat, train_feat_path = load_feature_bundle(slug, key, args.train_split)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Train split feature bundle not found for dataset '{slug}' "
                f"(expected split='{args.train_split}'). "
                "Please run featurize_construct_from_supercells.py with split-aware output."
            ) from exc

        try:
            test_feat, test_feat_path = load_feature_bundle(slug, key, args.test_split)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Test split feature bundle not found for dataset '{slug}' "
                f"(expected split='{args.test_split}'). "
                "Please run featurize_construct_from_supercells.py with split-aware output."
            ) from exc

        target_property, property_columns = resolve_target_property(train_feat, args.target_name)
        test_target_property, _ = resolve_target_property(test_feat, target_property)
        if test_target_property != target_property:
            raise ValueError(
                f"Target property mismatch between train ({target_property}) "
                f"and test ({test_target_property}) for dataset '{slug}'."
            )

        print(
            f"[INFO] Target property for '{slug}': {target_property} "
            f"(columns={property_columns})"
        )
        print(f"[INFO] Train bundle: {train_feat_path}")
        print(f"[INFO] Test bundle:  {test_feat_path}")

        if "targets" not in train_feat:
            raise KeyError(
                f"Train feature bundle for '{slug}' does not contain 'targets'. "
                "Ensure featurize_construct_from_supercells.py completed successfully."
            )
        if "targets" not in test_feat:
            raise KeyError(
                f"Test feature bundle for '{slug}' does not contain 'targets'. "
                "Ensure featurize_construct_from_supercells.py completed successfully."
            )

        y_train = np.asarray(train_feat["targets"], dtype=np.float32)
        y_test = np.asarray(test_feat["targets"], dtype=np.float32)
        print(f"[INFO] Train samples: {len(y_train)} | Test samples: {len(y_test)}")

        train_layers = detect_available_layers(train_feat, key)
        test_layers = detect_available_layers(test_feat, key)
        available_layers = sorted(set(train_layers) & set(test_layers))
        if not available_layers:
            raise RuntimeError(
                f"No common layers found between train and test bundles for dataset '{slug}'."
            )

        if args.layer is not None and args.layer not in available_layers:
            raise ValueError(
                f"Requested layer {args.layer} unavailable. "
                f"Available layers (intersection): {available_layers}"
            )

        if args.layer is not None:
            candidate_layers = [args.layer]
        else:
            lower = max(args.min_layer, min(available_layers, default=args.min_layer))
            upper = min(args.max_layer, max(available_layers, default=args.max_layer))
            candidate_layers = [layer for layer in available_layers if lower <= layer <= upper]

        if not candidate_layers:
            raise RuntimeError(
                f"No layers remain after filtering for dataset '{slug}' "
                f"(available={available_layers}, requested min={args.min_layer}, max={args.max_layer})."
            )

        log_filename = (
            f"{slug}_{args.train_split}2{args.test_split}_{key}_{MLIP}.txt"
        )
        log_path = log_dir / log_filename
        if log_path.exists():
            log_path.unlink()
        with open(log_path, "w", encoding="utf-8") as log_fp:
            log_fp.write(
                f"# Dataset: {slug} | train_split={args.train_split} | test_split={args.test_split}\n"
            )
            log_fp.write(
                f"# Target property: {target_property} | property_columns: {property_columns}\n"
            )

        layer_metrics: List[Dict[str, float]] = []
        for layer in candidate_layers:
            print(f"[INFO] Training layer {layer} for {slug}")
            X_train = get_layer_features(train_feat, key, layer)
            X_test = get_layer_features(test_feat, key, layer)
            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError(
                    f"Train feature count mismatch for layer {layer}: "
                    f"{X_train.shape[0]} features vs {y_train.shape[0]} targets."
                )
            if X_test.shape[0] != y_test.shape[0]:
                raise ValueError(
                    f"Test feature count mismatch for layer {layer}: "
                    f"{X_test.shape[0]} features vs {y_test.shape[0]} targets."
                )
            metrics = train_and_evaluate_layer(
                layer=layer,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                target_name=target_property,
                fast_mode=args.fast,
                log_path=log_path,
            )
            metrics_dict, layer_model = metrics
            layer_metrics.append(metrics_dict)

            model_path = (
                layer_model_dir
                / f"{slug}_{args.train_split}2{args.test_split}_{key}_{MLIP}_l{layer}.modnet"
            )
            save_model(layer_model, model_path)
            print(f"[INFO] Saved layer model to {model_path}")

        if not layer_metrics:
            raise RuntimeError(f"No metrics recorded for dataset '{slug}'.")

        best_metrics = min(layer_metrics, key=lambda m: m["test_mae"])
        print(
            f"[RESULT] Dataset {slug} best layer {best_metrics['layer']} | "
            f"Train MAE {best_metrics['train_mae']:.6f} | Train R2 {best_metrics['train_r2']:.6f} | "
            f"Test MAE {best_metrics['test_mae']:.6f} | Test R2 {best_metrics['test_r2']:.6f}"
        )

        summary_row = {
            "dataset": slug,
            "train_split": args.train_split,
            "test_split": args.test_split,
            "target_property": target_property,
            "layer": best_metrics["layer"],
            "train_mae": best_metrics["train_mae"],
            "train_r2": best_metrics["train_r2"],
            "test_mae": best_metrics["test_mae"],
            "test_r2": best_metrics["test_r2"],
        }
        summary_rows.append(summary_row)

        if args.train_final:
            best_layer_idx = best_metrics["layer"]
            X_train_best = get_layer_features(train_feat, key, best_layer_idx)
            X_test_best = get_layer_features(test_feat, key, best_layer_idx)

            print(
                f"[INFO] Training final model on layer {best_layer_idx} for {slug} "
                "(using specified train split)."
            )
            final_model = fit_full_model(
                X=X_train_best,
                y=y_train,
                target_name=target_property,
                fast_mode=args.fast,
            )

            train_md = make_moddata(X_train_best, y_train, target_property)
            test_md = make_moddata(X_test_best, y_test, target_property)
            final_train_pred = flatten_predictions(
                final_model.predict(train_md, remap_out_of_bounds=False)
            )
            final_test_pred = flatten_predictions(
                final_model.predict(test_md, remap_out_of_bounds=False)
            )
            y_train_flat = np.asarray(y_train).reshape(-1)
            y_test_flat = np.asarray(y_test).reshape(-1)
            final_train_mae = mean_absolute_error(y_train_flat, final_train_pred)
            final_test_mae = mean_absolute_error(y_test_flat, final_test_pred)
            final_train_r2 = r2_score(y_train_flat, final_train_pred)
            final_test_r2 = r2_score(y_test_flat, final_test_pred)

            with open(log_path, "a", encoding="utf-8") as log_fp:
                log_fp.write(
                    f"# Final model (layer {best_layer_idx}) | "
                    f"Train MAE: {final_train_mae:.6f} | Train R2: {final_train_r2:.6f} | "
                    f"Test MAE: {final_test_mae:.6f} | Test R2: {final_test_r2:.6f}\n"
                )

            print(
                f"[INFO] Final model metrics — Train MAE {final_train_mae:.6f} | "
                f"Train R2 {final_train_r2:.6f} | Test MAE {final_test_mae:.6f} | "
                f"Test R2 {final_test_r2:.6f}"
            )

            if final_model_output is not None:
                model_path = final_model_output
                if model_path.is_dir():
                    model_path = model_path / f"{slug}_{key}_{MLIP}_l{best_layer_idx}.modnet"
                save_model(final_model, model_path)
                print(f"[INFO] Saved final model to {model_path}")

    if summary_rows:
        run_scores_path = run_dir / "benchmark_scores_supercells.csv"
        append_summary_csv(run_scores_path, summary_rows)

        global_scores_path = RESULTS_DIR / "benchmark_scores_supercells.csv"
        if global_scores_path != run_scores_path:
            append_summary_csv(global_scores_path, summary_rows)


if __name__ == "__main__":
    main()
