# script to find the best layer for each task with fixed hyper-parameters; ML model: MODnet

import os, csv, random, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from modnet.preprocessing import MODData
from modnet.models import MODNetModel
import tensorflow as tf
import torch
from run_benchmark import parse_task_list
from pathlib import Path

DATA_ROOT = Path(os.environ.get("BENCH_DATA_DIR", Path(__file__).resolve().parent / "benchmark_data")).resolve()
MLIP      = os.environ.get("BENCH_MLIP", "orb2")
MODEL     = os.environ.get("BENCH_MODEL", "results_modnet")
TASKS     = os.environ.get("BENCH_TASKS")

STRUCTURES_DIR = DATA_ROOT / "structures"
META_DIR       = DATA_ROOT / "metadata"
FEAT_DIR       = DATA_ROOT / f"feat_{MLIP}"
NPY_DIR        = FEAT_DIR / "npy"
RESULTS_DIR    = DATA_ROOT / MODEL
HP_DIR         = RESULTS_DIR / "hp"
PARITY_DIR     = RESULTS_DIR / "parity"

for p in [STRUCTURES_DIR, META_DIR, FEAT_DIR, NPY_DIR, RESULTS_DIR, HP_DIR, PARITY_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # expose only physical GPU #1
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch device:", device)
    print("TensorFlow GPUs:", tf.config.list_logical_devices("GPU"))

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    KEY   = "XPS"
    key   = KEY.lower()
    # data_dir = f'/home/sokim/ion_conductivity/feat/matbench/{key}2feat_{MLIP}'
    # out_dir = os.path.join(data_dir, 'modnet_results')
    # os.makedirs(out_dir, exist_ok = True)

    task_slugs = parse_task_list(TASKS)
    matbench_seed = 18012019
    kf = KFold(n_splits=5, shuffle=True, random_state=matbench_seed)

    for task in task_slugs[:]:

        feat_path = os.path.join(FEAT_DIR, f'{task}_{KEY}_{MLIP}.pkl')
        feat = pickle.load(open(feat_path, 'rb'))
        output_path = os.path.join(RESULTS_DIR, f'{task}_{KEY}_{MLIP}.txt')
        layer_mae_list = [] 

        for l in range(1, 16):
            X_all = feat[f'{KEY}_l{l}']
            y_all = feat['targets']
            result_mae = []
            result_r2 = []

            for j, (train_idx, test_idx) in enumerate(kf.split(X_all), start=1):
                # train_idx and test_idx are integer arrays of indices
                X_train = pd.DataFrame(X_all[train_idx])
                y_train = pd.Series(y_all[train_idx])
                
                X_test  = pd.DataFrame(X_all[test_idx])
                y_test  = pd.Series(y_all[test_idx])

                moddata = MODData(
                    df_featurized=X_train,
                    targets=y_train,
                    target_names=["g"],
                )

                moddata.optimal_features = list(X_train.columns)

                moddata_test = MODData(
                    df_featurized=X_test,
                    targets=y_test,
                    target_names=["g"],
                )

                moddata_test.optimal_features = list(X_train.columns)

                model = MODNetModel(
                    targets=[["g"]],  # Nested list, 1st block
                    weights={"g": 1.0},  # Equal weight on your single property
                    num_neurons=([256], [256], [256], [256]),  # Your custom network architecture
                    num_classes={"g": 0},  # Regression
                    # act="relu",
                    # out_act="linear",  # Regression output
                )

                # Fit the model
                model.fit(moddata)  
                # `fast=True` uses early stopping, very useful for materials datasets

                # Predict on new structures
                test_y_pred = model.predict(moddata_test, remap_out_of_bounds=False)
                train_y_pred = model.predict(moddata, remap_out_of_bounds=False)

                train_y_pred = train_y_pred.squeeze()  # if 1-column, becomes Series
                test_y_pred = test_y_pred.squeeze()

                mae = mean_absolute_error(y_test, test_y_pred)
                r2 = r2_score(y_test, test_y_pred)
                result_mae.append(mae)
                result_r2.append(r2)

                with open(output_path, "a") as out:
                    out.write(f"Task: {task} | Layer: {l} | Fold: {j} | MAE: {mae} | R2: {r2}\n")

            mae_mean = np.mean(result_mae)
            mae_std = np.std(result_mae)
            r2_mean = np.mean(result_r2)
            r2_std = np.std(result_r2)
            layer_mae_list.append((l, mae_mean, mae_std, r2_mean, r2_std))

        best_layer_info = min(layer_mae_list, key=lambda x: x[1])  # x[1] = mae
        best_layer, best_mae, best_mae_std, best_r2, best_r2_std = best_layer_info

        # Save result
        score_row = {
            "task": task,
            "layer": best_layer,
            "mae_mean": best_mae,
            "mae_std": best_mae_std,
            "r2_mean": best_r2,
            "r2_std": best_r2_std
        }

        score_path = os.path.join(RESULTS_DIR, "benchmark_scores.csv")
        write_header = not os.path.exists(score_path)

        with open(score_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=score_row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(score_row)

if __name__ == "__main__":
    main()