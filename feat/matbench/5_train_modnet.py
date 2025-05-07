# script to find the best layer for each task with fixed hyper-parameters; NN: modnet
# env: python 3.9, pip install modnet, compatible with matbench

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # expose only physical GPU #1

import tensorflow as tf
# now the only GPU TF sees is "/GPU:0"
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("PyTorch device:", device)
print("TensorFlow GPUs:", tf.config.list_logical_devices("GPU"))

import random
import pickle 
import numpy as np                                  
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from modnet.preprocessing import MODData
from modnet.models import MODNetModel

print(f"Command-line arguments: {' '.join(sys.argv)}")

# log the full source into the log file
src_path = pathlib.Path(__file__).resolve()
print(f"--- Begin source: {src_path.name} ---")
print(src_path.read_text())
print(f"--- End source: {src_path.name} ---")

mlip = "orb2"
data_dir = f'/home/sokim/ion_conductivity/feat/matbench/xps2feat_{mlip}'
out_dir = os.path.join(data_dir, 'results')
os.makedirs(out_dir, exist_ok = True)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # disables autotuning, useful for reproducibility

for i in [7]:

    feat_path = os.path.join(data_dir, f't{i}_XPS_{mlip}.pkl')
    feat = pickle.load(open(feat_path, 'rb'))
    output_path = os.path.join(out_dir, f't{i}_XPS_{mlip}.txt')
    layer_mae_list = [] 

    for l in range(10, 16):
        X_all = feat[f'XPS_l{l}']
        y_all = feat['targets']
        result_mae = []
        result_r2 = []

        for j in range(1,6):
            mask = np.array(feat[f"fold{j}"], dtype=bool)  # boolean mask shape (n_structures,)

            X_train = pd.DataFrame(X_all[mask])         # only those structures in the train‐mask
            y_train = pd.Series(y_all[mask])

            X_test  = pd.DataFrame(X_all[~mask])       # the complement
            y_test  = pd.Series(y_all[~mask])

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
                out.write(f"Task: {i} | Layer: {l} | Fold: {j} | MAE: {mae} | R2: {r2}\n")

            # Plot parity
            plt.figure(figsize=(6, 6))
            plt.scatter(y_train, train_y_pred,  alpha=0.5, label="Train")
            plt.scatter(y_test, test_y_pred,  alpha=0.5, label="Test")
            line_min = min(y_train.min(), y_test.min())
            line_max = max(y_train.max(), y_test.max())
            plt.plot([line_min, line_max], [line_min, line_max], 'k--', lw=1)
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.legend()
            plt.title(f"{i} — MAE: {mae:.3f}, R²: {r2:.3f}")
            # plt.tight_layout()
            # plt.gca().set_aspect('equal')

            plot_path = f"{out_dir}/parity_t{i}_l{l}_f{j}.png"
            plt.savefig(plot_path)
            plt.close()

            print(f"[INFO] t{i}_l{l}_f{j} Parity plot saved")

        mae_mean = np.mean(result_mae)
        mae_std = np.std(result_mae)
        r2_mean = np.mean(result_r2)
        r2_std = np.std(result_r2)
        layer_mae_list.append((l, mae_mean, mae_std, r2_mean, r2_std))

    best_layer_info = min(layer_mae_list, key=lambda x: x[1])  # x[1] = mae
    best_layer, best_mae, best_mae_std, best_r2, best_r2_std = best_layer_info

    # Save result
    score_row = {
        "task": i,
        "layer": best_layer,
        "mae_mean": best_mae,
        "mae_std": best_mae_std,
        "r2_mean": best_r2,
        "r2_std": best_r2_std
    }

    score_path = os.path.join(out_dir, "benchmark_scores.csv")
    write_header = not os.path.exists(score_path)

    with open(score_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=score_row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(score_row)
