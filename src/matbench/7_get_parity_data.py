# script to generate parity data

import os
import random, pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch  
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from modnet.preprocessing import MODData
from modnet.models import MODNetModel
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

mpl.rcParams["figure.dpi"] = 300
plt.rc("font", family="Arial", size=7)

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("PyTorch device:", device)
    print("TensorFlow GPUs:", tf.config.list_logical_devices("GPU"))

    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    tf.random.set_seed(seed)  

    # Read the optimized hyperparameters from the CSV file
    opt_hp_path = Path(HP_DIR) / "benchmark_optuna.csv"
    opted_hp_df = pd.read_csv(opt_hp_path)

    KEY = "XPS"
    key = KEY.lower()

    task_slugs = parse_task_list(TASKS)

    for task in task_slugs[:]:

    # Loop through each row in the CSV to use different hyperparameters
    # for _, row in opted_hp_df.iterrows():
        match = opted_hp_df.loc[opted_hp_df["task"] == task]
        if match.empty:
            raise ValueError(f"No hyperparams found for task: {task}")

        row = match.iloc[0]
        l           = int(row["layer"])
        batch_size  = int(row["batch_size"])
        lr          = float(row["learning_rate"])
        N_features  = int(row["N_features"])
        depth       = int(row["depth"])
        width       = int(row["width"])
        loss        = row["loss"]
        out_act     = row["out_act"]
        hidden = [[width] for _ in range(depth)]      # e.g. depth=3 → [[256],[256],[256]]
        blocks = tuple(hidden) + ([],) * (4 - depth)  # → ([256],[256],[256],[])
        print("hidden:", hidden)
        loss = row['loss']
        out_act = row['out_act']

        output_path = os.path.join(PARITY_DIR, 'benchmark.txt')

        # Load features and targets
        feat = pickle.load(open(os.path.join(FEAT_DIR, f'{task}_{KEY}_{MLIP}.pkl'), 'rb'))
        X_all = feat[f'{KEY}_l{l}']
        y_all = feat['targets']

        maes = []
        r2s = []

        matbench_seed = 18012019
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=matbench_seed)
        task_data = {}

        # Train and evaluate the model for each fold in cross-validation
        for j, (train_idx, test_idx) in enumerate(outer_cv.split(X_all), start=1):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y_all[train_idx], y_all[test_idx]

            cols = list(pd.DataFrame(X_train).columns)[:N_features]

            # Wrap into MODData
            md_tr = MODData(
                df_featurized=pd.DataFrame(X_train[:, :N_features], columns=cols),
                targets=pd.Series(y_train),
                target_names=["g"]
            )
            md_tr.optimal_features = cols

            md_te = MODData(
                df_featurized=pd.DataFrame(X_test[:, :N_features], columns=cols),
                targets=pd.Series(y_test),
                target_names=["g"]
            )
            md_te.optimal_features = cols

            # Build and fit the model
            model = MODNetModel(
                targets=[["g"]],
                weights={"g": 1.0},
                num_neurons=blocks,
                n_feat=N_features,
                num_classes={"g": 0},
                out_act=out_act
            )
            model.fit(
                md_tr,
                batch_size=batch_size,
                lr=lr,
                loss=loss
            )

            # Predict and score
            y_train_pred = model.predict(md_tr, remap_out_of_bounds=False).squeeze()
            y_pred = model.predict(md_te, remap_out_of_bounds=False).squeeze()
            maes.append(mean_absolute_error(y_test, y_pred))
            r2s.append(r2_score(y_test, y_pred))

            # Store fold data in the dictionary with unique keys for each fold
            task_data[f'f{j}_y_train_true'] = y_train.tolist()
            task_data[f'f{j}_y_train_pred'] = y_train_pred.tolist()
            task_data[f'f{j}_y_test_true'] = y_test.tolist()
            task_data[f'f{j}_y_test_pred'] = y_pred.tolist()

            with open(output_path, "a") as out:
                out.write(f"Task: {task} | Layer: {l} | Fold: {j} | MAE: {maes[-1]} | R2: {r2s[-1]}\n")

            # Plot parity
            plt.figure(figsize=(2.4, 2.4))
            plt.scatter(y_train, y_train_pred, alpha=0.3, s=3)
            plt.scatter(y_test, y_pred, alpha=0.3, s=3)
            line_min = min(y_train.min(), y_train_pred.min(), y_test.min(), y_pred.min())
            line_max = max(y_train.max(), y_train_pred.max(), y_test.max(), y_pred.max())
            plt.plot([line_min, line_max], [line_min, line_max], 'k--', lw=1)
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.title(f"Task: {task} | Layer: {l} | Fold: {j} | MAE: {maes[-1]:.5f} | R²: {r2s[-1]:.5f}")
            plt.legend(["Train", "Test"])
            plt.tight_layout()
            
            plot_path = os.path.join(PARITY_DIR, f"parity_{task}_l{l}_f{j}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[INFO] Parity plot saved for task {task}, layer {l}, fold {j}")

            # Save fold-specific data to a pickle file after each fold
            pickle_file = f'{PARITY_DIR}/parity_{task}_l{l}_f{j}.pkl'
            with open(pickle_file, 'wb') as f:
                pickle.dump(task_data, f)

        # Write summary of the results
        mean_mae = np.mean(maes)
        std_mae = np.std(maes)
        mean_r2 = np.mean(r2s)
        std_r2 = np.std(r2s)

        with open(output_path, "a") as out:
            out.write(f"Task: {task} | Layer: {l} | MAE: {mean_mae:.5f} ± {std_mae:.5f} | R2: {mean_r2:.5f} ± {std_r2:.5f}\n")

    print("[INFO] All parity plots and results have been saved.")

if __name__ == '__main__':
    main()
