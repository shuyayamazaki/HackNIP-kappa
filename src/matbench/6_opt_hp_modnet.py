# script to optimize modnet hyper-parameters

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("PyTorch device:", device)
print("TensorFlow GPUs:", tf.config.list_logical_devices("GPU"))

import random, sys, pathlib
import pickle 
import numpy as np                                  
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from modnet.preprocessing import MODData
from modnet.models import MODNetModel

import os, random, pickle
import numpy as np, pandas as pd
import torch
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
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

KEY        = "XPS"
key        = KEY.lower()
L_MIN, L_MAX = 1, 15  # valid layer bounds
n_trials   = 50       # total hyperparameter evaluations

# outer CV splitter
matbench_seed = 18012019
outer_cv = KFold(n_splits=5, shuffle=True, random_state=matbench_seed)

def make_early_stop_callback(patience=10, atol=0.0):
    state = {"best": float("inf"), "stale": 0}
    def _cb(study, trial):
        val = study.best_value
        # improvement?
        if val + atol < state["best"]:
            state["best"] = val
            state["stale"] = 0
        else:
            state["stale"] += 1
            if state["stale"] >= patience:
                print(f"[EarlyStop] No improvement for {state['stale']} trials → stopping.")
                study.stop()
    return _cb

def objective(trial, X_all, y_all):
    # 1) suggest hyperparameters
    batch_size    = trial.suggest_categorical("batch_size",    [16, 32, 128])
    lr            = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    N_features    = trial.suggest_int("N_features", 50, 256) # categorical("N_features",    [50, 100, 200])
    # nested list for MODNet num_neurons
    # e.g. choose depth between 1–3, width between 32–256
    depth         = trial.suggest_int("depth", 1, 4)
    width         = trial.suggest_int("width", 32, 256, log=True)
    hidden = [[width] for _ in range(depth)]      # e.g. depth=3 → [[256],[256],[256]]
    blocks = tuple(hidden) + ([],) * (4 - depth)  # → ([256],[256],[256],[])
    loss = trial.suggest_categorical("loss", ["mae"])
    out_act = trial.suggest_categorical("out_act", ["relu", "linear"])

    maes = []

    # 2) 5-fold outer CV
    for train_idx, test_idx in outer_cv.split(X_all):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        # select top-N features
        cols = list(pd.DataFrame(X_train).columns)[:N_features]

        # wrap into MODData
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

        # build & fit the model
        model = MODNetModel(
            targets=[["g"]],
            weights={"g":1.0},
            num_neurons=blocks,
            n_feat=N_features,
            num_classes={"g":0},
            out_act = out_act
        )
        model.fit(
            md_tr,
            batch_size=batch_size,
            lr=lr,
            loss=loss
            # fast="fast"
        )

        # predict & score
        y_pred = model.predict(md_te, remap_out_of_bounds=False).squeeze()
        maes.append(mean_absolute_error(y_test, y_pred))
    
    mean_mae = np.mean(maes)
    std_mae = np.std(maes)

    trial.set_user_attr("mae_std", float(std_mae))
    # return the mean MAE across folds
    return mean_mae

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
        torch.backends.cudnn.benchmark     = False

    task_slugs = parse_task_list(TASKS)

    for task in task_slugs[:]:
        feat = pickle.load(open(os.path.join(FEAT_DIR, f'{task}_{KEY}_{MLIP}.pkl'),'rb'))
        score_path = Path(RESULTS_DIR) / "benchmark_scores.csv"
        score_df = pd.read_csv(score_path)

        row = score_df.loc[score_df["task"] == task]
        base_layer = int(row.iloc[0]["layer"])

        candidate_layers = list(range(
            max(L_MIN, base_layer - 2),
            min(L_MAX, base_layer + 2) + 1
        ))

    # find layer corresponding to the task in score_csv and test five adjacent layers for instance if layer  4 test 2,3,4,5,6, if layer 1 test 1,2,3
        master_csv = os.path.join(HP_DIR, f"{task}_{KEY}_{MLIP}_optuna.csv")
        first = not os.path.exists(master_csv)

        best_tasks = []
        for l in candidate_layers:

            X_all = feat[f'{KEY}_l{l}']
            y_all = feat['targets']

            cb = make_early_stop_callback(patience=10)
            study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=seed))
            study.optimize(lambda t: objective(t, X_all, y_all),
                        n_trials=n_trials,
                        n_jobs=4,
                        callbacks=[cb],
                        show_progress_bar=True)

            # dump every trial if you like
            study.trials_dataframe().to_csv(
                os.path.join(HP_DIR, f"{task}_l{l}_optuna.csv"), index=False
            )

            best = study.best_trial
            rec = {"task":task, "layer":l,
                "best_mae":best.value, "best_mae_std": best.user_attrs["mae_std"],"trial":best.number, **best.params}

            # append only this layer’s record
            pd.DataFrame([rec]).to_csv(
                master_csv, mode="a", header=first, index=False
            )
            first = False

            best_tasks.append(rec)
            print(f"[task {task} l{l}] best MAE = {rec['best_mae']:.4f}")

        # after all layers, record the best‐layer for this task
        best_overall = min(best_tasks, key=lambda r: r["best_mae"])
        task_csv = os.path.join(HP_DIR, f"benchmark_optuna.csv")
        pd.DataFrame([best_overall]).to_csv(
            task_csv,
            mode="a",
            header=not os.path.exists(task_csv),
            index=False
        )

if __name__ == '__main__':
    main()