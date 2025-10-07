# sciprt to optimize modnet hyper-parameters
# env: python 3.9, pip install modnet, compatible with matbench

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # expose only physical GPU #1

import tensorflow as tf
# now the only GPU TF sees is "/GPU:0"
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
from modnet.matbench.benchmark import matbench_benchmark

import os, random, pickle
import numpy as np, pandas as pd
import torch
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from modnet.preprocessing import MODData
from modnet.models import MODNetModel

# ── settings ───────────────────────────────────────────────────
KEY        = "XPS"
key        = KEY.lower()
mlip       = "orb2"
tasks      = [1]
layers     = [4]
data_dir   = f'/home/sokim/ion_conductivity/feat/matbench/{key}2feat_{mlip}'
out_dir = os.path.join(data_dir, 'results_hp')
os.makedirs(out_dir, exist_ok=True)

seed       = 42
n_trials   = 50            # total hyperparameter evaluations

# reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

print(f"Command-line arguments: {' '.join(sys.argv)}")

# log the full source into the log file
src_path = pathlib.Path(__file__).resolve()
print(f"--- Begin source: {src_path.name} ---")
print(src_path.read_text())
print(f"--- End source: {src_path.name} ---")

# outer CV splitter
matbench_seed = 18012019
outer_cv = KFold(n_splits=5, shuffle=True, random_state=matbench_seed)

def stop_when_no_improve(study, trial):
    global no_improve, best_mae
    if study.best_value < best_mae:
        best_mae = study.best_value
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= 10:
        print(f"No improvement in {no_improve} trials; stopping study.")
        study.stop()

# ── objective for Optuna ──────────────────────────────────────
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
    out_act = trial.suggest_categorical("out_act", ["relu"])

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

# ── run the study ───────────────────────────────────────────────

for i in tasks:
    feat = pickle.load(open(os.path.join(data_dir, f't{i}_{KEY}_{mlip}.pkl'),'rb'))
    master_csv = os.path.join(out_dir, f"t{i}_{KEY}_{mlip}_optuna.csv")
    first = not os.path.exists(master_csv)

    best_tasks = []
    for l in layers:

        no_improve = 0
        best_mae  = float("inf")
        X_all = feat[f'{KEY}_l{l}']
        y_all = feat['targets']

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(lambda t: objective(t, X_all, y_all),
                       n_trials=n_trials,
                       n_jobs=4,
                       callbacks=[stop_when_no_improve],
                       show_progress_bar=True)

        # dump every trial if you like
        study.trials_dataframe().to_csv(
            os.path.join(out_dir, f"t{i}_l{l}_optuna.csv"), index=False
        )

        best = study.best_trial
        rec = {"task":i, "layer":l,
               "best_mae":best.value, "best_mae_std": best.user_attrs["mae_std"],"trial":best.number, **best.params}

        # append only this layer’s record
        pd.DataFrame([rec]).to_csv(
            master_csv, mode="a", header=first, index=False
        )
        first = False

        best_tasks.append(rec)
        print(f"[task {i} l{l}] best MAE = {rec['best_mae']:.4f}")

    # after all layers, record the best‐layer for this task
    best_overall = min(best_tasks, key=lambda r: r["best_mae"])
    task_csv = os.path.join(out_dir, f"benchmark_optuna.csv")
    pd.DataFrame([best_overall]).to_csv(
        task_csv,
        mode="a",
        header=not os.path.exists(task_csv),
        index=False
    )