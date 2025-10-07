import os, sys, pathlib
import random, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from modnet.preprocessing import MODData
from modnet.models import MODNetModel

# Set CUDA visibility and device configurations
os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # Expose only physical GPU #1
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("PyTorch device:", device)
print("TensorFlow GPUs:", tf.config.list_logical_devices("GPU"))

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
tf.random.set_seed(seed)  # Set TensorFlow's random seed for deterministic results

print(f"Command-line arguments: {' '.join(sys.argv)}")

# log the full source into the log file
src_path = pathlib.Path(__file__).resolve()
print(f"--- Begin source: {src_path.name} ---")
print(src_path.read_text())
print(f"--- End source: {src_path.name} ---")

# Read the optimized hyperparameters from the CSV file
opted_hp_df = pd.read_csv('/home/sokim/ion_conductivity/matbench/modnet/opted_hp.csv')

# Directories
KEY = "XPS"
key = KEY.lower()
mlip = "orb2"
data_dir = f'/home/sokim/ion_conductivity/matbench/{key}2feat_{mlip}'
out_dir = '/home/sokim/ion_conductivity/matbench/modnet/results'
os.makedirs(out_dir, exist_ok=True)

# Loop through each row in the CSV to use different hyperparameters
for _, row in opted_hp_df.iterrows():
    i = row['task']
    l = row['layer']
    
    # Extract the optimized hyperparameters
    batch_size = row['batch_size']
    lr = row['learning_rate']
    N_features = row['N_features']
    depth = row['depth']
    width = row['width']
    hidden = [[width] for _ in range(depth)]      # e.g. depth=3 → [[256],[256],[256]]
    blocks = tuple(hidden) + ([],) * (4 - depth)  # → ([256],[256],[256],[])
    print("hidden:", hidden)
    loss = row['loss']
    out_act = row['out_act']

    output_path = os.path.join(out_dir, 'benchmark.txt')

    # Load features and targets
    feat = pickle.load(open(os.path.join(data_dir, f't{i}_{KEY}_{mlip}.pkl'), 'rb'))
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
            out.write(f"Task: {i} | Layer: {l} | Fold: {j} | MAE: {maes[-1]} | R2: {r2s[-1]}\n")

        # Plot parity
        plt.figure(figsize=(2.4, 2.4))
        plt.scatter(y_train, y_train_pred, alpha=0.3, s=3)
        plt.scatter(y_test, y_pred, alpha=0.3, s=3)
        line_min = min(y_train.min(), y_train_pred.min(), y_test.min(), y_pred.min())
        line_max = max(y_train.max(), y_train_pred.max(), y_test.max(), y_pred.max())
        plt.plot([line_min, line_max], [line_min, line_max], 'k--', lw=1)
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"Task: {i} | Layer: {l} | Fold: {j} | MAE: {maes[-1]:.5f} | R²: {r2s[-1]:.5f}")
        plt.legend(["Train", "Test"])

        mpl.rcParams['figure.dpi'] = 300
        plt.rc('font', family='Arial', size=7)
        plt.tight_layout()
        
        plot_path = os.path.join(out_dir, f"parity_t{i}_l{l}_f{j}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Parity plot saved for task {i}, layer {l}, fold {j}")

        # Save fold-specific data to a pickle file after each fold
        pickle_file = f'{out_dir}/parity_t{i}_l{l}_f{j}.pkl'
        with open(pickle_file, 'wb') as f:
            pickle.dump(task_data, f)

    # Write summary of the results
    mean_mae = np.mean(maes)
    std_mae = np.std(maes)
    mean_r2 = np.mean(r2s)
    std_r2 = np.std(r2s)

    with open(output_path, "a") as out:
        out.write(f"Task: {i} | Layer: {l} | MAE: {mean_mae:.5f} ± {std_mae:.5f} | R2: {mean_r2:.5f} ± {std_r2:.5f}\n")

print("[INFO] All parity plots and results have been saved.")
