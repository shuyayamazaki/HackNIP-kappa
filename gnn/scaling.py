# script to fine-tune orb for direct property prediction
# env: conda env create -f env.yml

import os, sys, pathlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import wandb
from pymatgen.io.ase import AseAtomsAdaptor
import pandas as pd
import numpy as np
import random

from functools import partial
from orb_models.forcefield import pretrained
from model.model import OrbFrozenMLP
from datamodule.datamodule import collate_fn, PickledCrystalDataset, CrystalDatasetSplitter
from train.optim import Frankenstein
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# ─── Training / Validation Loops ───────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, device, global_step):
    model.train()
    running_loss = 0.0
    total = 0
    for graph, labels in tqdm(dataloader, desc="Train", leave=False):
        preds = model(graph).squeeze(-1)
        loss = model.loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        # log per‑step
        wandb.log({"train/loss": loss.item()}, step=global_step)
        global_step += 1
        torch.cuda.empty_cache()

    return running_loss / total, global_step

def validate_one_epoch(model, dataloader, device, global_step):
    model.eval()
    running_loss = 0.0
    total = 0
    all_pred, all_lab = [], []

    with torch.no_grad():
        for graph, labels in tqdm(dataloader, desc="Val", leave=False):
            preds = model(graph).squeeze(-1)
            loss = model.loss_fn(preds, labels)

            bs = labels.size(0)
            running_loss += loss.item() * bs
            total += bs

            all_pred.extend(preds.cpu().tolist())
            all_lab.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_mae  = mean_absolute_error(all_lab, all_pred)

    # log once per epoch
    wandb.log({
        "val/loss": epoch_loss,
        "val/mae":  epoch_mae
    }, step=global_step)
    torch.cuda.empty_cache()

    return epoch_loss, epoch_mae

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Command-line arguments: {' '.join(sys.argv)}")

    # log the full source into the log file
    src_path = pathlib.Path(__file__).resolve()
    print(f"--- Begin source: {src_path.name} ---")
    print(src_path.read_text())
    print(f"--- End source: {src_path.name} ---")

    global_seed = 42
    task_nums = [3,4,5,6,7] #3: shear moduli, 4: bulk moduli, 5: Perovskite formation energy, 6: band gap, 7: formation energy
    targets = [100, 1_000, 10_000]

    results_dir = '/home/sokim/ion_conductivity/gnn/results'
    os.makedirs(results_dir, exist_ok = True)
    txt_path = os.path.join(results_dir, 'scaling_results.txt')

    for task_num in task_nums:    

        for target in targets:

            maes = []

            for seed in range(5): 
                # give each run a distinguishable name
                run_name = f"orbft_t{task_num}_{target}_s{seed}"

                print(f"\n\n=== Starting run: {run_name} ===\n")

                # 1) W&B setup
                wandb.init(
                    project="orb-finetune",
                    name=run_name,
                    config={
                        "epochs": 200,
                        "batch_size": 64,
                        "data_path" : f'/home/sokim/ion_conductivity/feat/matbench/metadata/t{task_num}_data.pkl',
                        "random_seed" : seed,
                        "lr_backbone": 1e-5,
                        "lr_head": 1e-3,
                        "weight_decay": 1e-4,
                        "task": f"t{task_num}"
                    }
                )
                config = wandb.config
                
                torch.manual_seed(global_seed)
                np.random.seed(global_seed)
                random.seed(global_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(global_seed)

                
                orbff = pretrained.orb_v2(device=device)

                # 5) Datasets & loaders
                dataset = PickledCrystalDataset(
                    pickle_path=config.data_path, 
                )

                # reproducible random‐subset
                full_size = len(dataset)
                fraction = min(1.0, target / full_size)
                
                all_indices = np.arange(len(dataset))
                subset_indices, _ = train_test_split(
                    all_indices,
                    train_size=fraction,
                    random_state=config.random_seed,
                    shuffle=True
                )

                dataset = Subset(dataset, subset_indices)

                splitter = CrystalDatasetSplitter(
                dataset,
                random_seed=config.random_seed,
                )

                splitter.random_split(valid_size=0.1, test_size=0.1)

                collate = partial(collate_fn, device=device)
                train_loader = splitter.get_dataloader('train', batch_size=config.batch_size, shuffle=True, collate_fn=collate)
                val_loader = splitter.get_dataloader('valid', batch_size=config.batch_size, shuffle=False, collate_fn=collate)
                test_loader = splitter.get_dataloader('test', batch_size=config.batch_size, shuffle=False, collate_fn=collate)
                print(f"Train size: {len(train_loader.dataset)} | Valid size: {len(val_loader.dataset)} | Test size: {len(test_loader.dataset)}")
                print(len(dataset))

                # 6) Model (unfrozen ORB + MLP head)
                model = OrbFrozenMLP(
                    orb_model=orbff.model,
                    hidden_dim=128,
                    output_dim=1,
                    freeze_orb=False,
                    task_type="regression"
                ).to(device)


                for param in model.orb_model._decoder.parameters():
                    param.requires_grad = False

                for param in model.orb_model._encoder._edge_fn.parameters():
                    param.requires_grad = False

                print("\n Checking which model parameters are trainable:")
                for name, param in model.named_parameters():
                    print(f"{name}: requires_grad = {param.requires_grad}")

                wandb.watch(model, log="all", log_freq=100)

                # 7) Optimizer with discriminative LRs
                optimizer = optim.AdamW([
                    {"params": model.orb_model.parameters(), "lr": config.lr_backbone},
                    {"params": model.mlp.parameters(),  "lr": config.lr_head}
                ], weight_decay=config.weight_decay)

                # 8) Training loop
                best_val_mae = float("inf")
                global_step = 0

                for epoch in range(config.epochs):
                    print(f"Epoch {epoch+1}/{config.epochs}")
                    train_loss, global_step = train_one_epoch(model, train_loader, optimizer, device, global_step)
                    val_loss, val_mae      = validate_one_epoch(model, val_loader, device, global_step)

                    print(f"  Train Loss: {train_loss:.4f}")
                    print(f"  Val   Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

                    # checkpoint
                    if val_mae < best_val_mae:
                        best_val_mae = val_mae
                        ckpt_name = f"best_orb_finetuned_{config.task}_{fraction:.4f}_s{seed}.pt"
                        torch.save(model.state_dict(), ckpt_name)

                        # log artifact
                        art = wandb.Artifact(f"orb-{config.task}-model", type="model")
                        art.add_file(ckpt_name)
                        wandb.log_artifact(art)
                        print("Best model saved!")

                # 9) Final test eval
                model.load_state_dict(torch.load(f"best_orb_finetuned_{config.task}_{fraction:.4f}_s{seed}.pt"))
                test_loss, test_mae = validate_one_epoch(model, test_loader, device, global_step)
                print(f"Test Loss: {test_loss} | Test MAE: {test_mae}")
                maes.append(test_mae)

                with open(txt_path, "a") as f:
                    f.write(f"Task: {config.task} | Fraction: {fraction:.4f} | Seed: {seed} | Test loss: {test_loss} | Test MAE: {test_mae}\n")

                wandb.finish()

            mae_mean = np.mean(maes)
            mae_std = np.std(maes)

            with open(txt_path, "a") as f:
                f.write(f"Fraction: {fraction:.4f} | Test MAE: {mae_mean} ± {mae_std}\n")

if __name__ == "__main__":
    main()
