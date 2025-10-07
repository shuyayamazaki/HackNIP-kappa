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

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    print(f"Command-line arguments: {' '.join(sys.argv)}")

    # log the full source into the log file
    src_path = pathlib.Path(__file__).resolve()
    print(f"--- Begin source: {src_path.name} ---")
    print(src_path.read_text())
    print(f"--- End source: {src_path.name} ---")

    run_name = f"orbft_t6_f0.1_s42"

    # 1) W&B setup
    wandb.init(
        project="orb-finetune",
        name = run_name,
        config={
            "epochs": 200,
            "batch_size": 64,
            "data_path" : '/home/sokim/ion_conductivity/feat/matbench/metadata/t6_data.pkl', # "/home/sokim/ion_conductivity/matbench/preprocessed_data/xps2feat_orb2/3DSC_MP_XPS_orb2_logfiltered_xpsatoms.pkl", # '/home/sokim/ion_conductivity/feat/matbench/metadata/t6_data.pkl',
            "random_seed" : 42,
            "lr_backbone": 1e-5,
            "lr_head": 1e-3,
            "weight_decay": 1e-4,
            "task": "t6_re"
        }
    )
    config = wandb.config

    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    
    orbff = pretrained.orb_v2(device=device)

    # 5) Datasets & loaders
    dataset = PickledCrystalDataset(
        pickle_path=config.data_path, 
    )

    fraction = 0.1
<<<<<<< Updated upstream
    # # reproducible random‐subset
=======
    # p = 1
    # reproducible random‐subset
>>>>>>> Stashed changes
    # full_size = len(dataset)
    # target    = 10_000

    # # automatically choose fraction to get ~10k samples
    # fraction = min(1.0, target / full_size)
<<<<<<< Updated upstream
    # print(fraction)
=======
    print(fraction)
>>>>>>> Stashed changes
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

    # if p == 1:
    #     p += 1

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
            ckpt_name = f"best_orb_finetuned_{config.task}_{fraction:.4f}.pt"
            torch.save(model.state_dict(), ckpt_name)

            # log artifact
            art = wandb.Artifact(f"orb-{config.task}-model", type="model")
            art.add_file(ckpt_name)
            wandb.log_artifact(art)
            print("Best model saved!")

    # 9) Final test eval
<<<<<<< Updated upstream
    model.load_state_dict(torch.load(f"best_orb_finetuned_{config.task}_{fraction:.3f}.pt"))
=======
    model.load_state_dict(torch.load(f"best_orb_finetuned_{config.task}_{fraction:.4f}.pt"))
>>>>>>> Stashed changes
    test_loss, test_mae = validate_one_epoch(model, test_loader, device, global_step)
    print(f"Test Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()
