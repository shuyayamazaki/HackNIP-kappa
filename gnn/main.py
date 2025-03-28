import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from orb_models.forcefield import pretrained
from model.model import OrbFrozenMLP
from datamodule.datamodule import BaseMoleculeDataset, collate_fn, MoleculeDatasetSplitter
from train.optim import Frankenstein
from sklearn.metrics import roc_auc_score, accuracy_score, mean_absolute_error


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch using the given dataloader.
    """
    model.train()
    running_loss = 0.0
    total = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        # The structure of `batch` depends on your collate_fn
        # Typically, you'd get something like (graph, labels), but adapt as needed:
        graph, labels = batch
        graph = graph.to(device)
        labels = labels.to(device)

        # Forward pass
        predictions = model(graph)
        loss = model.loss_fn(predictions, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

    epoch_loss = running_loss / total
    return epoch_loss


def validate_one_epoch(model, dataloader, device):
    """
    Validates the model performance on the given dataloader.
    Depending on model.task_type, it will compute:
      - "binary":   AUROC
      - "multiclass": Accuracy
      - "regression": MAE
    Returns: (epoch_loss, epoch_metric)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    # Lists to collect full-batch predictions and labels for metric calculation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            graph, labels = batch
            graph = graph.to(device)
            labels = labels.to(device)

            # Forward pass
            predictions = model(graph)
            loss = model.loss_fn(predictions, labels)

            # Accumulate loss
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            # ---------------------------------------------------
            # Collect data for metric calculation
            # ---------------------------------------------------
            if model.task_type == "binary":
                # Assumes binary classification with single-logit output (BCEWithLogitsLoss).
                # Apply sigmoid to get probabilities in [0, 1].
                probs = torch.sigmoid(predictions).squeeze(dim=-1)
                all_preds.extend(probs.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

            elif model.task_type == "multiclass":
                # Assumes shape (N, num_classes) and CrossEntropyLoss
                predicted_class = torch.argmax(predictions, dim=1)
                all_preds.extend(predicted_class.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

            elif model.task_type == "regression":
                # For regression, collect raw predictions
                # If predictions.shape is (N, 1), flatten it.
                preds = predictions.squeeze(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

            else:
                raise ValueError(f"Unknown task type: {model.task_type}")

    # Compute average loss
    epoch_loss = running_loss / total_samples

    # ---------------------------------------------------
    # Compute the appropriate metric
    # ---------------------------------------------------
    if model.task_type == "binary":
        # AUROC expects labels in {0,1} and predicted probabilities
        # Convert them to numpy arrays or lists
        epoch_metric = roc_auc_score(all_labels, all_preds)  # AUROC

    elif model.task_type == "multiclass":
        # Accuracy for multiclass
        epoch_metric = accuracy_score(all_labels, all_preds) * 100.0  # percentage
        # print(all_labels, all_preds)
        epoch_metric = roc_auc_score(all_labels, all_preds)  # AUROC

    elif model.task_type == "regression":
        # Mean Absolute Error for regression
        epoch_metric = mean_absolute_error(all_labels, all_preds)

    else:
        raise ValueError(f"Unknown task type: {model.task_type}")

    return epoch_loss, epoch_metric

def main():
    # ----------------------------
    # 1. Basic configuration
    # ----------------------------
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    num_epochs = 200
    learning_rate = 1e-3
    batch_size = 200
    data_path = '/home/lucky/Projects/llacha/data/data/ClinTox_dataset.csv'
    task_type = 'multiclass'
    relaxation = False
    random_seed = 2
    split_type = 'random'
    exp_name = 'orbff_clintox_CT_TOX_frank'
    # ----------------------------
    # 2. Prepare data and model
    # ----------------------------
    dataset = BaseMoleculeDataset(
        csv_path=data_path, 
        smiles_col='smiles', 
        property_cols='CT_TOX', 
        relaxation=relaxation,
        task_type=task_type
    )
    splitter = MoleculeDatasetSplitter(
        dataset,
        smiles_list=dataset.smiles_list,
        random_seed=random_seed,
        split_type=split_type
    )
    splitter.random_split(valid_size=0.1, test_size=0.1)
    train_loader = splitter.get_dataloader('train', batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = splitter.get_dataloader('valid', batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = splitter.get_dataloader('test', batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Train size: {len(train_loader.dataset)} | Valid size: {len(val_loader.dataset)} | Test size: {len(test_loader.dataset)}")
    print(len(dataset))
    # Load pretrained orb model
    orbff = pretrained.orb_v2(device=device)

    # Initialize our model
    model = OrbFrozenMLP(
        orb_model=orbff.model,
        hidden_dim=128,
        output_dim=2,         # number of classes (e.g., 10)
        freeze_orb=True,
        task_type=task_type # set to False if doing regression
    ).to(device)

    # ----------------------------
    # 3. Define loss function & optimizer
    # ----------------------------
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    optimizer = Frankenstein(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    # ----------------------------
    # 4. Training and validation loops
    # ----------------------------
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metric = validate_one_epoch(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {val_loss:.4f} | Valid Metric: {val_metric:.4f}")

        # If you want to save the model when validation improves:
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model_{exp_name}.pt")
            print("Best model saved!")

    # ----------------------------
    # 5. Evaluate on the test set
    # ----------------------------
    model.load_state_dict(torch.load(f"best_model_{exp_name}.pt"))
    test_loss, test_metric = validate_one_epoch(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Test Metric: {test_metric:.4f}")

if __name__ == "__main__":
    main()
