from datasets import load_from_disk
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import math
import wandb
import argparse
from torch.utils.data import Sampler, WeightedRandomSampler
from transformers import get_scheduler
import numpy as np
import sys

sys.path.insert(0, ".")
from baseline_models import *
from baseline_datasets import ArxivDataset, ArxivDatasetLinear


def get_balanced_sampler(dataset, num_bins=10):
    # Get all labels from dataset
    labels = np.array(
        [dataset[i][1].item() for i in range(len(dataset))]
    )  # Assuming (x, label, orig_label)

    # Compute histogram bins
    hist, bin_edges = np.histogram(labels, bins=num_bins)

    # Assign each sample to a bin
    bin_indices = np.digitize(labels, bins=bin_edges[:-1], right=True)

    # Inverse frequency weights
    bin_counts = np.bincount(bin_indices, minlength=num_bins + 1)
    weights = 1.0 / (bin_counts[bin_indices] + 1e-6)  # Avoid division by zero

    # Normalize weights
    weights = weights / weights.sum()

    sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(dataset), replacement=True
    )
    return sampler


def get_argparser():
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Train a configurable MLP with cosine LR scheduler and W&B logging"
    )
    parser.add_argument(
        "--model_name", type=str, default="transformer", help="Model name"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/share/dean/arxiv-data/model_dev/baseline_benchmarking",
        help="data root",
    )
    parser.add_argument(
        "--input_horizon", type=int, default=365, help="Input feature size"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="Hidden layer size"
    )
    parser.add_argument("--output_size", type=int, default=1, help="Output size")
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of layers in the MLP"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_epochs", type=float, default=0.1, help="Number of warmup epochs"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--ablation", default=False, action="store_true")
    parser.add_argument("--ablation_name", default="accesses", type=str)
    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--scheduler", default="linear", type=str)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_cycles", default=None)
    parser.add_argument("--power", default=None)
    return parser


def load_input_mean(horizon):
    input_mean = np.load(
        "/share/dean/arxiv-data/model_dev/baseline_benchmarking/inputs_mean.npy"
    )
    citations_mean = input_mean[:365]
    accesses_mean = input_mean[365:]  # TODO: why??
    citations_mean = citations_mean[:horizon]
    accesses_mean = accesses_mean[:horizon]
    combined = np.concatenate([citations_mean, accesses_mean])
    return combined


def get_model(args):
    if args.model_name == "mlp":
        model = MLP(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            num_layers=args.num_layers,
        )
    elif args.model_name == "lstm":
        model = LSTMModel(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            num_layers=args.num_layers,
        )
    elif args.model_name == "transformer":
        model = TransformerModel(
            num_feats=args.input_size,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            num_layers=args.num_layers,
        )

    return model


def get_mae(preds, gts):
    absolute_errors = np.abs(preds - gts)
    return np.mean(absolute_errors), np.std(absolute_errors)


def get_relative_mae(preds, gts, thresh):
    mask = gts >= thresh
    absolute_errors = np.abs(preds - gts)
    relative_errors = absolute_errors[mask] / gts[mask]
    return np.mean(relative_errors), np.std(relative_errors)


def eval(model, loader, split_name):
    all_preds = []
    all_labels = []
    model.eval()
    for batch_X, _, batch_y_linear in tqdm(loader):
        batch_X = batch_X.cuda()
        # batch_y = batch_y.cuda()
        with torch.no_grad():
            outputs = model(batch_X).detach().cpu().numpy().flatten()
        all_preds.append(outputs)
        all_labels.append(batch_y_linear.numpy())
    all_preds = np.concatenate(all_preds) + 10.73
    # all_preds = (np.exp(all_preds) - 1).round()
    all_labels = np.concatenate(all_labels)
    abs_mae, abs_mae_std = get_mae(all_preds, all_labels)
    rel_mae1, rel_mae1_std = get_relative_mae(all_preds, all_labels, 1.0)
    rel_mae5, rel_mae5_std = get_relative_mae(all_preds, all_labels, 5.0)
    return {
        f"{split_name}/absolute_MAE": abs_mae,
        f"{split_name}/absolute_MAE_std": abs_mae_std,
        f"{split_name}/relative_MAE@1": rel_mae1,
        f"{split_name}/relative_MAE@1_std": rel_mae1_std,
        f"{split_name}/relative_MAE@5": rel_mae5,
        f"{split_name}/relative_MAE@5_std": rel_mae5_std,
    }


def get_datasets(
    input_horizon,
    root="baseline_benchmarking",
    train_name="train",
    val_name="val",
    test_name="labeled_test",
    ablation=False,
    ablation_name="accesses",
):
    if ablation:
        train_dataset = ArxivDatasetAblation(
            root, train_name, input_horizon, ablation_name=ablation_name
        )
        val_dataset = ArxivDatasetAblation(
            root, val_name, input_horizon, ablation_name=ablation_name
        )
        test_dataset = ArxivDatasetAblation(
            root, test_name, input_horizon, ablation_name=ablation_name
        )
    else:
        train_dataset = ArxivDatasetLinear(root, train_name, input_horizon)
        val_dataset = ArxivDatasetLinear(root, val_name, input_horizon)
        test_dataset = ArxivDatasetLinear(root, test_name, input_horizon)
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    train_dataset, val_dataset, test_dataset, batch_size, train_sampler
):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )  # sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_optimizer(args, model):
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    return optimizer


def main(args):
    if args.ablation:
        args.input_size = args.input_horizon
    else:
        args.input_size = args.input_horizon * 2
    train_dataset, val_dataset, test_dataset = get_datasets(
        args.input_horizon,
        args.data_root,
        ablation=args.ablation,
        ablation_name=args.ablation_name,
    )
    train_sampler = get_balanced_sampler(train_dataset)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        train_sampler=train_sampler,
    )
    model = get_model(args).cuda()

    # Hyperparameters
    epochs = args.epochs
    warmup_epochs = args.warmup_epochs

    exp_tag = f"sweep_linear_norm_balanced_gelu_noact_model_{args.model_name}_layer_{args.num_layers}_lr_{args.lr}_ep_{args.epochs}_bs_{args.batch_size}_hs_{args.hidden_size}_horizon_{args.input_horizon}_weight_decay_{args.weight_decay}_optimizer_{args.optimizer}_warmup_epochs_{args.warmup_epochs}_scheduler_{args.scheduler}_num_cyles_{args.num_cycles}_power_{args.power}_ablation_{args.ablation}_{args.ablation_name}"
    output_dir = osp.join("outputs", exp_tag)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model, loss, optimizer
    criterion = nn.MSELoss()
    optimizer = get_optimizer(args, model)

    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs

    scheduler_kwargs = dict(
        name=args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        scheduler_specific_kwargs={},  # start empty
    )

    if args.scheduler == "cosine_with_restarts":
        scheduler_kwargs["scheduler_specific_kwargs"]["num_cycles"] = int(
            args.num_cycles
        )
    elif args.scheduler == "polynomial":
        scheduler_kwargs["scheduler_specific_kwargs"]["power"] = float(args.power)

    scheduler = get_scheduler(**scheduler_kwargs)

    # Training loop
    wandb.init(project="sweep", name=exp_tag, config=vars(args))
    step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_X, batch_y, _ in progress_bar:
            batch_X = batch_X.cuda()
            batch_y = batch_y.cuda()
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            total_loss += loss.item()
            lr_val = scheduler.get_last_lr()[0]
            progress_bar.set_postfix(loss=loss.item(), lr=lr_val)
            wandb.log(
                {"train/loss": loss.item(), "train/lr": lr_val, "train/step": step}
            )

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")
        wandb.log({"train/avg_epoch_loss": avg_loss, "train/epoch": epoch + 1})
        # Save model checkpoint
        checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
            },
            checkpoint_path,
        )

        val_metrics = eval(model, val_loader, "val")
        val_metrics.update({"val/epoch": epoch + 1})
        wandb.log(val_metrics)
        test_metrics = eval(model, test_loader, "test")
        test_metrics.update({"test/epoch": epoch + 1})
        wandb.log(test_metrics)

    wandb.finish()


if __name__ == "__main__":
    args = get_argparser().parse_args()
    main(args)
