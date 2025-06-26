import os
import numpy as np
import argparse
import pickle
import torch
from train import GPT2LightningModule, ArxivPretokenizedDataset
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
import json
from argparse import Namespace
from tqdm import tqdm


def evaluate(model, dataset, device, batch_size=32):
    """Evaluate model on a dataset and return predictions and labels."""
    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True
    )

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with truncated inputs
            outputs = model(
                citation_input_ids=batch["citation_input_ids"],
                access_input_ids=batch["access_input_ids"],
                attention_mask=batch["attention_mask"],
            )

            preds = outputs.logits[:, 1, :6378].argmax(
                dim=-1
            )  # TODO: check how often you need it
            # print(preds[0, :])
            # print(batch["labels"].shape)
            # print(batch["labels"][0, :])
            labels = batch["labels"][:, 1]

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_preds, all_labels


def evaluate_mae(args, preds=None, labels=None, relative=0):
    folder = args.bin_edges_path
    with open(os.path.join(folder, "citation_bins.pkl"), "rb") as f:
        citation_bins = pickle.load(f)
    bin_midpoints = (citation_bins[:-1] + citation_bins[1:]) / 2
    # Create a mapping tensor that uses bin edges for first 5000 bins, midpoints for the rest
    citation_bins = torch.tensor(citation_bins)
    bin_midpoints = torch.tensor(bin_midpoints)
    mapping = torch.zeros_like(bin_midpoints)
    mapping[:5000] = citation_bins[:5000]
    mapping[5000:] = bin_midpoints[5000:]  # Use bin midpoints for the rest
    preds = mapping[preds]
    labels = mapping[labels]

    if relative == 0:
        return torch.nn.functional.l1_loss(preds, labels)

    mask = labels > relative

    # Calculate absolute errors
    abs_errors = torch.abs(preds - labels)

    # For labels > 5, calculate relative errors (abs_error / true_label)
    # Only compute for the masked values
    if torch.any(mask):
        filtered_abs_errors = abs_errors[mask]
        filtered_labels = labels[mask]
        relative_errors = filtered_abs_errors / filtered_labels

        # Return mean of relative errors
        return torch.mean(relative_errors)

    # mask = labels > 5

    # # Calculate absolute errors
    # abs_errors = torch.abs(preds - labels)

    # # For labels > 5, calculate relative errors (abs_error / true_label)
    # # Only compute for the masked values
    # if torch.any(mask):
    #     filtered_abs_errors = abs_errors[mask]
    #     filtered_labels = labels[mask]
    #     relative_errors = filtered_abs_errors / filtered_labels

    #     # Return mean of relative errors
    #     return torch.mean(relative_errors)
    # else:
    #     # If no labels > 5, return 0 or some default value
    #     return torch.tensor(0.0, device=preds.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path to the checkpoint directory",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/share/dean/arxiv-data/tokenized_pred_val/",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--bin_edges_path",
        type=str,
        default="/share/dean/arxiv-data/bin_edges/",
        help="Path to the citation bins",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    args = parser.parse_args()
    config_path = os.path.join(args.checkpoint_dir, "config.json")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    c_args = Namespace(**config_dict)

    model_config = GPT2Config(
        vocab_size=c_args.vocab_size,
        n_positions=c_args.n_positions,
        n_embd=c_args.n_embd,
        n_layer=c_args.n_layer,
        n_head=c_args.n_head,
        embd_pdrop=c_args.embd_pdrop,
        resid_pdrop=c_args.resid_pdrop,
        use_cache=False,
    )

    c_args.learning_rate = 1e-4
    c_args.warmup_steps = 200
    c_args.weight_decay = 0.0
    c_args.train_batch_size = 8
    c_args.eval_batch_size = 8
    c_args.forecast = False

    model = GPT2LightningModule(
        data_dir=args.data_dir,
        learning_rate=c_args.learning_rate,
        warmup_steps=c_args.warmup_steps,
        weight_decay=c_args.weight_decay,
        train_batch_size=c_args.train_batch_size,
        eval_batch_size=c_args.eval_batch_size,
        pretrained_checkpoint=args.checkpoint_dir,
        config=model_config,
        forecast=c_args.forecast,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    dataset = ArxivPretokenizedDataset(
        citation_input_path=os.path.join(args.data_dir, "citation_input_val.txt"),
        access_input_path=os.path.join(args.data_dir, "access_input_val.txt"),
        citation_target_path=os.path.join(args.data_dir, "citation_target_val.txt"),
    )

    # Evaluate model (call the standalone function)
    preds, labels = evaluate(model, dataset, device, batch_size=args.batch_size)
    np.save("preds.npy", preds)
    np.save("labels.npy", labels)

    print(
        f"mean of labels: {labels.mean(dtype=torch.float)} and mean of preds: {preds.mean(dtype=torch.float)}"
    )
    print(f"median of labels: {labels.median()} and median of preds: {preds.median()}")

    mae = evaluate_mae(args, preds=preds, labels=labels)
    rel1_mae = evaluate_mae(args, preds=preds, labels=labels, relative=1)
    rel5_mae = evaluate_mae(args, preds=preds, labels=labels, relative=5)
    print(f"MAE: {mae}")
    print(f"Relative>1 MAE: {rel1_mae}")
    print(f"Relative>5 MAE: {rel5_mae}")

    # TODO: get chunks of size 100 of test set
