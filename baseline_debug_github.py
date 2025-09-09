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
import numpy as np
import sys
sys.path.insert(0, '.')
from baseline_models import *
from baseline_datasets import *
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_argparser():
    # Argument parser
    parser = argparse.ArgumentParser(description="Train a configurable MLP with cosine LR scheduler and W&B logging")
    parser.add_argument("--model_name", type=str, default='mlp', help="Model name")
    parser.add_argument("--data_root", type=str, default='github_baseline_benchmarking', help="data root")
    parser.add_argument("--input_horizon", type=int, default=365, help="Input feature size")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden layer size")
    parser.add_argument("--output_size", type=int, default=1, help="Output size")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the MLP")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--warmup_epochs", type=float, default=0.1, help="Number of warmup epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument('--use_sgd', default=False, action='store_true')
    parser.add_argument('--ablation', default=False, action='store_true')
    parser.add_argument('--ablation_name', default='accesses', type=str)
    parser.add_argument('--extra_tag', default='default', type=str)
    return parser

    
def get_model(args):
    if args.model_name == 'mlp':
        model = MLP(input_size=args.input_size, 
                    hidden_size=args.hidden_size, 
                    output_size=args.output_size, 
                    num_layers=args.num_layers)
    elif args.model_name == 'lstm':
        model = LSTMModel(input_size=args.input_size, 
                          hidden_size=args.hidden_size,
                          output_size=args.output_size, 
                          num_layers=args.num_layers)
    elif args.model_name == 'transformer':
        model = TransformerModel(num_feats=3,
            input_size=args.input_size, 
                                 hidden_size=args.hidden_size,
                                 output_size=args.output_size,
                                 num_layers=args.num_layers)
        
    return model

def get_mae(preds, gts):
    absolute_errors = np.abs(preds - gts)
    return np.mean(absolute_errors), np.std(absolute_errors)

def get_relative_mae(preds, gts, thresh):
    mask = gts >= thresh
    absolute_errors = np.abs(preds - gts)
    relative_errors = absolute_errors[mask] / gts[mask]
    return np.mean(relative_errors), np.std(relative_errors)

def eval(model, loader, split_name, scaler):
    all_preds = []
    all_labels = []
    model.eval()
    for batch_X, _, batch_y_linear in tqdm(loader):
        batch_X = scaler.transform(batch_X)
        batch_X = batch_X.cuda()
        # batch_y = batch_y.cuda()
        with torch.no_grad():
            outputs = model(batch_X).detach().cpu().numpy().flatten()
        all_preds.append(outputs)
        all_labels.append(batch_y_linear.numpy())
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)# + 10.73
    rmse = np.sqrt(mean_squared_error(np.log1p(all_labels), all_preds))
    log_abs_mae, log_abs_mae_std = get_mae(all_preds, np.log1p(all_labels))
    all_preds = (np.exp(all_preds) - 1).round()
    abs_mae, abs_mae_std = get_mae(all_preds, all_labels)
    rel_mae1, rel_mae1_std = get_relative_mae(all_preds, all_labels, 1.)
    rel_mae5, rel_mae5_std = get_relative_mae(all_preds, all_labels, 5.)
    rel_mae50, rel_mae50_std = get_relative_mae(all_preds, all_labels, 50.)
    pred_std = np.std(all_preds)
    pred_max = np.max(all_preds)
    pred_min = np.min(all_preds)
    return {
        f'{split_name}/log_RMSE': rmse, 
        f'{split_name}/log_MAE': log_abs_mae, 
        f'{split_name}/log_MAE_std': log_abs_mae_std,
        f'{split_name}/absolute_MAE': abs_mae, 
        f'{split_name}/absolute_MAE_std': abs_mae_std,
        f'{split_name}/relative_MAE@1': rel_mae1, 
        f'{split_name}/relative_MAE@1_std': rel_mae1_std,
        f'{split_name}/relative_MAE@5': rel_mae5,
        f'{split_name}/relative_MAE@5_std': rel_mae5_std,
        f'{split_name}/relative_MAE@50': rel_mae50,
        f'{split_name}/relative_MAE@50_std': rel_mae50_std,
        f'{split_name}/pred_std': pred_std,
        f'{split_name}/pred_max': pred_max,
        f'{split_name}/pred_min': pred_min,
    }

def get_datasets(input_horizon, root='github_baseline_benchmarking', train_name='train', 
                 val_name = 'val', test_name='test', ablation=False, ablation_name='accesses'):

    train_dataset = GitHubDataset(root, train_name, input_horizon)
    val_dataset = GitHubDataset(root, val_name, input_horizon)
    test_dataset = GitHubDataset(root, test_name, input_horizon)
    return train_dataset, val_dataset, test_dataset 

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)# sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def main(args):
    args.input_size = args.input_horizon * 3
    # import ipdb; ipdb.set_trace()
    train_dataset, val_dataset, test_dataset = get_datasets(args.input_horizon, 
                                                            args.data_root, 
                                                            ablation=args.ablation,
                                                            ablation_name=args.ablation_name)
    train_features = get_train_dataset_features_github(train_dataset, transform=np.log1p)
    scaler = TorchStandardScaler()
    scaler.fit(train_features)
    print(scaler.mean, scaler.std)
    train_loader, val_loader, test_loader = get_dataloaders(train_dataset=train_dataset, 
                                                            val_dataset=val_dataset,
                                                            test_dataset=test_dataset, 
                                                            batch_size=args.batch_size)
    model = get_model(args).cuda()

    # Hyperparameters
    lr = args.lr
    epochs = args.epochs
    warmup_epochs = 1

    exp_tag = f'{args.model_name}_{args.input_horizon}_lr{args.lr}_ep{args.epochs}_{args.extra_tag}'
    # exp_tag = f'test_linear_norm_balanced_gelu_noact_model_{args.model_name}_layer_{args.num_layers}_lr_{args.lr}_ep_{args.epochs}_bs_{args.batch_size}_hs_{args.hidden_size}_horizon_{args.input_horizon}_ablation_{args.ablation}_{args.ablation_name}'
    output_dir = osp.join('github_outputs', exp_tag)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model, loss, optimizer
    criterion = nn.MSELoss()
    if args.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr) #, weight_decay=1e-5)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs

    # Scheduler with linear warmup and cosine decay
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1. + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    wandb.init(project="eslop", name=exp_tag)
    step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_X, batch_y, _ in progress_bar:
            # import ipdb; ipdb.set_trace()
            batch_X = scaler.transform(batch_X)
            batch_X = batch_X.cuda()
            # batch_y = batch_y.T
            # import ipdb; ipdb.set_trace()
            assert len(batch_y.shape) == 2
            batch_y = batch_y.cuda()
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            total_loss += loss.item()
            lr_val = scheduler.get_last_lr()[0]
            # lr_val = args.lr
            progress_bar.set_postfix(loss=loss.item(), lr=lr_val)
            wandb.log({"train/loss": loss.item(), "train/lr": lr_val, "train/step": step})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}")
        wandb.log({"train/avg_epoch_loss": avg_loss, "train/epoch": epoch+1})
        # Save model checkpoint
        checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)

        val_metrics = eval(model, val_loader, 'val', scaler)
        val_metrics.update({"val/epoch": epoch+1})
        wandb.log(val_metrics)
        test_metrics = eval(model, test_loader, 'test', scaler)
        test_metrics.update({'test/epoch': epoch+1})
        wandb.log(test_metrics)

if __name__ == '__main__':
    args = get_argparser().parse_args()
    main(args)