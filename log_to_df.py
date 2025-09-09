import pandas as pd
import wandb


def get_experiment_results(project_name="eslop"):
    """
    Pull all your experiment results from wandb into a nice DataFrame
    """
    api = wandb.Api()
    runs = api.runs(project_name)

    results = []
    for run in runs:
        # Extract hyperparameters from config
        config = run.config

        # Extract final metrics from summary
        summary = run.summary

        # Combine into one row
        row = {
            "run_name": run.name,
            # All hyperparameters from argparse
            "model_name": config.get("model_name", "unknown"),
            "data_root": config.get("data_root", None),
            "input_horizon": config.get("input_horizon", None),
            "input_size": config.get("input_size", None),  # Added this
            "hidden_size": config.get("hidden_size", None),
            "output_size": config.get("output_size", None),
            "num_layers": config.get("num_layers", None),
            "lr": config.get("lr", None),
            "epochs": config.get("epochs", None),
            "warmup_epochs": config.get("warmup_epochs", None),
            "batch_size": config.get("batch_size", None),
            "ablation": config.get("ablation", False),
            "ablation_name": config.get("ablation_name", None),
            "use_adaptive_pooling": config.get("use_adaptive_pooling", True),
            # Validation metrics
            "val_mae": summary.get("val/absolute_MAE", None),
            "val_mae_std": summary.get("val/absolute_MAE_std", None),
            "val_rel_mae_1": summary.get("val/relative_MAE@1", None),
            "val_rel_mae_1_std": summary.get("val/relative_MAE@1_std", None),
            "val_rel_mae_5": summary.get("val/relative_MAE@5", None),
            "val_rel_mae_5_std": summary.get("val/relative_MAE@5_std", None),
            # Test metrics
            "test_mae": summary.get("test/absolute_MAE", None),
            "test_mae_std": summary.get("test/absolute_MAE_std", None),
            "test_rel_mae_1": summary.get("test/relative_MAE@1", None),
            "test_rel_mae_1_std": summary.get("test/relative_MAE@1_std", None),
            "test_rel_mae_5": summary.get("test/relative_MAE@5", None),
            "test_rel_mae_5_std": summary.get("test/relative_MAE@5_std", None),
            # Training metrics
            "final_train_loss": summary.get("train/avg_epoch_loss", None),
            "final_lr": summary.get("train/lr", None),
            # Run metadata
            "runtime": run.summary.get("_runtime", None),
            "state": run.state,
            "created_at": run.created_at,
        }

        # Debug: Print available config keys for first run
        if len(results) == 0:
            print(f"Available config keys for run {run.name}: {list(config.keys())}")
            print(f"Available summary keys for run {run.name}: {list(summary.keys())}")

        results.append(row)

    df = pd.DataFrame(results)

    # Sort by test performance
    df = df.sort_values("test_mae")

    # save df
    df.to_csv("experiment_results.csv", index=False)
    return df
