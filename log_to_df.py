import pandas as pd
import wandb


def get_experiment_results(project_name="sweep"):
    """
    Pull all your experiment results from wandb into a nice DataFrame
    """
    api = wandb.Api()
    runs = api.runs(project_name)

    results = []
    for run in runs:
        config = run.config
        summary = run.summary
        row = {
            "run_name": run.name,
            **config,
            **summary,
        }

        # Debug: Print available config keys for first run
        if len(results) == 0:
            print(f"Available config keys for run {run.name}: {list(config.keys())}")
            print(f"Available summary keys for run {run.name}: {list(summary.keys())}")

        results.append(row)

    df = pd.DataFrame(results)

    df.to_csv("experiment_results.csv", index=False)
    return df


if __name__ == "__main__":
    df = get_experiment_results()
    print(df.head(10))
