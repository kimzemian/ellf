from datasets import load_from_disk
import pandas as pd
from tqdm import tqdm


def get_one_pd(row):
    df = pd.DataFrame(
        {
            "unique_id": row["id"],
            "ds": list(range(len(row["daily_citations"]))),
            "y": row["daily_citations"],
            "daily_accesses": row["daily_accesses"],
        }
    )
    return {"df": df}


def main():
    path = "/share/dean/arxiv-data/cumulative_cnts_with_metadata_citation_offset_v4/baseline_daily_train"
    train_ds = load_from_disk(path).remove_columns(
        [
            "publication_time",
            "id2",
            "submitted_date",
            "category",
            "cumulative_citations",
            "cumulative_accesses",
            "cumulative_citations_offset",
        ]
    )
    train_ds.save_to_disk("/share/dean/tmp/ds_copy_train")
    train_ds = load_from_disk("/share/dean/tmp/ds_copy_train")
    # train_ds = train_ds.select([0,1,2,3,4])
    # train_ds.set_format('pandas')
    train_ds = train_ds.map(get_one_pd, remove_columns=train_ds.column_names)
    all_dfs = train_ds["df"].tolist()
    all_dfs = [pd.DataFrame.from_dict(df) for df in tqdm(all_dfs)]
    # import ipdb; ipdb.set_trace()
    train_df = pd.concat(all_dfs)

    try:
        train_df.to_csv(
            "/share/dean/arxiv-data/cumulative_cnts_with_metadata_citation_offset_v4/baseline_daily_train.csv",
            index=False,
        )
    except:
        import ipdb

        ipdb.set_trace()


if __name__ == "__main__":
    main()
