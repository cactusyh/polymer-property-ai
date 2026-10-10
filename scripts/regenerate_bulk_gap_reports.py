from pathlib import Path

import pandas as pd

import bulk_gap_end_to_end as bulk


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result_path = root / "results/bulk_gap_e2e_fold_results.csv"
    results = pd.read_csv(result_path)
    neural = results.model.isin(["A", "B", "C"])
    results.loc[neural, "regime"] = results.loc[neural, "regime"].astype(str).str.extract(r"^(grouped|random)", expand=False)
    results.to_csv(result_path, index=False)
    summary = bulk.summarize(results)
    paired = bulk.paired_statistics(results)
    summary.to_csv(root / "results/bulk_gap_e2e_summary.csv", index=False)
    data = bulk.load_bulk_data()
    assignments = pd.read_csv(root / "results/bulk_gap_e2e_fold_assignments.csv")
    similarities = pd.read_csv(root / "results/bulk_gap_e2e_similarity.csv")
    fold_rows = []
    for (regime, repeat, fold), group in assignments.groupby(["regime", "repeat", "fold"]):
        test_clusters = group.loc[group.partition == "test", "cluster_id"]
        fold_rows.append({"regime": regime, "repeat": repeat, "fold": fold, "n_train": int((group.partition == "train").sum()), "n_validation": int((group.partition == "validation").sum()), "n_test": int((group.partition == "test").sum()), "n_train_clusters": int(group.loc[group.partition == "train", "cluster_id"].nunique()), "n_validation_clusters": int(group.loc[group.partition == "validation", "cluster_id"].nunique()), "n_test_clusters": int(test_clusters.nunique()), "test_largest_cluster": int(test_clusters.value_counts().max())})
    folds = pd.DataFrame(fold_rows)
    hyper = {"encoder_lr": 2e-5, "head_lr": 3e-4, "weight_decay": 0.01, "dropout": 0.1, "max_epochs": 20, "patience_limit": 5, "num_workers": 0}
    bulk.write_bulk_report(data, folds, similarities, summary, paired, hyper)
    bulk.write_comparison_report(summary, paired, data)
    print(summary[summary.metric == "RMSE"].to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
