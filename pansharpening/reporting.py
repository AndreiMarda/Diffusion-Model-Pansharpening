import csv
from pathlib import Path


def add_metrics_row(rows, train_dataset, eval_dataset, phase, epoch, metrics):
    row = {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "phase": phase,
        "epoch": epoch,
    }
    row.update(metrics)
    rows.append(row)


def write_metrics_table(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in {"train_dataset", "eval_dataset", "phase", "epoch"}
        }
    )
    fieldnames = ["train_dataset", "eval_dataset", "phase", "epoch", *metric_keys]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved metrics table: {output_path}")


def print_results_table(rows, title):
    metric_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in {"train_dataset", "eval_dataset", "phase", "epoch"}
        }
    )
    columns = ["train_dataset", "eval_dataset", "phase", "epoch", *metric_keys]

    print(f"\n{title}")
    print(" | ".join(columns))
    print(" | ".join("---" for _ in columns))
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        print(" | ".join(values))
