import time
from pathlib import Path

import torch

from pansharpening.inference import checkpoint_path_for_epoch, run_inference
from pansharpening.reporting import add_metrics_row, print_results_table, write_metrics_table
from pansharpening.scheduler import set_seed
from pansharpening.training_loop import (
    format_duration,
    run_dataset_experiment,
    verify_required_datasets,
)

TRAINING_DATASETS = (
    # {"name": "gf2", "test_datasets": ("gf2",)},
    # {"name": "qb", "test_datasets": ("qb",)},
    {"name": "wv3", "test_datasets": ("wv3", "wv2")},
    # {"name": "wv3", "test_datasets": ("wv2")},
)


def main():
    set_seed(42)

    base_config = {
        "training_dir": "dataset/training",
        "validation_dir": "dataset/validation",
        "testing_dir": "dataset/testing",
        "batch_size": 8,
        "num_workers": 0,
        "num_epochs": 100,
        "num_time_steps": 200,
        "sampling_num_time_steps": 200,
        "feature_channels": (32, 64, 128),
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "max_train_batches": None,
        "max_validation_batches": None,
        "max_test_batches": None,
        "sample_dir": "workflow_samples",
        "checkpoint_dir": "checkpoints",
        "results_dir": "results",
        "checkpoint_interval": 20,
        "train_augmentation": True,
        "inference_num_runs": 10,
        "cross_dataset_inference_num_runs": 10,
    }

    verify_required_datasets(
        training_datasets=TRAINING_DATASETS,
        training_dir=base_config["training_dir"],
        validation_dir=base_config["validation_dir"],
        testing_dir=base_config["testing_dir"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []
    full_run_started_at = time.perf_counter()

    for experiment in TRAINING_DATASETS:
        dataset_name = experiment["name"]
        final_checkpoint = checkpoint_path_for_epoch(
            base_config["checkpoint_dir"], dataset_name, base_config["num_epochs"]
        )

        if final_checkpoint.exists():
            print(f"Found existing checkpoint for {dataset_name} at {final_checkpoint}; running inference instead of training.")
            for test_dataset_name in experiment["test_datasets"]:
                is_primary_test_set = test_dataset_name == dataset_name
                num_runs = (
                    base_config["inference_num_runs"]
                    if is_primary_test_set
                    else base_config["cross_dataset_inference_num_runs"]
                )

                inference_result = run_inference(
                    dataset_name=dataset_name,
                    test_dataset_name=test_dataset_name,
                    epoch=base_config["num_epochs"],
                    checkpoint_dir=base_config["checkpoint_dir"],
                    testing_dir=base_config["testing_dir"],
                    sample_dir=base_config["sample_dir"],
                    device=device,
                    num_runs=num_runs,
                )
                inference_runs = inference_result if isinstance(inference_result, list) else [inference_result]
                for run_index, run_result in enumerate(inference_runs):
                    inference_metrics = {
                        key: value
                        for key, value in run_result.items()
                        if key not in ("hrms_pred", "snapshots")
                    }
                    phase = "inference" if len(inference_runs) == 1 else f"inference_run{run_index:02d}"
                    add_metrics_row(
                        all_rows,
                        dataset_name,
                        test_dataset_name,
                        phase,
                        base_config["num_epochs"],
                        inference_metrics,
                    )
        else:
            print(f"No checkpoint found for {dataset_name}; starting training from scratch.")
            rows = run_dataset_experiment(experiment, base_config, device)
            all_rows.extend(rows)

    write_metrics_table(all_rows, Path(base_config["results_dir"]) / "all_metrics.csv")
    print_results_table(all_rows, "All experiment results")

    if device.type == "cuda":
        torch.cuda.synchronize()

    full_run_duration = time.perf_counter() - full_run_started_at
    print(f"Full multi-dataset run lasted {format_duration(full_run_duration)}.")


if __name__ == "__main__":
    main()
