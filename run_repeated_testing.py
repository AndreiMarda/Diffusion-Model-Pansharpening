import csv
import time
from pathlib import Path

import torch

from pansharpening.data import create_test_loader
from pansharpening.inference import checkpoint_path_for_epoch, load_models_from_checkpoint
from pansharpening.scheduler import DDPM_Scheduler, set_seed
from pansharpening.training_loop import format_duration, test_one_epoch_repeated

# Each entry: a checkpoint trained on `checkpoint_dataset`, evaluated on every
# dataset in `test_datasets` (wv2 is cross-dataset generalization for the wv3 model).
TEST_EXPERIMENTS = (
    {"checkpoint_dataset": "gf2", "test_datasets": ("gf2",)},
    {"checkpoint_dataset": "qb", "test_datasets": ("qb",)},
    {"checkpoint_dataset": "wv3", "test_datasets": ("wv3", "wv2")},
)

NUM_REPEATS = 5
FINAL_EPOCH = 100
CHECKPOINT_DIR = "checkpoints"
TESTING_DIR = "dataset/testing"
RESULTS_PATH = Path("results") / "all_results_5x20.csv"

CSV_FIELDNAMES = [
    "train_dataset",
    "eval_dataset",
    "image_index",
    "run_index",
    "qnr",
    "d_lambda",
    "d_s",
    "lms_l1",
    "generation_seconds",
]


def write_rows_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved {len(rows)} rows to {output_path}")


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows = []
    started_at = time.perf_counter()

    for experiment in TEST_EXPERIMENTS:
        checkpoint_dataset = experiment["checkpoint_dataset"]
        checkpoint_path = checkpoint_path_for_epoch(CHECKPOINT_DIR, checkpoint_dataset, FINAL_EPOCH)
        model_parts, config = load_models_from_checkpoint(checkpoint_path, device)
        num_time_steps = config["num_time_steps"]
        scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

        for test_dataset_name in experiment["test_datasets"]:
            test_loader = create_test_loader(
                testing_dir=TESTING_DIR,
                dataset_name=test_dataset_name,
                batch_size=1,
                shuffle=False,
            )

            print(
                f"Testing checkpoint '{checkpoint_dataset}' on '{test_dataset_name}' "
                f"({len(test_loader)} images x {NUM_REPEATS} runs)..."
            )

            rows = test_one_epoch_repeated(
                test_loader=test_loader,
                scheduler=scheduler,
                num_time_steps=num_time_steps,
                device=device,
                model_parts=model_parts,
                num_repeats=NUM_REPEATS,
            )

            for row in rows:
                row["train_dataset"] = checkpoint_dataset
                row["eval_dataset"] = test_dataset_name
                all_rows.append(row)

    write_rows_csv(all_rows, RESULTS_PATH)

    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"Total duration: {format_duration(time.perf_counter() - started_at)}")


if __name__ == "__main__":
    main()
