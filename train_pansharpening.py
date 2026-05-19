import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from data_loading import create_test_loader, create_train_loader, create_validation_loader
from metrics import qnr_metrics, reference_metrics
from models.conditioning import GatedFusionPyramid, UNetFeatureExtractor
from models.ddpm_unet import ConditionalDDPMUNet
from models.utils import DDPM_Scheduler, set_seed
from training_step import run_training_step, run_validation_step, sample_hrms
from visualization import save_workflow_samples

REFERENCE_METRIC_KEYS = ("psnr", "ssim", "scc", "sam", "ergas")
TRAINING_DATASETS = (
    {"name": "gf2", "test_datasets": ("gf2",)},
    {"name": "qb", "test_datasets": ("qb",)},
    {"name": "wv3", "test_datasets": ("wv3", "wv2")},
)

# def inspect_loader_range(loader, name, num_batches=3):
#     print(f"\n{name} range check")
#
#     for batch_idx, batch in enumerate(loader):
#         if batch_idx >= num_batches:
#             break
#
#         print(f"Batch {batch_idx}")
#
#         for key in ["pan", "ms", "lms", "gt"]:
#             if key not in batch:
#                 continue
#
#             x = batch[key].float()
#             print(
#                 f"  {key:>3} | "
#                 f"shape={tuple(x.shape)} | "
#                 f"min={x.min().item():.6f} | "
#                 f"max={x.max().item():.6f} | "
#                 f"mean={x.mean().item():.6f} | "
#                 f"std={x.std().item():.6f}"
#             )

def build_models(image_channels, feature_channels, num_time_steps, device):
    spatial_unet = UNetFeatureExtractor(
        in_channels=1,
        feature_channels=feature_channels,
    ).to(device)
    spectral_unet = UNetFeatureExtractor(
        in_channels=image_channels,
        feature_channels=feature_channels,
    ).to(device)
    gated_fusion_pyramid = GatedFusionPyramid(feature_channels=feature_channels).to(device)
    denoiser = ConditionalDDPMUNet(
        image_channels=image_channels,
        feature_channels=feature_channels,
        num_time_steps=num_time_steps,
    ).to(device)

    return spatial_unet, spectral_unet, gated_fusion_pyramid, denoiser


def model_parts_train(model_parts):
    for model in model_parts:
        model.train()


def model_parts_eval(model_parts):
    for model in model_parts:
        model.eval()


def update_metric_totals(totals, outputs):
    for key in totals:
        totals[key] += outputs[key].item()


def average_metric_totals(totals, count):
    return {key: value / max(count, 1) for key, value in totals.items()}


def update_test_metric_totals(totals, outputs, batch, device):
    hrms_pred = outputs["hrms_pred"]

    lms_value = None
    if "lms" in batch:
        lms = batch["lms"].to(device)
        lms_value = F.l1_loss(hrms_pred, lms).item()

    metrics = qnr_metrics(
        hrms_pred=hrms_pred,
        ms=batch["ms"].to(device),
        pan=batch["pan"].to(device),
    )
    for key, value in metrics.items():
        totals[key] += value.item()

    return lms_value


def train_one_epoch(
    train_loader,
    scheduler,
    num_time_steps,
    device,
    model_parts,
    optimizer,
    max_batches=None,
    sample_dir=None,
):
    spatial_unet, spectral_unet, gated_fusion_pyramid, denoiser = model_parts
    model_parts_train(model_parts)

    totals = {key: 0.0 for key in ("loss", "recon_l1")}
    batch_count = 0

    for batch_index, batch in enumerate(train_loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        outputs = run_training_step(
            batch=batch,
            scheduler=scheduler,
            num_time_steps=num_time_steps,
            device=device,
            spatial_unet=spatial_unet,
            spectral_unet=spectral_unet,
            gated_fusion_pyramid=gated_fusion_pyramid,
            denoiser=denoiser,
            optimizer=optimizer,
        )
        update_metric_totals(totals, outputs)
        if sample_dir is not None:
            save_batch_workflow_samples(outputs, sample_dir, "train", batch_index)
        batch_count += 1

    return average_metric_totals(totals, batch_count)


def validate_one_epoch(
    validation_loader,
    scheduler,
    num_time_steps,
    device,
    model_parts,
    max_batches=None,
    sample_dir=None,
):
    spatial_unet, spectral_unet, gated_fusion_pyramid, denoiser = model_parts
    model_parts_eval(model_parts)

    totals = {key: 0.0 for key in ("recon_l1", *REFERENCE_METRIC_KEYS)}
    batch_count = 0

    for batch_index, batch in enumerate(validation_loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        outputs = run_validation_step(
            batch=batch,
            scheduler=scheduler,
            num_time_steps=num_time_steps,
            device=device,
            spatial_unet=spatial_unet,
            spectral_unet=spectral_unet,
            gated_fusion_pyramid=gated_fusion_pyramid,
            denoiser=denoiser,
        )
        update_metric_totals(totals, outputs)
        if sample_dir is not None:
            save_batch_workflow_samples(outputs, sample_dir, "valid", batch_index)
        batch_count += 1

    return average_metric_totals(totals, batch_count)


def test_one_epoch(
    test_loader,
    scheduler,
    num_time_steps,
    device,
    model_parts,
    max_batches=None,
    sample_dir=None,
):
    spatial_unet, spectral_unet, gated_fusion_pyramid, denoiser = model_parts
    model_parts_eval(model_parts)

    totals = {
        "qnr": 0.0,
        "d_lambda": 0.0,
        "d_s": 0.0,
    }
    lms_total = 0.0
    lms_count = 0
    batch_count = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            outputs = sample_hrms(
                batch=batch,
                scheduler=scheduler,
                num_time_steps=num_time_steps,
                device=device,
                spatial_unet=spatial_unet,
                spectral_unet=spectral_unet,
                gated_fusion_pyramid=gated_fusion_pyramid,
                denoiser=denoiser,
            )

            lms_value = update_test_metric_totals(totals, outputs, batch, device)
            if lms_value is not None:
                lms_total += lms_value
                lms_count += 1

            if sample_dir is not None:
                save_batch_workflow_samples(outputs, sample_dir, "test", batch_index)

            batch_count += 1

    averages = average_metric_totals(totals, batch_count)
    if lms_count > 0:
        averages["lms_l1"] = lms_total / lms_count

    return averages


def print_metrics(prefix, metrics):
    values = " | ".join(f"{key}: {value:.6f}" for key, value in metrics.items())
    print(f"{prefix} | {values}")


def save_batch_workflow_samples(outputs, sample_dir, split_name, batch_index):
    if sample_dir is None:
        return

    batch_size = outputs["pan"].shape[0]
    for sample_index in range(batch_size):
        prefix = f"{split_name}_batch_{batch_index:04d}_sample_{sample_index:02d}"
        save_workflow_samples(
            outputs=outputs,
            output_dir=sample_dir,
            prefix=prefix,
            sample_index=sample_index,
        )


def save_checkpoint(
    checkpoint_dir,
    epoch,
    model_parts,
    optimizer,
    train_metrics,
    validation_metrics,
    config,
):
    spatial_unet, spectral_unet, gated_fusion_pyramid, denoiser = model_parts
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "spatial_unet": spatial_unet.state_dict(),
        "spectral_unet": spectral_unet.state_dict(),
        "gated_fusion_pyramid": gated_fusion_pyramid.state_dict(),
        "denoiser": denoiser.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "config": config,
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def format_duration(seconds):
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


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


def verify_required_datasets(training_datasets, training_dir, validation_dir, testing_dir):
    missing = []
    for experiment in training_datasets:
        dataset_name = experiment["name"]
        required_files = [
            Path(training_dir) / f"train_{dataset_name}.h5",
            Path(validation_dir) / f"valid_{dataset_name}.h5",
        ]

        for test_dataset_name in experiment["test_datasets"]:
            required_files.append(Path(testing_dir) / test_dataset_name / "Full")

        for path in required_files:
            if path.is_dir():
                if not list(path.glob("*.h5")):
                    missing.append(f"{path}/*.h5")
            elif not path.exists():
                missing.append(str(path))

    if missing:
        missing_lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Cannot start the full multi-dataset run because required data is missing:\n"
            f"{missing_lines}"
        )


def run_dataset_experiment(experiment, base_config, device):
    dataset_name = experiment["name"]
    print(f"\nStarting experiment for training dataset: {dataset_name}")

    config = {**base_config, "dataset_name": dataset_name}
    checkpoint_dir = Path(base_config["checkpoint_dir"]) / dataset_name
    sample_root = Path(base_config["sample_dir"]) / dataset_name
    results_dir = Path(base_config["results_dir"]) / dataset_name

    train_loader = create_train_loader(
        training_dir=base_config["training_dir"],
        dataset_name=dataset_name,
        batch_size=base_config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=base_config["num_workers"],
        augment=base_config["train_augmentation"],
    )
    validation_loader = create_validation_loader(
        validation_dir=base_config["validation_dir"],
        dataset_name=dataset_name,
        batch_size=base_config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=base_config["num_workers"],
    )

    scheduler = DDPM_Scheduler(num_time_steps=base_config["num_time_steps"])
    sampling_scheduler = scheduler

    batch = next(iter(train_loader))
    image_channels = batch["gt"].shape[1]
    model_parts = build_models(
        image_channels=image_channels,
        feature_channels=base_config["feature_channels"],
        num_time_steps=base_config["num_time_steps"],
        device=device,
    )
    spatial_unet, spectral_unet, gated_fusion_pyramid, denoiser = model_parts

    optimizer = optim.AdamW(
        list(spatial_unet.parameters())
        + list(spectral_unet.parameters())
        + list(gated_fusion_pyramid.parameters())
        + list(denoiser.parameters()),
        lr=base_config["learning_rate"],
        weight_decay=base_config["weight_decay"],
    )

    rows = []
    training_started_at = time.perf_counter()

    for epoch in range(base_config["num_epochs"]):
        current_epoch = epoch + 1
        checkpoint_epoch = (
            current_epoch % base_config["checkpoint_interval"] == 0
            or current_epoch == base_config["num_epochs"]
        )
        epoch_sample_dir = sample_root / f"epoch_{current_epoch:04d}" if checkpoint_epoch else None

        train_metrics = train_one_epoch(
            train_loader=train_loader,
            scheduler=scheduler,
            num_time_steps=base_config["num_time_steps"],
            device=device,
            model_parts=model_parts,
            optimizer=optimizer,
            max_batches=base_config["max_train_batches"],
            sample_dir=None,
        )
        validation_metrics = validate_one_epoch(
            validation_loader=validation_loader,
            scheduler=sampling_scheduler,
            num_time_steps=base_config["sampling_num_time_steps"],
            device=device,
            model_parts=model_parts,
            max_batches=base_config["max_validation_batches"],
            sample_dir=epoch_sample_dir,
        )

        add_metrics_row(rows, dataset_name, dataset_name, "train", current_epoch, train_metrics)
        add_metrics_row(rows, dataset_name, dataset_name, "valid", current_epoch, validation_metrics)

        print_metrics(f"{dataset_name} epoch {current_epoch}/{base_config['num_epochs']} train", train_metrics)
        print_metrics(f"{dataset_name} epoch {current_epoch}/{base_config['num_epochs']} valid", validation_metrics)

        if checkpoint_epoch:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=current_epoch,
                model_parts=model_parts,
                optimizer=optimizer,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                config=config,
            )

    for test_dataset_name in experiment["test_datasets"]:
        test_loader = create_test_loader(
            testing_dir=base_config["testing_dir"],
            dataset_name=test_dataset_name,
            batch_size=1,
            shuffle=False,
            num_workers=base_config["num_workers"],
        )
        test_sample_dir = sample_root / f"test_{test_dataset_name}"
        test_metrics = test_one_epoch(
            test_loader=test_loader,
            scheduler=sampling_scheduler,
            num_time_steps=base_config["sampling_num_time_steps"],
            device=device,
            model_parts=model_parts,
            max_batches=base_config["max_test_batches"],
            sample_dir=test_sample_dir,
        )
        add_metrics_row(
            rows,
            dataset_name,
            test_dataset_name,
            "test",
            base_config["num_epochs"],
            test_metrics,
        )
        print_metrics(f"{dataset_name} model test on {test_dataset_name}", test_metrics)

    if device.type == "cuda":
        torch.cuda.synchronize()

    training_duration = time.perf_counter() - training_started_at
    print(f"{dataset_name} experiment lasted {format_duration(training_duration)}.")

    metrics_path = results_dir / "metrics.csv"
    write_metrics_table(rows, metrics_path)
    print_results_table(rows, f"{dataset_name} results")
    return rows


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
