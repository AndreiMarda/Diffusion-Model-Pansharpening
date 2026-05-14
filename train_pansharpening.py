import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from data_loading import create_test_loader, create_train_loader, create_validation_loader
from metrics import qnr_metrics
from models.conditioning import GatedFusionPyramid, UNetFeatureExtractor
from models.ddpm_unet import ConditionalDDPMUNet
from models.utils import DDPM_Scheduler, set_seed
from training_step import run_training_step, run_validation_step, sample_hrms
from visualization import save_workflow_samples

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
    totals["pred_mean"] += hrms_pred.mean().item()
    totals["pred_std"] += hrms_pred.std().item()

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
        if sample_dir is not None and batch_index == 0:
            save_workflow_samples(outputs, sample_dir, "train")
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

    totals = {key: 0.0 for key in ("recon_l1", "psnr")}
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
        if sample_dir is not None and batch_index == 0:
            save_workflow_samples(outputs, sample_dir, "valid")
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
        "pred_mean": 0.0,
        "pred_std": 0.0,
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

            if sample_dir is not None and batch_index == 0:
                save_workflow_samples(outputs, sample_dir, "test")

            batch_count += 1

    averages = average_metric_totals(totals, batch_count)
    if lms_count > 0:
        averages["lms_l1"] = lms_total / lms_count

    return averages


def print_metrics(prefix, metrics):
    values = " | ".join(f"{key}: {value:.6f}" for key, value in metrics.items())
    print(f"{prefix} | {values}")


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


def main():
    set_seed(42)

    dataset_name = "gf2"
    batch_size = 8
    # ideal num_epochs = 100
    num_epochs = 25
    # ideal num_time_steps = 1000; 100 for debugging/testing
    num_time_steps = 100
    feature_channels = (32, 64, 128)
    learning_rate = 2e-4
    # ideal train_batches = None
    max_train_batches = None
    # ideal valid batches = 4
    max_validation_batches = 2
    # ideal test_batches = None
    max_test_batches = 1
    sampling_num_time_steps = num_time_steps
    sample_dir = "workflow_samples"
    checkpoint_dir = "checkpoints"
    checkpoint_interval = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "num_time_steps": num_time_steps,
        "feature_channels": feature_channels,
        "learning_rate": learning_rate,
        "max_train_batches": max_train_batches,
        "max_validation_batches": max_validation_batches,
        "max_test_batches": max_test_batches,
        "sampling_num_time_steps": sampling_num_time_steps,
    }

    train_loader = create_train_loader(
        training_dir="dataset/training",
        dataset_name=dataset_name,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    validation_loader = create_validation_loader(
        validation_dir="dataset/validation",
        dataset_name=dataset_name,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    test_loader = create_test_loader(
        testing_dir="dataset/testing",
        dataset_name=dataset_name,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # inspect_loader_range(train_loader, "train")
    # inspect_loader_range(validation_loader, "validation")
    # inspect_loader_range(test_loader, "test")
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    sampling_scheduler = scheduler

    batch = next(iter(train_loader))
    image_channels = batch["gt"].shape[1]
    model_parts = build_models(
        image_channels=image_channels,
        feature_channels=feature_channels,
        num_time_steps=num_time_steps,
        device=device,
    )
    spatial_unet, spectral_unet, gated_fusion_pyramid, denoiser = model_parts

    optimizer = optim.AdamW(
        list(spatial_unet.parameters())
        + list(spectral_unet.parameters())
        + list(gated_fusion_pyramid.parameters())
        + list(denoiser.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    training_started_at = time.perf_counter()

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            train_loader=train_loader,
            scheduler=scheduler,
            num_time_steps=num_time_steps,
            device=device,
            model_parts=model_parts,
            optimizer=optimizer,
            max_batches=max_train_batches,
            sample_dir=sample_dir,
        )
        validation_metrics = validate_one_epoch(
            validation_loader=validation_loader,
            scheduler=sampling_scheduler,
            num_time_steps=sampling_num_time_steps,
            device=device,
            model_parts=model_parts,
            max_batches=max_validation_batches,
            sample_dir=sample_dir,
        )

        print_metrics(f"Epoch {epoch + 1}/{num_epochs} train", train_metrics)
        print_metrics(f"Epoch {epoch + 1}/{num_epochs} valid", validation_metrics)

        current_epoch = epoch + 1
        if current_epoch % checkpoint_interval == 0:
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=current_epoch,
                model_parts=model_parts,
                optimizer=optimizer,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                config=config,
            )

    test_metrics = test_one_epoch(
        test_loader=test_loader,
        scheduler=sampling_scheduler,
        num_time_steps=sampling_num_time_steps,
        device=device,
        model_parts=model_parts,
        max_batches=max_test_batches,
        sample_dir=sample_dir,
    )
    print_metrics(f"Testing metrics", test_metrics)

    if device.type == "cuda":
        torch.cuda.synchronize()

    training_duration = time.perf_counter() - training_started_at
    print(f"Training lasted {format_duration(training_duration)}.")


if __name__ == "__main__":
    main()
