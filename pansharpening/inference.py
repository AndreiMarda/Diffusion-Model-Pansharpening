import time
from pathlib import Path

import torch

from pansharpening.data import create_test_loader
from pansharpening.diffusion import interp23, reverse_diffusion_sample
from pansharpening.evaluation import qnr_metrics
from pansharpening.models import build_models
from pansharpening.scheduler import DDPM_Scheduler
from pansharpening.training_loop import format_duration
from pansharpening.visualization import save_snapshot_grid, save_tensor_png

DEFAULT_SNAPSHOT_STEPS = (200, 150, 100, 50, 30, 20, 10, 0)
REPEATED_RUN_METRIC_KEYS = ("qnr", "d_lambda", "d_s", "generation_seconds")


def checkpoint_path_for_epoch(checkpoint_dir, dataset_name, epoch):
    return Path(checkpoint_dir) / dataset_name / f"checkpoint_epoch_{epoch:04d}.pth"


def clamp_snapshot_steps(snapshot_steps, num_time_steps):
    # snapshot_steps may reference T itself (e.g. 200 when num_time_steps=200),
    # but valid indices only go up to num_time_steps - 1.
    clamped = {max(0, min(step, num_time_steps - 1)) for step in snapshot_steps}
    return sorted(clamped, reverse=True)


def load_models_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    feature_channels = tuple(config["feature_channels"])
    num_time_steps = config["num_time_steps"]
    image_channels = checkpoint["denoiser"]["input_conv.weight"].shape[1]

    model_parts = build_models(
        image_channels=image_channels,
        feature_channels=feature_channels,
        num_time_steps=num_time_steps,
        device=device,
    )
    spatial_unet, spectral_unet, gated_fusion_pyramid, denoiser = model_parts
    spatial_unet.load_state_dict(checkpoint["spatial_unet"])
    spectral_unet.load_state_dict(checkpoint["spectral_unet"])
    gated_fusion_pyramid.load_state_dict(checkpoint["gated_fusion_pyramid"])
    denoiser.load_state_dict(checkpoint["denoiser"])

    for model in model_parts:
        model.eval()

    return model_parts, config


@torch.no_grad()
def _load_inference_context(checkpoint_dataset_name, test_dataset_name, epoch, checkpoint_dir, testing_dir, device):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = checkpoint_path_for_epoch(checkpoint_dir, checkpoint_dataset_name, epoch)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    model_parts, config = load_models_from_checkpoint(checkpoint_path, device)
    spatial_unet, spectral_unet, gated_fusion_pyramid, denoiser = model_parts
    num_time_steps = config["num_time_steps"]
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

    test_loader = create_test_loader(
        testing_dir=testing_dir,
        dataset_name=test_dataset_name,
        batch_size=1,
        shuffle=False,
    )
    batch = next(iter(test_loader))

    pan = batch["pan"].to(device)
    ms = batch["ms"].to(device)
    if "lms" in batch:
        ms_up = batch["lms"].to(device)
    else:
        ms_up = interp23(ms, ratio=pan.shape[-1] // ms.shape[-1])

    spatial_feats = spatial_unet(pan)
    spectral_feats = spectral_unet(ms_up)
    cond = gated_fusion_pyramid(spatial_feats, spectral_feats)

    return {
        "device": device,
        "denoiser": denoiser,
        "scheduler": scheduler,
        "num_time_steps": num_time_steps,
        "pan": pan,
        "ms": ms,
        "ms_up": ms_up,
        "cond": cond,
    }


@torch.no_grad()
def _generate_once(context, snapshot_steps=None):
    device = context["device"]

    if device.type == "cuda":
        torch.cuda.synchronize()
    generation_started_at = time.perf_counter()

    result = reverse_diffusion_sample(
        shape=context["ms_up"].shape,
        denoiser=context["denoiser"],
        cond=context["cond"],
        scheduler=context["scheduler"],
        num_time_steps=context["num_time_steps"],
        device=device,
        snapshot_steps=snapshot_steps,
    )
    residual_pred, snapshots = result if snapshot_steps is not None else (result, {})

    if device.type == "cuda":
        torch.cuda.synchronize()
    generation_duration = time.perf_counter() - generation_started_at

    hrms_pred = context["ms_up"] + residual_pred
    metrics = qnr_metrics(hrms_pred=hrms_pred, ms=context["ms"], pan=context["pan"])
    metrics = {key: value.item() for key, value in metrics.items()}
    metrics["generation_seconds"] = generation_duration

    return hrms_pred, snapshots, metrics


@torch.no_grad()
def run_inference(
    dataset_name,
    epoch,
    checkpoint_dir="checkpoints",
    testing_dir="dataset/testing",
    sample_dir="inference_samples",
    snapshot_steps=DEFAULT_SNAPSHOT_STEPS,
    sample_index_in_batch=0,
    device=None,
    num_runs=1,
    test_dataset_name=None,
):
    test_dataset_name = test_dataset_name or dataset_name
    context = _load_inference_context(dataset_name, test_dataset_name, epoch, checkpoint_dir, testing_dir, device)
    steps = clamp_snapshot_steps(snapshot_steps, context["num_time_steps"])
    pan, ms_up = context["pan"], context["ms_up"]

    output_dir = Path(sample_dir) / dataset_name / f"epoch_{epoch:04d}"
    run_index_width = max(2, len(str(num_runs - 1)))

    run_results = []
    for run_index in range(num_runs):
        hrms_pred, snapshots, metrics = _generate_once(context, snapshot_steps=steps)

        prefix = f"{dataset_name}_epoch{epoch:04d}"
        if test_dataset_name != dataset_name:
            prefix += f"_test{test_dataset_name}"
        if num_runs > 1:
            prefix += f"_run{run_index:0{run_index_width}d}"

        for step, snapshot in snapshots.items():
            save_tensor_png(
                snapshot,
                output_dir / f"{prefix}_xt_t{step:03d}.png",
                sample_index=sample_index_in_batch,
            )

        save_snapshot_grid(
            snapshots,
            output_dir / f"{prefix}_xt_grid.png",
            sample_index=sample_index_in_batch,
        )

        save_tensor_png(pan, output_dir / f"{prefix}_pan.png", sample_index=sample_index_in_batch)
        save_tensor_png(ms_up, output_dir / f"{prefix}_ms_up.png", sample_index=sample_index_in_batch)
        save_tensor_png(hrms_pred, output_dir / f"{prefix}_hrms_pred.png", sample_index=sample_index_in_batch)

        print(
            f"[{dataset_name} epoch {epoch}] run {run_index + 1}/{num_runs}: "
            f"generated output in {format_duration(metrics['generation_seconds'])} "
            f"({context['num_time_steps']} diffusion steps)."
        )
        print(
            f"QNR: {metrics['qnr']:.6f} | D_lambda: {metrics['d_lambda']:.6f} | "
            f"D_s: {metrics['d_s']:.6f}"
        )

        run_results.append({
            "hrms_pred": hrms_pred,
            "snapshots": snapshots,
            **metrics,
        })

    print(f"Saved inference snapshots to {output_dir}")

    if num_runs == 1:
        return run_results[0]
    return run_results
