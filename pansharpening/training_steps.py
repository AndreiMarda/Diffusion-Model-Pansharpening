import torch
import torch.nn.functional as F

from pansharpening.diffusion import (
    interp23,
    predict_x0_from_eps,
    q_sample,
    reverse_diffusion_sample,
    sample_timesteps,
)
from pansharpening.evaluation import reference_metrics


def prepare_forward_diffusion_batch(batch, scheduler, num_time_steps, device):
    pan = batch["pan"].to(device)
    ms = batch["ms"].to(device)
    gt = batch["gt"].to(device)

    #Use dataset LMS when available, otherwise upsample MS to PAN/GT resolution.
    if "lms" in batch:
        ms_up = batch["lms"].to(device)
    else:
        ms_up = interp23(ms, ratio=pan.shape[-1] // ms.shape[-1])

    if ms_up.shape != gt.shape:
        raise ValueError(f"ms_up shape {ms_up.shape} does not match gt shape {gt.shape}")

    #Compute residual/difference target.
    residual = gt - ms_up

    #Sample timestep and Gaussian noise.
    t = sample_timesteps(batch_size=pan.shape[0], num_time_steps=num_time_steps, device=device)
    noise = torch.randn_like(residual)

    #Forward diffusion.
    x_t = q_sample(residual, t, noise, scheduler)

    return {
        "pan": pan,
        "ms": ms,
        "gt": gt,
        "ms_up": ms_up,
        "residual": residual,
        "t": t,
        "noise": noise,
        "x_t": x_t,
    }


def run_training_step(
    batch,
    scheduler,
    num_time_steps,
    device,
    spatial_unet,
    spectral_unet,
    gated_fusion_pyramid,
    denoiser,
    optimizer,
):
    diffusion_batch = prepare_forward_diffusion_batch(
        batch=batch,
        scheduler=scheduler,
        num_time_steps=num_time_steps,
        device=device,
    )

    outputs = compute_conditioned_losses(
        diffusion_batch=diffusion_batch,
        scheduler=scheduler,
        spatial_unet=spatial_unet,
        spectral_unet=spectral_unet,
        gated_fusion_pyramid=gated_fusion_pyramid,
        denoiser=denoiser,
    )

    optimizer.zero_grad()
    outputs["loss"].backward()
    optimizer.step()

    return detach_training_outputs(outputs)


def compute_conditioned_losses(
    diffusion_batch,
    scheduler,
    spatial_unet,
    spectral_unet,
    gated_fusion_pyramid,
    denoiser,
):
    pan = diffusion_batch["pan"]
    gt = diffusion_batch["gt"]
    ms_up = diffusion_batch["ms_up"]
    x_t = diffusion_batch["x_t"]
    t = diffusion_batch["t"]
    noise = diffusion_batch["noise"]

    # extracting the features
    spatial_feats = spatial_unet(pan)
    spectral_feats = spectral_unet(ms_up)

    # Gated fusion at all scales
    cond = gated_fusion_pyramid(spatial_feats, spectral_feats)

    # Denoiser
    eps_pred = denoiser(x_t, t, cond)

    # Main diffusion loss.
    loss = F.mse_loss(eps_pred, noise)

    # Reconstruct HRMS for a simple validation metric.
    residual_pred = predict_x0_from_eps(x_t, t, eps_pred, scheduler)
    hrms_pred = ms_up + residual_pred
    recon_l1 = F.l1_loss(hrms_pred, gt)

    return {
        **diffusion_batch,
        "spatial_feats": spatial_feats,
        "spectral_feats": spectral_feats,
        "cond": cond,
        "eps_pred": eps_pred,
        "residual_pred": residual_pred,
        "hrms_pred": hrms_pred,
        "loss": loss,
        "recon_l1": recon_l1
    }


def detach_training_outputs(outputs):
    detached = {}
    for key, value in outputs.items():
        if torch.is_tensor(value):
            detached[key] = value.detach()
        elif isinstance(value, list):
            detached[key] = [
                item.detach() if torch.is_tensor(item) else item
                for item in value
            ]
        else:
            detached[key] = value
    return detached


@torch.no_grad()
def run_validation_step(
    batch,
    scheduler,
    num_time_steps,
    device,
    spatial_unet,
    spectral_unet,
    gated_fusion_pyramid,
    denoiser,
):
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

    gt = batch["gt"].to(device)
    outputs["gt"] = gt
    outputs["recon_l1"] = F.l1_loss(outputs["hrms_pred"], gt)
    outputs.update(reference_metrics(outputs["hrms_pred"], gt))

    return detach_training_outputs(outputs)


def sample_hrms(
    batch,
    scheduler,
    num_time_steps,
    device,
    spatial_unet,
    spectral_unet,
    gated_fusion_pyramid,
    denoiser,
):
    pan = batch["pan"].to(device)
    ms = batch["ms"].to(device)

    # Reverse Phase 1: Use LMS when provided, otherwise upsample MS to PAN resolution.
    if "lms" in batch:
        ms_up = batch["lms"].to(device)
    else:
        ms_up = interp23(ms, ratio=pan.shape[-1] // ms.shape[-1])

    # Reverse Phase 2: Extract spatial and spectral conditioning features.
    spatial_feats = spatial_unet(pan)
    spectral_feats = spectral_unet(ms_up)

    # Reverse Phase 3: Fuse conditioning features at all U-Net scales.
    cond = gated_fusion_pyramid(spatial_feats, spectral_feats)

    # Reverse Phase 4: DDPM reverse process from Gaussian noise to residual.
    residual_pred = reverse_diffusion_sample(
        shape=ms_up.shape,
        denoiser=denoiser,
        cond=cond,
        scheduler=scheduler,
        num_time_steps=num_time_steps,
        device=device,
    )

    # Reverse Phase 5: Add sampled high-frequency residual to upsampled MS.
    hrms_pred = ms_up + residual_pred

    return {
        "pan": pan,
        "ms": ms,
        "lms": ms_up,
        "ms_up": ms_up,
        "spatial_feats": spatial_feats,
        "spectral_feats": spectral_feats,
        "cond": cond,
        "residual_pred": residual_pred,
        "hrms_pred": hrms_pred,
    }
