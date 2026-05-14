import torch
import torch.nn.functional as F

# d_lambda Spectral Divergence (how well a pansharpened image preserves the spectral characteristics, from the MS)
# d_s Spatial Divergence (how well a pansharpened image preserves the spatial characteristics, from the PAN)
# QNR = (1 - d_lambda) * (1 - d_s)

def psnr(pred, target, data_range=1.0, eps=1e-8):
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(
        torch.tensor(data_range ** 2, device=pred.device, dtype=pred.dtype)
        / mse.clamp_min(eps)
    )


def _q_index(x, y, eps=1e-8):
    # x, y: [B, 1, H, W] or [B, C, H, W]
    x = x.flatten(start_dim=2)
    y = y.flatten(start_dim=2)

    mean_x = x.mean(dim=2)
    mean_y = y.mean(dim=2)

    var_x = ((x - mean_x[..., None]) ** 2).mean(dim=2)
    var_y = ((y - mean_y[..., None]) ** 2).mean(dim=2)
    cov_xy = ((x - mean_x[..., None]) * (y - mean_y[..., None])).mean(dim=2)

    numerator = 4.0 * cov_xy * mean_x * mean_y
    denominator = (var_x + var_y) * (mean_x ** 2 + mean_y ** 2)

    return numerator / denominator.clamp_min(eps)


def spectral_distortion(hrms_pred, ms, p=1, eps=1e-8):
    # D_lambda: preserve inter-band relationships.
    channels = hrms_pred.shape[1]

    if channels < 2:
        return torch.zeros((), device=hrms_pred.device, dtype=hrms_pred.dtype)

    vals = []
    for i in range(channels):
        for j in range(channels):
            if i == j:
                continue

            q_hr = _q_index(hrms_pred[:, i:i+1], hrms_pred[:, j:j+1], eps=eps)
            q_ms = _q_index(ms[:, i:i+1], ms[:, j:j+1], eps=eps)

            vals.append((q_hr - q_ms).abs().pow(p).mean())

    return torch.stack(vals).mean().pow(1.0 / p).clamp(0.0, 1.0)


def spatial_distortion(hrms_pred, ms, pan, q=1, eps=1e-8):
    # D_s: preserve relationship between each MS band and PAN.
    # This is an approximation: ideally pan_lr should be produced by sensor-aware low-pass degradation.
    pan_lr = F.interpolate(
        pan,
        size=ms.shape[-2:],
        mode="bicubic",
        align_corners=False,
    )

    channels = hrms_pred.shape[1]
    vals = []

    for i in range(channels):
        q_hr = _q_index(hrms_pred[:, i:i+1], pan, eps=eps)
        q_ms = _q_index(ms[:, i:i+1], pan_lr, eps=eps)
        vals.append((q_hr - q_ms).abs().pow(q).mean())

    return torch.stack(vals).mean().pow(1.0 / q).clamp(0.0, 1.0)


def qnr_metrics(hrms_pred, ms, pan, alpha=1.0, beta=1.0):
    d_lambda = spectral_distortion(hrms_pred, ms)
    d_s = spatial_distortion(hrms_pred, ms, pan)

    qnr = ((1.0 - d_lambda).clamp(0.0, 1.0) ** alpha) * \
          ((1.0 - d_s).clamp(0.0, 1.0) ** beta)

    return {
        "qnr": qnr,
        "d_lambda": d_lambda,
        "d_s": d_s,
    }