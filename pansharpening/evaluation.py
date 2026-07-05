import torch
import torch.nn.functional as F

REFERENCE_METRIC_KEYS = ("psnr", "ssim", "scc", "sam", "ergas")

# d_lambda Spectral Divergence (how well a pansharpened image preserves the spectral characteristics, from the MS)
# d_s Spatial Divergence (how well a pansharpened image preserves the spatial characteristics, from the PAN)
# QNR = (1 - d_lambda) * (1 - d_s)

# reacon_l1 = mean absolute error between the predicted high-resolution multispectral image and the ground truth
# lower is better
# pred_mean is just the average pixel value of the predicted pansharpened output, hrms_pred

# lms_l1 tells how much the model changed the bicubic/upsampled ms input.
#  Too low may mean the model is doing almost nothing; too high may mean it is over-injecting detail or distorting spectra.

# QNR = 1 (ideal)
# D_lambda = 0 (ideal)
# D_s = 0 (ideal)


def psnr(pred, target, data_range=2.0, eps=1e-8):
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(
        torch.tensor(data_range ** 2, device=pred.device, dtype=pred.dtype)
        / mse.clamp_min(eps)
    )


def _check_same_shape(pred, target):
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} does not match target shape {target.shape}")


def _to_reflectance(x, min_value=-1.0, data_range=2.0):
    # The loaders normalize images to [-1, 1]. Spectral metrics expect positive radiance/reflectance.
    return (x - min_value) / data_range


def _pearson_corr(x, y, eps=1e-8):
    x = x.flatten(start_dim=2)
    y = y.flatten(start_dim=2)

    x = x - x.mean(dim=2, keepdim=True)
    y = y - y.mean(dim=2, keepdim=True)

    numerator = (x * y).mean(dim=2)
    denominator = torch.sqrt((x.pow(2).mean(dim=2) * y.pow(2).mean(dim=2)).clamp_min(eps))
    return numerator / denominator


def _high_pass(x):
    channels = x.shape[1]
    kernel = torch.tensor(
        [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.expand(channels, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1, groups=channels)


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, window_size, window_size).expand(channels, 1, window_size, window_size)


def ssim(pred, target, data_range=2.0, window_size=11, sigma=1.5, eps=1e-8):
    _check_same_shape(pred, target)

    height, width = pred.shape[-2:]
    window_size = min(window_size, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    window_size = max(window_size, 1)

    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)
    padding = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_pred_target

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_pred_target + c1) * (2.0 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)

    return (numerator / denominator.clamp_min(eps)).mean()


def scc(pred, target, high_pass=True, eps=1e-8):
    _check_same_shape(pred, target)

    if high_pass:
        pred = _high_pass(pred)
        target = _high_pass(target)

    return _pearson_corr(pred, target, eps=eps).mean()


def sam(pred, target, min_value=-1.0, data_range=2.0, degrees=True, eps=1e-8):
    _check_same_shape(pred, target)

    pred = _to_reflectance(pred, min_value=min_value, data_range=data_range)
    target = _to_reflectance(target, min_value=min_value, data_range=data_range)

    dot = (pred * target).sum(dim=1)
    pred_norm = pred.pow(2).sum(dim=1).sqrt()
    target_norm = target.pow(2).sum(dim=1).sqrt()
    cosine = dot / (pred_norm * target_norm).clamp_min(eps)
    angle = torch.acos(cosine.clamp(-1.0, 1.0))

    if degrees:
        angle = angle * (180.0 / torch.pi)

    return angle.mean()


def ergas(pred, target, scale_ratio=4.0, min_value=-1.0, data_range=2.0, eps=1e-8):
    _check_same_shape(pred, target)

    pred = _to_reflectance(pred, min_value=min_value, data_range=data_range)
    target = _to_reflectance(target, min_value=min_value, data_range=data_range)

    rmse_per_band = (pred - target).pow(2).mean(dim=(0, 2, 3)).sqrt()
    mean_per_band = target.mean(dim=(0, 2, 3)).abs().clamp_min(eps)
    relative_error = rmse_per_band / mean_per_band

    return (100.0 / scale_ratio) * torch.sqrt(relative_error.pow(2).mean())


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


def reference_metrics(pred, target, scale_ratio=4.0, data_range=2.0, min_value=-1.0):
    return {
        "psnr": psnr(pred, target, data_range=data_range),
        "ssim": ssim(pred, target, data_range=data_range),
        "scc": scc(pred, target),
        "sam": sam(pred, target, min_value=min_value, data_range=data_range),
        "ergas": ergas(pred, target, scale_ratio=scale_ratio, min_value=min_value, data_range=data_range),
    }


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
