from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def _normalize(image):
    image = image.detach().float().cpu()
    low = torch.quantile(image.flatten(), 0.02)
    high = torch.quantile(image.flatten(), 0.98)

    if high <= low:
        low = image.min()
        high = image.max()

    if high <= low:
        return torch.zeros_like(image)

    return ((image.clamp(low, high) - low) / (high - low)).clamp(0.0, 1.0)


def _to_display_image(tensor):
    image = tensor.detach().float().cpu()

    if image.ndim == 4:
        image = image[0]

    if image.ndim == 3 and image.shape[0] == 1:
        return _normalize(image[0]), "gray"

    if image.ndim == 3 and image.shape[0] >= 3:
        rgb = torch.stack([image[0], image[1], image[2]], dim=-1)
        return _normalize(rgb), None

    if image.ndim == 3:
        return _normalize(image[0]), "gray"

    return _normalize(image), "gray"


def save_tensor_png(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    image, cmap = _to_display_image(tensor)
    plt.imsave(path, image.numpy(), cmap=cmap)


def save_workflow_samples(outputs, output_dir, prefix):
    output_dir = Path(output_dir)

    keys = [
        "pan",
        "ms",
        "lms",
        "ms_up",
        "gt",
        "residual",
        "noise",
        "x_t",
        "eps_pred",
        "residual_pred",
        "hrms_pred",
    ]

    for key in keys:
        if key not in outputs or outputs[key] is None:
            continue

        value = outputs[key]
        if key == "ms":
            target_size = outputs.get("pan", value).shape[-2:]
            value = F.interpolate(value, size=target_size, mode="nearest")

        save_tensor_png(value, output_dir / f"{prefix}_{key}.png")
