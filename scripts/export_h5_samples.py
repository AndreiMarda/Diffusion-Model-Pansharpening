from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


root_path = "C:/Users/amardaloescu/Downloads/valid_gf2.h5"
sample_index = 0


def normalize_for_png(image):
    image = np.asarray(image, dtype=np.float32)

    valid = np.isfinite(image)
    if not valid.any():
        return np.zeros_like(image, dtype=np.float32)

    low, high = np.percentile(image[valid], (2, 98))
    if high <= low:
        low, high = image[valid].min(), image[valid].max()

    if high <= low:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip(image, low, high)
    return (image - low) / (high - low)


def channel_first_to_display(sample, rgb_bands=(0, 1, 2)):
    sample = np.asarray(sample)

    if sample.ndim == 2:
        return normalize_for_png(sample)

    if sample.ndim != 3:
        raise ValueError(f"Expected 2D or 3D sample, got shape {sample.shape}")

    # GF2 data is shaped as (channels, height, width).
    if sample.shape[0] <= 16:
        channels_first = sample
    else:
        channels_first = np.moveaxis(sample, -1, 0)

    if channels_first.shape[0] == 1:
        return normalize_for_png(channels_first[0])

    rgb = np.stack([channels_first[band] for band in rgb_bands], axis=-1)
    return normalize_for_png(rgb)


def save_png(image, filename, cmap=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / filename
    plt.imsave(destination, image, cmap=cmap)
    print(f"Saved {destination}")


with h5py.File(root_path, "r") as f:
    print("Keys:", list(f.keys()))

    for dataset_name in ("pan", "ms", "lms", "gt"):
        dataset = f[dataset_name]
        sample = dataset[sample_index]

        print(f"{dataset_name}: shape={dataset.shape}, dtype={dataset.dtype}")

        image = channel_first_to_display(sample)
        cmap = "gray" if image.ndim == 2 else None
        save_png(image, f"{dataset_name}_sample_{sample_index}.png", cmap=cmap)

    lms_sample = f["lms"][sample_index]
    if lms_sample.ndim == 3 and lms_sample.shape[0] >= 4:
        ms_band_4 = lms_sample[3]
    elif lms_sample.ndim == 3 and lms_sample.shape[-1] >= 4:
        ms_band_4 = lms_sample[:, :, 3]
    else:
        raise ValueError(f"Cannot find the 4th band in ms sample shape {lms_sample.shape}")

    save_png(
        normalize_for_png(ms_band_4),
        f"ms_band_4_sample_{sample_index}.png",
        cmap="gray",
    )
