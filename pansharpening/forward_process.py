import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

IMAGE_PATH   = "C:\\Users\\andre\\PycharmProjects\\Diffusion-Model-Pansharpening\\workflow_samples\\gf2\\epoch_0100\\gf2_epoch0100_run00_xt_t000.png"
OUTPUT_PATH  = "C:\\Users\\andre\\PycharmProjects\\Diffusion-Model-Pansharpening\\pansharpening\\forward_diffusion_steps"

T            = 200
BETA_START   = 1e-4
BETA_END     = 0.02
TIMESTEPS    = [0, 50, 100, 150, 199]

SEED         = 42


np.random.seed(SEED)

betas      = np.linspace(BETA_START, BETA_END, T)
alphas     = 1.0 - betas
alpha_bars = np.cumprod(alphas)


if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] Image not found: '{IMAGE_PATH}'\n"
        f"Place your image in the same folder as this script\n"
        f"and update IMAGE_PATH at the top of the file.\n"
    )

img_pil  = Image.open(IMAGE_PATH).convert("RGB")
img_np   = np.array(img_pil, dtype=np.float32) / 255.0
img_norm = img_np * 2.0 - 1.0

print(f"Image loaded: {IMAGE_PATH}  |  size: {img_pil.size}  |  shape: {img_norm.shape}")

def forward_sample(x0, t, alpha_bars):
    ab = alpha_bars[t]
    eps = np.random.randn(*x0.shape).astype(np.float32)
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps


def to_display(x):
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)

os.makedirs(OUTPUT_PATH, exist_ok=True)          # ← fixed: was OUTPUT_DIR

for t in TIMESTEPS:

    if t == 0:
        x_t = img_norm.copy()
    else:
        x_t = forward_sample(img_norm, t, alpha_bars)

    ab = alpha_bars[t] if t > 0 else 1.0

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.imshow(to_display(x_t))
    ax.axis("off")

    fig.tight_layout(pad=0)

    out_path = os.path.join(OUTPUT_PATH, f"diffusion_t{t:03d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)

    print(f"  Saved: {out_path}  |  ᾱ_t = {ab:.6f}")

print(f"\nDone. All {len(TIMESTEPS)} images saved to: {os.path.abspath(OUTPUT_PATH)}/")