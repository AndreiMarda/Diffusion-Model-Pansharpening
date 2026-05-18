# Diffusion Model Pansharpening

This project trains a conditional diffusion model for image pansharpening. It combines high-resolution panchromatic images with lower-resolution multispectral images to produce high-resolution multispectral outputs.

The model uses separate U-Net feature extractors for spatial and spectral inputs, fuses those features with a gated pyramid, and denoises residuals with a conditional DDPM U-Net.

## Project Structure

- `train_pansharpening.py` - main training, validation, testing, checkpointing, and sample export script
- `data_loading.py` - H5 dataset and DataLoader utilities
- `diffusion.py` and `models/utils.py` - diffusion scheduling helpers
- `models/` - conditioning modules and conditional DDPM U-Net
- `training_step.py` - training, validation, and sampling steps
- `metrics.py` - reference and no-reference pansharpening metrics
- `visualization.py` - workflow sample image export

## Data

The expected dataset layout is:

```text
dataset/
  training/train_gf2.h5
  validation/valid_gf2.h5
  testing/gf2/Full/*.h5
```

H5 files should contain `pan`, `ms`, and, for supervised training or validation, `gt`. If available, `lms` is also used for comparison and visualization.

## Running

Install the main Python dependencies:

```bash
pip install torch h5py numpy matplotlib
```

Start training with:

```bash
python train_pansharpening.py
```

By default, the script trains on the GF2 dataset, saves checkpoints in `checkpoints/`, and writes visual samples to `workflow_samples/`.

## Notes

Configuration values such as dataset name, batch size, number of epochs, diffusion steps, and learning rate are currently set inside `train_pansharpening.py`.
