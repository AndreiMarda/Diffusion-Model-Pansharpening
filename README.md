# Diffusion Model Pansharpening

This project trains a conditional diffusion model for image pansharpening. It combines high-resolution panchromatic images with lower-resolution multispectral images to produce high-resolution multispectral outputs.

The model uses separate U-Net feature extractors for spatial and spectral inputs, fuses those features with a gated pyramid, and denoises residuals with a conditional DDPM U-Net.

## Project Structure

- `train_pansharpening.py` - entry point: builds the config, runs `verify_required_datasets`, then per dataset either trains (`run_dataset_experiment`) or, if a final-epoch checkpoint already exists, runs inference (`run_inference`) instead
- `pansharpening/` - the library package
  - `data.py` - H5 dataset (`PanCollectionDataset`) and DataLoader factory functions
  - `diffusion.py` - forward/reverse diffusion math (`q_sample`, `p_sample`, `reverse_diffusion_sample`, `interp23`, ...)
  - `scheduler.py` - `DDPM_Scheduler` (beta/alpha schedules) and `set_seed`
  - `models/` - network definitions
    - `__init__.py` - `build_models()` factory wiring up all four model parts
    - `conditioning.py` - `UNetFeatureExtractor` and `GatedFusionPyramid`
    - `denoiser.py` - `ConditionalDDPMUNet` (the conditional DDPM U-Net)
  - `training_steps.py` - single-batch training/validation/sampling steps (`run_training_step`, `run_validation_step`, `sample_hrms`)
  - `training_loop.py` - per-epoch loops, checkpoint save/load helpers, and `run_dataset_experiment`
  - `inference.py` - checkpoint-based inference (`run_inference`), including diffusion snapshot capture (t=200,150,100,50,0)
  - `evaluation.py` - reference metrics (PSNR/SSIM/SCC/SAM/ERGAS) and no-reference QNR metrics
  - `reporting.py` - metrics CSV/table writers
  - `visualization.py` - workflow sample and diffusion-snapshot image export
- `scripts/export_h5_samples.py` - standalone helper to export sample PNGs from an H5 file

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
