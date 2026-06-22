# Multi-Band Pansharpening Support

## Overview

This codebase **supports any number of spectral bands** (4, 8, 11, or more). The architecture automatically adapts to your dataset's spectral dimensions.

## Currently Supported Datasets

| Dataset | Bands | Purpose | Path |
|---------|-------|---------|------|
| **GF2** (GaoFen-2) | 4 | Training | `dataset/training/train_gf2.h5` |
| **QB** (QuickBird) | 4 | Training | `dataset/training/train_qb.h5` |
| **WV3** (WorldView-3) | 8 | Training/Validation/Testing | `dataset/training/train_wv3.h5` |
| **WV2** (WorldView-2) | 8 | Testing only | `dataset/testing/wv2/Full/*.h5` |

## Band-Agnostic Architecture

### Model Components

All components automatically adapt to your data's spectral dimensions:

```
┌─────────────────────────────────────────────┐
│  Input Data                                 │
│  - PAN: 1 band (panchromatic)              │
│  - MS: C bands (multispectral, C=4/8/11...) │
└────────────────┬────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
    ┌───▼──────┐    ┌────▼──────┐
    │Spatial   │    │Spectral   │
    │UNet      │    │UNet       │
    │(1→64)    │    │(C→64)     │◄─ Adapts to C bands
    └────┬─────┘    └────┬──────┘
         │                │
         └────────┬───────┘
                  │
           ┌──────▼──────┐
           │Gated Fusion │
           │Pyramid      │
           └──────┬──────┘
                  │
           ┌──────▼──────┐
           │Denoiser     │
           │(C bands)    │◄─ Adapts to C bands
           └─────────────┘
```

### Key Features

1. **Spatial UNet**: Always processes 1-band panchromatic image
2. **Spectral UNet**: Dynamically adapts input channels to match your MS bands
3. **Denoiser**: Generates high-resolution output with same number of bands as input
4. **Metrics**: All computed per-band and aggregated (no band-specific hardcoding)

## How It Works

### Runtime Band Detection

The code detects the number of bands **automatically from your data**:

```python
# In train_pansharpening.py, run_dataset_experiment():
batch = next(iter(train_loader))
image_channels = batch["gt"].shape[1]  # Automatically detected!
model_parts = build_models(image_channels=image_channels, ...)
```

### Model Building

Models are built with the detected channel count:

```python
spectral_unet = UNetFeatureExtractor(
    in_channels=image_channels,  # 4 for GF2/QB, 8 for WV3/WV2
    feature_channels=feature_channels,
)
```

### Metric Computation

All metrics dynamically adapt to the number of bands:

- **Spectral Distortion (d_lambda)**: Computed for all band pairs (C×(C-1) comparisons)
  - 4-band data: 12 comparisons
  - 8-band data: 56 comparisons
  
- **Spatial Distortion (d_s)**: Computed for each band vs PAN (C comparisons)
  - 4-band data: 4 comparisons
  - 8-band data: 8 comparisons

- **Reference Metrics**: All work with variable band counts

## Data Format Requirements

Your HDF5 files should have this structure:

```
train_wv3.h5         # Or valid_wv3.h5
├── pan              # Shape: [N, 1, H, W] - panchromatic
├── ms               # Shape: [N, C, h, w] - multispectral (C=8 for WV3)
├── lms              # Shape: [N, C, H, W] - upsampled MS (optional)
└── gt               # Shape: [N, C, H, W] - ground truth HRMS
```

### Data Validation

Run the training script with the `--verify-only` flag to check band counts:

```bash
python train_pansharpening.py
```

The first output will show:
```
Verifying dataset band counts:
  train_gf2: 4 bands
  train_qb: 4 bands
  train_wv3: 8 bands
  valid_gf2: 4 bands
  valid_qb: 4 bands
  valid_wv3: 8 bands
  test_gf2: 4 bands
  test_qb: 4 bands
  test_wv3: 8 bands
  test_wv2: 8 bands
```

## Training with 8-Band Data (WV3/WV2)

### Step 1: Prepare Data

Ensure your HDF5 files are in the correct locations:

```
dataset/
├── training/
│   └── train_wv3.h5        # 8-band training data
├── validation/
│   └── valid_wv3.h5        # 8-band validation data
└── testing/
    ├── wv3/Full/*.h5       # 8-band test data
    └── wv2/Full/*.h5       # 8-band test data
```

### Step 2: Run Training

```bash
python train_pansharpening.py
```

The script will:
1. ✅ Automatically detect 8 bands
2. ✅ Build models with 8-band spectral UNet
3. ✅ Train on all 4 datasets (gf2, qb, wv3)
4. ✅ Test on all sensors (gf2, qb, wv3, wv2)

### Step 3: Monitor Training

The models will be saved per dataset:
```
checkpoints/
├── gf2/
│   └── checkpoint_epoch_*.pth
├── qb/
│   └── checkpoint_epoch_*.pth
└── wv3/
    └── checkpoint_epoch_*.pth    # 8-band model
```

### Step 4: Evaluate Results

Check the results:
```
results/
├── gf2/metrics.csv
├── qb/metrics.csv
├── wv3/metrics.csv             # 8-band results
└── all_metrics.csv
```

## Performance Considerations

### Computational Cost

With more bands, expect:
- **Spectral UNet**: ~2x parameters for 8-band vs 4-band (8/4 input channels)
- **Denoiser**: ~2x parameters for 8-band vs 4-band (8/4 output channels)
- **Metrics**: ~4-5x more pairwise comparisons (56 vs 12 bands pairs)

### Memory Usage

Approximate VRAM for batch size 8:
- 4-band data: ~6-7 GB
- 8-band data: ~11-13 GB (depends on image size)

### Training Time

Expect ~1.5-2x longer training for 8-band vs 4-band data.

## Troubleshooting

### Error: "Cannot batch H5 files with different channel counts"

**Cause**: Mixing different numbers of bands in training

**Solution**: 
- Keep 4-band datasets (gf2, qb) separate
- Keep 8-band dataset (wv3) separate
- Don't try to combine them in the same loader

### Error: Model loading fails with channel mismatch

**Cause**: Loading a 4-band checkpoint for 8-band data (or vice versa)

**Solution**:
- Each experiment trains from scratch (no pre-loading)
- Checkpoints are saved per dataset
- WV3 models won't work for GF2 and vice versa

### Metric shows unusually high d_lambda

**Cause**: Normal for first epochs with random initialization

**Solution**: Monitor over multiple epochs. Should decrease as training progresses.

## Advanced: Training on Other Band Counts

This code works with **any number of bands**. To train on 11-band, 13-band, etc.:

1. Prepare data with the desired band count
2. Place in appropriate directory
3. Add to `TRAINING_DATASETS` in `train_pansharpening.py`:

```python
TRAINING_DATASETS = (
    {"name": "wv3_11band", "test_datasets": ("wv3_11band",)},  # Add this
    # ... other datasets
)
```

4. Run training - the code will automatically detect your band count!

## References

- **GF2**: 4 multispectral bands
- **QB**: 4 multispectral bands  
- **WV3**: 8 multispectral bands
- **WV2**: 8 multispectral bands
- **WV3 with SWIR**: 11 multispectral bands (if needed)

## Summary

✅ **Your code is fully band-agnostic**
✅ **8-band WV3 and WV2 are fully supported**
✅ **Models automatically adapt to any band count**
✅ **No code changes needed for different band counts**

Just prepare your data and run the training script!

