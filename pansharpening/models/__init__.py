from pansharpening.models.conditioning import GatedFusionPyramid, UNetFeatureExtractor
from pansharpening.models.denoiser import ConditionalDDPMUNet


def build_models(image_channels, feature_channels, num_time_steps, device):
    spatial_unet = UNetFeatureExtractor(
        in_channels=1,  # 1 for panchromatic
        feature_channels=feature_channels,
    ).to(device)
    spectral_unet = UNetFeatureExtractor(
        in_channels=image_channels,  # number of spectral bands
        feature_channels=feature_channels,
    ).to(device)
    gated_fusion_pyramid = GatedFusionPyramid(feature_channels=feature_channels).to(device)
    denoiser = ConditionalDDPMUNet(
        image_channels=image_channels,  # Adapts to number of spectral bands
        feature_channels=feature_channels,
        num_time_steps=num_time_steps,
    ).to(device)

    return spatial_unet, spectral_unet, gated_fusion_pyramid, denoiser
