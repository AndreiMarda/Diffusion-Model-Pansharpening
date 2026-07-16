import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        return self.conv(torch.cat([x, skip], dim=1))


class UNetFeatureExtractor(nn.Module):
    def __init__(self, in_channels, feature_channels=(32, 64, 128)):
        super().__init__()
        if len(feature_channels) != 3:
            raise ValueError("UNetFeatureExtractor expects exactly three feature scales.")

        c1, c2, c3 = feature_channels
        bottleneck_channels = c3 * 2

        # Encoder
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = DownBlock(c1, c2)
        self.enc3 = DownBlock(c2, c3)

        # bottleneck
        self.bottleneck = DownBlock(c3, bottleneck_channels)

        # Decoder
        self.dec3 = UpBlock(bottleneck_channels, c3, c3)
        self.dec2 = UpBlock(c3, c2, c2)
        self.dec1 = UpBlock(c2, c1, c1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        bottleneck = self.bottleneck(enc3)

        dec3 = self.dec3(bottleneck, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)

        return [dec1, dec2, dec3]


class GatedFusionPyramid(nn.Module):
    def __init__(self, feature_channels=(32, 64, 128)):
        super().__init__()
        self.gates = nn.ModuleList(
            nn.Conv2d(channels * 2, channels, kernel_size=1)
            for channels in feature_channels
        )
        self.projections = nn.ModuleList(
            nn.Conv2d(channels * 2, channels, kernel_size=1)
            for channels in feature_channels
        )

    def forward(self, spatial_feats, spectral_feats):
        if len(spatial_feats) != len(spectral_feats):
            raise ValueError("Spatial and spectral pyramids must have the same number of scales.")

        fused_features = []
        for spatial, spectral, gate_layer, projection in zip(
            spatial_feats, spectral_feats, self.gates, self.projections
        ):
            if spatial.shape != spectral.shape:
                raise ValueError(
                    f"Feature shapes must match before fusion: {spatial.shape} != {spectral.shape}"
                )

            combined = torch.cat([spatial, spectral], dim=1)
            gate = torch.sigmoid(gate_layer(combined))
            candidate = projection(combined)
            fused_features.append(gate * candidate + (1.0 - gate) * spatial)

        return fused_features
