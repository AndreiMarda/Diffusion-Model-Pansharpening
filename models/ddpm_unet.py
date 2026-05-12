import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels, max_groups=8):
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=torch.float32) * -scale
        )
        args = t.float()[:, None] * frequencies[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class TimeConditionedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, cond_channels=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.cond_proj = (
            nn.Conv2d(cond_channels, out_channels, kernel_size=1)
            if cond_channels is not None
            else None
        )
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_embedding, cond=None):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_embedding)[:, :, None, None]

        if self.cond_proj is not None and cond is not None:
            if cond.shape[-2:] != h.shape[-2:]:
                cond = F.interpolate(cond, size=h.shape[-2:], mode="bilinear", align_corners=False)
            h = h + self.cond_proj(cond)

        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalDDPMUNet(nn.Module):
    """DDPM U-Net denoiser that predicts epsilon from x_t, timestep, and condition pyramid."""

    def __init__(
        self,
        image_channels,
        feature_channels=(32, 64, 128),
        time_dim=256,
        num_time_steps=1000,
    ):
        super().__init__()
        if len(feature_channels) != 3:
            raise ValueError("ConditionalDDPMUNet expects exactly three condition scales.")

        c1, c2, c3 = feature_channels
        c4 = c3 * 2

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(image_channels, c1, kernel_size=3, padding=1)

        self.enc1 = TimeConditionedBlock(c1, c1, time_dim, cond_channels=c1)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.enc2 = TimeConditionedBlock(c2, c2, time_dim, cond_channels=c2)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)
        self.enc3 = TimeConditionedBlock(c3, c3, time_dim, cond_channels=c3)
        self.down3 = nn.Conv2d(c3, c4, kernel_size=4, stride=2, padding=1)

        self.mid1 = TimeConditionedBlock(c4, c4, time_dim)
        self.mid2 = TimeConditionedBlock(c4, c4, time_dim)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=4, stride=2, padding=1)
        self.dec3 = TimeConditionedBlock(c3 + c3, c3, time_dim, cond_channels=c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.dec2 = TimeConditionedBlock(c2 + c2, c2, time_dim, cond_channels=c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.dec1 = TimeConditionedBlock(c1 + c1, c1, time_dim, cond_channels=c1)

        self.output = nn.Sequential(
            nn.GroupNorm(_num_groups(c1), c1),
            nn.SiLU(),
            nn.Conv2d(c1, image_channels, kernel_size=3, padding=1),
        )

    def forward(self, x_t, t, cond):
        if len(cond) != 3:
            raise ValueError("ConditionalDDPMUNet expects cond=[64x64, 32x32, 16x16].")

        cond1, cond2, cond3 = cond
        time_embedding = self.time_mlp(t)

        x = self.input_conv(x_t)
        enc1 = self.enc1(x, time_embedding, cond1)
        enc2 = self.enc2(self.down1(enc1), time_embedding, cond2)
        enc3 = self.enc3(self.down2(enc2), time_embedding, cond3)

        x = self.down3(enc3)
        x = self.mid1(x, time_embedding)
        x = self.mid2(x, time_embedding)

        x = self.up3(x)
        x = self.dec3(torch.cat([x, enc3], dim=1), time_embedding, cond3)
        x = self.up2(x)
        x = self.dec2(torch.cat([x, enc2], dim=1), time_embedding, cond2)
        x = self.up1(x)
        x = self.dec1(torch.cat([x, enc1], dim=1), time_embedding, cond1)

        return self.output(x)
