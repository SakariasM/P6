"""
U-Net segmentation student with CBAM attention and distillation hooks.

Input: [B, 3, H, W] RGB image
Output: [B, 1, H, W] person segmentation mask (sigmoid)

Architecture (depth=4, base_channels=32):

  initial  -> [B,  32, H,   W  ]
  enc[0]   -> [B,  64, H/2, W/2]  + CBAM
  enc[1]   -> [B, 128, H/4, W/4]  + CBAM
  enc[2]   -> [B, 256, H/8, W/8]  + CBAM
  enc[3]   -> [B, 512, H/16,W/16] + CBAM
  bottleneck (dilated residuals)
  dec[0..3] with skip connections
  output   -> [B, 1, H, W]  (Sigmoid)

Distillation info exposes features, attention maps, and projected features
from the last N encoder levels aligned to the teacher's feature scales.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from student.attention import CBAM, AttentionProjection


class DownBlock(nn.Module):
    """Encoder: stride-2 conv -> BN -> LeakyReLU -> conv -> BN -> LeakyReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Decoder: upsample 2x -> cat(skip) -> conv -> BN -> ReLU -> conv -> BN -> ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResBlock(nn.Module):
    """Dilated residual block for the bottleneck."""
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class StudentSegmentation(nn.Module):
    """U-Net segmentation student with CBAM attention and distillation hooks.

    Args:
        in_channels: Input channels (3 for RGB)
        base_channels: Base channel count (doubled each encoder level)
        depth: Number of encoder/decoder levels
        teacher_channels: Channel counts of teacher features at each scale.
                         E.g. [128, 128, 256] for YOLO26n at layers 4, 6, 10.

    Forward returns:
        output: [B, 1, H, W] segmentation mask (sigmoid)
        distill_info: dict with "features", "attention_maps", "projected"
    """

    def __init__(self, in_channels=3, base_channels=32, depth=4,
                 teacher_channels: Optional[List[int]] = None):
        super().__init__()

        if teacher_channels is None:
            teacher_channels = [128, 128, 256]

        self.depth = depth
        self.teacher_channels = teacher_channels

        # Encoder channels: [64, 128, 256, 512] for base=32, depth=4
        self._enc_channels = [base_channels * (2 ** i) for i in range(1, depth + 1)]

        # Initial conv (no downsampling)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.encoders = nn.ModuleList()
        self.cbams = nn.ModuleList()
        enc_in = base_channels
        for i in range(depth):
            enc_out = self._enc_channels[i]
            self.encoders.append(DownBlock(enc_in, enc_out))
            self.cbams.append(CBAM(enc_out))
            enc_in = enc_out

        # Bottleneck
        bottleneck_ch = self._enc_channels[-1]
        self.bottleneck = nn.Sequential(
            ResBlock(bottleneck_ch, dilation=1),
            ResBlock(bottleneck_ch, dilation=2),
            ResBlock(bottleneck_ch, dilation=1),
        )

        # Decoder
        self.decoders = nn.ModuleList()
        dec_in_ch = bottleneck_ch
        for i in range(depth):
            skip_idx = depth - 1 - i
            skip_ch = self._enc_channels[skip_idx - 1] if skip_idx > 0 else base_channels
            dec_out = skip_ch
            self.decoders.append(UpBlock(dec_in_ch + skip_ch, dec_out))
            dec_in_ch = dec_out

        # Output head
        self.output_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
            nn.Sigmoid(),
        )

        # Distillation projections
        n_align = len(teacher_channels)
        assert n_align <= depth
        aligned_start = depth - n_align
        self.projections = nn.ModuleList([
            AttentionProjection(self._enc_channels[aligned_start + i], teacher_channels[i])
            for i in range(n_align)
        ])
        self._aligned_start = aligned_start

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        x = self.initial(x)

        # Encoder
        skips = []
        enc_features = []
        attn_maps = []
        for i in range(self.depth):
            skips.append(x)
            x = self.encoders[i](x)
            sp_map, x = self.cbams[i](x)
            enc_features.append(x)
            attn_maps.append(sp_map)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(self.depth):
            skip_idx = self.depth - 1 - i
            x = self.decoders[i](x, skips[skip_idx])

        output = self.output_head(x)

        # Distillation info from aligned encoder levels
        n = len(self.teacher_channels)
        s = self._aligned_start
        distill_info = {
            "features": [enc_features[s + i] for i in range(n)],
            "attention_maps": [attn_maps[s + i] for i in range(n)],
            "projected": [self.projections[i](enc_features[s + i]) for i in range(n)],
        }

        return output, distill_info
