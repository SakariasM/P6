"""
Student Generator with Attention Hooks

A U-Net inpainting generator augmented with CBAM attention modules at each
encoder level.  During the forward pass it also returns:
  - intermediate encoder feature maps  (for feature mimicry loss)
  - spatial attention maps from each CBAM block  (for attention transfer loss)

These are kept separate from the model's main output so the distillation
losses can be applied without affecting the standard inference path.

Architecture (default depth=4, base_channels=64):

  Input: [B, 4, H, W]  (masked RGB + 1-channel mask)

  initial  →  [B,  64, H,   W  ]
  enc[0]   →  [B, 128, H/2, W/2]  + CBAM → att_map[0], feat[0]
  enc[1]   →  [B, 256, H/4, W/4]  + CBAM → att_map[1], feat[1]
  enc[2]   →  [B, 512, H/8, W/8]  + CBAM → att_map[2], feat[2]  ← aligned to teacher P4
  enc[3]   →  [B,1024,H/16,W/16]  + CBAM → att_map[3], feat[3]  ← aligned to teacher P5
  bottleneck (dilated residuals)
  dec[0..3] with skip connections
  output   →  [B, 3, H, W]  (Tanh)

AttentionProjection layers map student encoder channels to teacher channels
so the feature mimicry loss can be computed in a common space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolo_distillation.attention import CBAM, AttentionProjection


# ---------------------------------------------------------------------------
# Primitive blocks (self-contained, no dependency on existing models/)
# ---------------------------------------------------------------------------

class GatedConv2d(nn.Module):
    """Gated convolution — learns a per-location mask over activations."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 1):
        super().__init__()
        kw = dict(stride=stride, padding=padding, dilation=dilation)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, **kw)
        self.gate = nn.Conv2d(in_ch, out_ch, kernel_size, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) * torch.sigmoid(self.gate(x))


class DownBlock(nn.Module):
    """Encoder stage: stride-2 gated conv → norm → act → gated conv → norm → act."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = GatedConv2d(in_ch, out_ch, stride=2)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.conv2 = GatedConv2d(out_ch, out_ch)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.act   = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    """Decoder stage: nearest upsample → cat(skip) → gated conv × 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = GatedConv2d(in_ch, out_ch)
        self.norm1 = nn.InstanceNorm2d(out_ch)
        self.conv2 = GatedConv2d(out_ch, out_ch)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class ResBlock(nn.Module):
    """Dilated residual block for the bottleneck."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = GatedConv2d(channels, channels, dilation=dilation,
                                  padding=dilation)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = GatedConv2d(channels, channels)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + r)


# ---------------------------------------------------------------------------
# Student Generator
# ---------------------------------------------------------------------------

class StudentGenerator(nn.Module):
    """U-Net inpainting generator with CBAM attention and distillation hooks.

    Args:
        config (dict): Hyperparameters:
            in_channels    (int) : default 4  (masked RGB + mask)
            out_channels   (int) : default 3  (RGB)
            base_channels  (int) : default 64
            depth          (int) : default 4
        teacher_channels (list[int]): Channel sizes of the teacher features
            at each scale you want to align.  Length must be ≤ depth.
            E.g. [256, 512, 1024] for a yolov8s teacher aligned at the last
            three encoder levels.

    Forward returns:
        output        : [B, 3, H, W]  inpainted image (Tanh)
        distill_info  : dict
            "features"      : list of [B, C, H', W'] encoder feature tensors
            "attention_maps": list of [B, 1, H', W'] spatial attention maps
            "projected"     : list of [B, C_t, H_t, W_t] features projected
                              into teacher channel space (ready for mimicry loss)
    """

    def __init__(self, config: dict, teacher_channels: list):
        super().__init__()

        in_ch    = config.get("in_channels",   4)
        out_ch   = config.get("out_channels",  3)
        base     = config.get("base_channels", 64)
        depth    = config.get("depth",          4)

        # ---- Initial conv (no downsampling) --------------------------------
        self.initial = nn.Sequential(
            GatedConv2d(in_ch, base, kernel_size=7, padding=3),
            nn.InstanceNorm2d(base),
            nn.ReLU(inplace=True),
        )

        # ---- Encoder --------------------------------------------------------
        self.encoder = nn.ModuleList()
        self.cbam    = nn.ModuleList()
        ch = base
        self._enc_channels = []          # track channels at each encoder output
        for _ in range(depth):
            self.encoder.append(DownBlock(ch, ch * 2))
            ch *= 2
            self.cbam.append(CBAM(ch))
            self._enc_channels.append(ch)

        # ---- Bottleneck -----------------------------------------------------
        self.bottleneck = nn.Sequential(
            ResBlock(ch, dilation=1),
            ResBlock(ch, dilation=2),
            ResBlock(ch, dilation=4),
            ResBlock(ch, dilation=2),
            ResBlock(ch, dilation=1),
        )

        # ---- Decoder --------------------------------------------------------
        self.decoder = nn.ModuleList()
        for i in range(depth):
            in_d  = ch + ch // 2
            out_d = ch // 2
            self.decoder.append(UpBlock(in_d, out_d))
            ch //= 2

        # ---- Output head ----------------------------------------------------
        self.output_head = nn.Sequential(
            GatedConv2d(ch, ch),
            nn.InstanceNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, out_ch, kernel_size=7, padding=3),
            nn.Tanh(),
        )

        # ---- Distillation projection layers ---------------------------------
        # We align the last len(teacher_channels) encoder levels to teacher.
        # e.g. depth=4, teacher has 3 scales → align enc levels 1, 2, 3
        n_align = len(teacher_channels)
        assert n_align <= depth, (
            f"teacher_channels has {n_align} scales but depth={depth}. "
            "Cannot align more scales than encoder levels."
        )
        student_aligned = self._enc_channels[depth - n_align:]   # last n_align levels
        self.projections = nn.ModuleList([
            AttentionProjection(s_ch, t_ch)
            for s_ch, t_ch in zip(student_aligned, teacher_channels)
        ])
        self._n_align   = n_align
        self._depth     = depth

    def forward(self, masked_image: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            masked_image: [B, 3, H, W]
            mask        : [B, 1, H, W]  (1 = hole region)

        Returns:
            output      : [B, 3, H, W]
            distill_info: dict (see class docstring)
        """
        x = torch.cat([masked_image, mask], dim=1)
        x = self.initial(x)

        # Encoder — capture features and attention maps at every level
        skips        = []
        enc_features = []
        att_maps     = []
        for enc, cbam in zip(self.encoder, self.cbam):
            skips.append(x)
            x = enc(x)
            att_map, x = cbam(x)
            enc_features.append(x)
            att_maps.append(att_map)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, dec in enumerate(self.decoder):
            skip = skips[-(i + 1)]
            x = dec(x, skip)

        output = self.output_head(x)

        # Build distillation info
        # "aligned" encoder features = last n_align encoder levels
        aligned_feats = enc_features[self._depth - self._n_align:]
        aligned_atts  = att_maps[self._depth - self._n_align:]

        projected = [
            proj(feat)
            for proj, feat in zip(self.projections, aligned_feats)
        ]

        distill_info = {
            "features":       aligned_feats,   # raw student features (aligned levels)
            "attention_maps": aligned_atts,     # spatial attention maps from CBAM
            "projected":      projected,        # projected into teacher channel space
        }

        return output, distill_info
