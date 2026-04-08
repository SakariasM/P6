"""
CBAM attention modules and projection layers for distillation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c = x.shape[:2]
        a = self.fc(x.mean(dim=(2, 3)))
        m = self.fc(x.amax(dim=(2, 3)))
        gate = self.sig(a + m).view(b, c, 1, 1)
        return gate, x * gate


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg_f = x.mean(dim=1, keepdim=True)
        max_f = x.max(dim=1, keepdim=True)[0]
        attn = self.sig(self.conv(torch.cat([avg_f, max_f], dim=1)))
        return attn, x * attn


class CBAM(nn.Module):
    """Channel attention first, then spatial."""
    def __init__(self, channels, reduction=16, k=7):
        super().__init__()
        self.ch = ChannelAttention(channels, reduction)
        self.spa = SpatialAttention(k)

    def forward(self, x):
        _, x = self.ch(x)
        sp_map, out = self.spa(x)
        return sp_map, out


class AttentionProjection(nn.Module):
    """1x1 conv to map student channels into teacher feature space."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, target_size=None):
        x = self.proj(x)
        if target_size is not None and (x.shape[2], x.shape[3]) != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x
