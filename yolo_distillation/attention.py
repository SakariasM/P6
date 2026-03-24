import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        
        b, c = x.shape[:2]
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        gate = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return gate, x * gate


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
       
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True)[0]
        att_map = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return att_map, x * att_map


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor):
       
        _, x = self.channel_att(x)
        spatial_map, out = self.spatial_att(x)
        return spatial_map, out


class AttentionProjection(nn.Module):
    def __init__(self, student_channels: int, teacher_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        x = self.proj(x)
        if target_size is not None and (x.shape[2], x.shape[3]) != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x
