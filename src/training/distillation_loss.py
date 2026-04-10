"""
Distillation losses for offline segmentation knowledge distillation.

Three losses:
    AttentionTransferLoss   - MSE between student CBAM maps and teacher attention maps
    FeatureMimicryLoss      - L2 between projected student features and teacher features
    RelationDistillationLoss - Gram matrix L1 loss (second-order statistics)
    SegmentationDistillationLoss - weighted combination
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionTransferLoss(nn.Module):
    def forward(self, student_atts: list, teacher_atts: list) -> torch.Tensor:
        losses = []
        for s, t in zip(student_atts, teacher_atts):
            if (t.shape[2], t.shape[3]) != (s.shape[2], s.shape[3]):
                t = F.interpolate(t, size=s.shape[2:], mode='bilinear', align_corners=False)
            losses.append(F.mse_loss(s, t.detach()))
        return torch.stack(losses).mean()


class FeatureMimicryLoss(nn.Module):
    def forward(self, projected_student_feats: list, teacher_feats: list) -> torch.Tensor:
        assert len(projected_student_feats) == len(teacher_feats)
        losses = []
        for s_feat, t_feat in zip(projected_student_feats, teacher_feats):
            if (s_feat.shape[2], s_feat.shape[3]) != (t_feat.shape[2], t_feat.shape[3]):
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:], mode='bilinear', align_corners=False)
            losses.append(F.mse_loss(s_feat, t_feat.detach()))
        return torch.stack(losses).mean()


class RelationDistillationLoss(nn.Module):
    @staticmethod
    def _gram(feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        f = feat.view(b, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

    def forward(self, projected_student_feats: list, teacher_feats: list) -> torch.Tensor:
        assert len(projected_student_feats) == len(teacher_feats)
        losses = []
        for s_feat, t_feat in zip(projected_student_feats, teacher_feats):
            g_s = self._gram(s_feat)
            g_t = self._gram(t_feat.detach())
            losses.append(F.l1_loss(g_s, g_t))
        return torch.stack(losses).mean()


class SegmentationLoss(nn.Module):
    """BCE + Dice loss comparing student mask against teacher pseudo ground truth."""

    def forward(self, student_mask: torch.Tensor, teacher_mask: torch.Tensor) -> torch.Tensor:
        # student_mask: [B, 1, H, W] sigmoid output
        # teacher_mask: [B, 1, H, W] binary {0, 1}
        if student_mask.shape[2:] != teacher_mask.shape[2:]:
            teacher_mask = F.interpolate(teacher_mask, size=student_mask.shape[2:],
                                         mode='bilinear', align_corners=False)
        bce = F.binary_cross_entropy(student_mask, teacher_mask.float())
        # Dice loss
        smooth = 1.0
        s_flat = student_mask.view(-1)
        t_flat = teacher_mask.float().view(-1)
        intersection = (s_flat * t_flat).sum()
        dice = 1 - (2. * intersection + smooth) / (s_flat.sum() + t_flat.sum() + smooth)
        return bce + dice


class SegmentationDistillationLoss(nn.Module):
    def __init__(self, attention_weight=1.0, mimicry_weight=0.5, relation_weight=0.5,
                 seg_weight=0.0):
        super().__init__()
        self.att_w = attention_weight
        self.mim_w = mimicry_weight
        self.rel_w = relation_weight
        self.seg_w = seg_weight
        self.att_loss = AttentionTransferLoss()
        self.mim_loss = FeatureMimicryLoss()
        self.rel_loss = RelationDistillationLoss()
        if seg_weight > 0:
            self.seg_loss = SegmentationLoss()

    def forward(self, student_atts, teacher_atts, projected_student_feats, teacher_feats,
                student_mask=None, teacher_mask=None):
        losses = {}
        att = self.att_loss(student_atts, teacher_atts)
        losses["attention"] = att.item()
        mim = self.mim_loss(projected_student_feats, teacher_feats)
        losses["mimicry"] = mim.item()
        rel = self.rel_loss(projected_student_feats, teacher_feats)
        losses["relation"] = rel.item()
        total = self.att_w * att + self.mim_w * mim + self.rel_w * rel

        if self.seg_w > 0 and student_mask is not None and teacher_mask is not None:
            seg = self.seg_loss(student_mask, teacher_mask)
            losses["segmentation"] = seg.item()
            total = total + self.seg_w * seg

        losses["total"] = total.item()
        return total, losses
