"""
Distillation Losses

Three complementary losses transfer knowledge from the YOLO teacher to the
student inpainting network:

    AttentionTransferLoss   — forces the student's CBAM spatial attention maps
                              to match the teacher's spatial attention maps.
                              Operates in [0, 1] normalised attention space.

    FeatureMimicryLoss      — L2 distance between projected student features
                              and teacher features (after spatially aligning them).

    RelationDistillationLoss — preserves pairwise inter-sample feature
                               relationships via Gram matrix comparison
                               (captures structural / textural patterns).

    DistillationLoss        — weighted combination of all three, plus optional
                              standard inpainting reconstruction term.

All losses work on lists of tensors (one per scale) and average across scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual loss modules
# ---------------------------------------------------------------------------

class AttentionTransferLoss(nn.Module):
    """Align student CBAM attention maps with teacher spatial attention maps.

    Both maps are in [0, 1].  Teacher maps are bilinearly interpolated to the
    student spatial resolution before computing MSE.

    Args:
        reduction (str): "mean" or "sum"
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self,
                student_atts: list,
                teacher_atts: list) -> torch.Tensor:
        """
        Args:
            student_atts: list of [B, 1, H_s, W_s] — student CBAM maps
            teacher_atts: list of [B, 1, H_t, W_t] — teacher pseudo-attention maps

        Returns:
            Scalar loss averaged across scales.
        """
        assert len(student_atts) == len(teacher_atts), (
            "Number of student and teacher attention scales must match. "
            f"Got {len(student_atts)} vs {len(teacher_atts)}."
        )
        total = 0.0
        for s_att, t_att in zip(student_atts, teacher_atts):
            # Resize teacher attention to student spatial dims
            if (t_att.shape[2], t_att.shape[3]) != (s_att.shape[2], s_att.shape[3]):
                t_att = F.interpolate(t_att, size=s_att.shape[2:],
                                      mode="bilinear", align_corners=False)
            if self.reduction == "mean":
                total += F.mse_loss(s_att, t_att.detach())
            else:
                total += F.mse_loss(s_att, t_att.detach(), reduction="sum")

        return total / len(student_atts)


class FeatureMimicryLoss(nn.Module):
    """L2 distance between student (projected) and teacher feature maps.

    The student's AttentionProjection layers already map student channels to
    teacher channel space, so this loss only needs to handle spatial alignment.

    Args:
        reduction (str): "mean" or "sum"
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self,
                projected_student_feats: list,
                teacher_feats: list) -> torch.Tensor:
        """
        Args:
            projected_student_feats: list of [B, C_t, H_s, W_s]
            teacher_feats          : list of [B, C_t, H_t, W_t]

        Returns:
            Scalar loss averaged across scales.
        """
        assert len(projected_student_feats) == len(teacher_feats)
        total = 0.0
        for s_feat, t_feat in zip(projected_student_feats, teacher_feats):
            # Spatially align student to teacher resolution
            if (s_feat.shape[2], s_feat.shape[3]) != (t_feat.shape[2], t_feat.shape[3]):
                s_feat = F.interpolate(s_feat, size=t_feat.shape[2:],
                                       mode="bilinear", align_corners=False)
            if self.reduction == "mean":
                total += F.mse_loss(s_feat, t_feat.detach())
            else:
                total += F.mse_loss(s_feat, t_feat.detach(), reduction="sum")

        return total / len(projected_student_feats)


class RelationDistillationLoss(nn.Module):
    """Gram-matrix based relational distillation.

    Computes the Gram matrix (channel covariance) of student and teacher
    features and penalises their difference.  This preserves second-order
    statistics (texture / style patterns) rather than exact feature values.

    Args:
        reduction (str): "mean" or "sum"
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    @staticmethod
    def _gram(feat: torch.Tensor) -> torch.Tensor:
        """Normalised Gram matrix.

        Args:
            feat: [B, C, H, W]

        Returns:
            [B, C, C]
        """
        b, c, h, w = feat.shape
        f = feat.view(b, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

    def forward(self,
                projected_student_feats: list,
                teacher_feats: list) -> torch.Tensor:
        """
        Args:
            projected_student_feats: list of [B, C_t, H_s, W_s]
            teacher_feats          : list of [B, C_t, H_t, W_t]

        Returns:
            Scalar loss averaged across scales.
        """
        assert len(projected_student_feats) == len(teacher_feats)
        total = 0.0
        for s_feat, t_feat in zip(projected_student_feats, teacher_feats):
            # Gram matrices are spatial-size independent, no interpolation needed
            g_s = self._gram(s_feat)
            g_t = self._gram(t_feat.detach())
            if self.reduction == "mean":
                total += F.l1_loss(g_s, g_t)
            else:
                total += F.l1_loss(g_s, g_t, reduction="sum")

        return total / len(projected_student_feats)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class DistillationLoss(nn.Module):
    """Weighted combination of all distillation and reconstruction losses.

    Weights are read from config dict:
        config["distillation"]["attention_weight"]   (default 1.0)
        config["distillation"]["mimicry_weight"]     (default 0.5)
        config["distillation"]["relation_weight"]    (default 0.5)
        config["distillation"]["reconstruction_l1_weight"] (default 1.0)

    Args:
        config (dict): Top-level training config.
    """

    def __init__(self, config: dict):
        super().__init__()
        dc = config.get("distillation", {})
        self.att_w   = dc.get("attention_weight",        1.0)
        self.mim_w   = dc.get("mimicry_weight",          0.5)
        self.rel_w   = dc.get("relation_weight",         0.5)
        self.rec_w   = dc.get("reconstruction_l1_weight", 1.0)

        self.att_loss = AttentionTransferLoss()
        self.mim_loss = FeatureMimicryLoss()
        self.rel_loss = RelationDistillationLoss()

    def forward(self,
                student_out: torch.Tensor,
                target: torch.Tensor,
                student_atts: list,
                teacher_atts: list,
                projected_student_feats: list,
                teacher_feats: list) -> tuple:
        """
        Args:
            student_out             : [B, 3, H, W] inpainted output
            target                  : [B, 3, H, W] ground-truth image
            student_atts            : list of student CBAM attention maps
            teacher_atts            : list of teacher spatial attention maps
            projected_student_feats : list of projected student features
            teacher_feats           : list of teacher feature maps

        Returns:
            (total_loss, loss_dict)
            total_loss : scalar
            loss_dict  : {
                "reconstruction": float,
                "attention":      float,
                "mimicry":        float,
                "relation":       float,
                "total":          float,
            }
        """
        losses = {}
        total  = 0.0

        # Pixel-level reconstruction
        rec = F.l1_loss(student_out, target)
        losses["reconstruction"] = rec.item()
        total += self.rec_w * rec

        # Attention transfer
        att = self.att_loss(student_atts, teacher_atts)
        losses["attention"] = att.item()
        total += self.att_w * att

        # Feature mimicry
        mim = self.mim_loss(projected_student_feats, teacher_feats)
        losses["mimicry"] = mim.item()
        total += self.mim_w * mim

        # Relational distillation
        rel = self.rel_loss(projected_student_feats, teacher_feats)
        losses["relation"] = rel.item()
        total += self.rel_w * rel

        losses["total"] = total.item()
        return total, losses
