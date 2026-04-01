"""
YOLO Attention Distillation for Inpainting

A self-contained module for training an inpainting student network guided by
attention maps extracted from a frozen pretrained YOLO teacher.

Structure:
    teacher.py          — YOLO teacher wrapper with feature hook registration
    attention.py        — Attention modules (CBAM, SpatialAttention, ChannelAttention,
                          AttentionProjection)
    student.py          — Student inpainting generator with built-in attention hooks
    distillation_loss.py — Attention transfer, feature mimicry, and relation losses
    trainer.py          — DistillationTrainer orchestrating teacher + student
    config.yaml         — All hyperparameters

Quick start:
    from yolo_distillation.teacher import YOLOTeacher
    from yolo_distillation.student import StudentGenerator
    from yolo_distillation.distillation_loss import DistillationLoss
    from yolo_distillation.trainer import DistillationTrainer
"""
