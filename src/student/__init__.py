"""
Student model architectures for knowledge distillation.
Includes feature adapters for matching teacher dimensions.
"""
from .student_model import (
    StudentYOLO,
    FeatureMatchingLayer,
    create_student_from_teacher
)
from .attention import CBAM, AttentionProjection

__all__ = [
    'StudentYOLO',
    'FeatureMatchingLayer',
    'create_student_from_teacher',
    'CBAM',
    'AttentionProjection',
]
