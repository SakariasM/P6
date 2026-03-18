"""
Student model architectures for knowledge distillation.
Includes feature adapters for matching teacher dimensions.
"""
from .student_model import (
    StudentYOLO,
    TinyStudentYOLO,
    FeatureMatchingLayer,
    create_student_from_teacher
)

__all__ = [
    'StudentYOLO',
    'TinyStudentYOLO',
    'FeatureMatchingLayer',
    'create_student_from_teacher',
]
