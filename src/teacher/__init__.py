"""
Teacher model inference and feature extraction.
Extracts both predictions and intermediate features for distillation.
"""
from .predictions import TeacherPrediction, YOLOTeacherInference, load_predictions
from .feature_extractor import YOLOFeatureExtractor, FeatureHook
from .hybrid_predictions import (
    HybridTeacherPrediction,
    HybridYOLOInference,
    load_hybrid_predictions
)

__all__ = [
    'TeacherPrediction',
    'YOLOTeacherInference',
    'load_predictions',
    'YOLOFeatureExtractor',
    'FeatureHook',
    'HybridTeacherPrediction',
    'HybridYOLOInference',
    'load_hybrid_predictions',
]
