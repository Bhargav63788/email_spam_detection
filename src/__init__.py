"""
Email Spam Detection Package
Production-ready Machine Learning system for classifying text/email messages
as SPAM or NOT_SPAM.
"""

from src.dataset import load_and_preprocess_dataset, split_dataset
from src.evaluator import evaluate_model, print_evaluation_summary
from src.pipeline import build_pipeline, load_pipeline, save_pipeline
from src.predictor import SpamPredictor
from src.preprocessor import clean_text, validate_input

__all__ = [
    "clean_text",
    "validate_input",
    "load_and_preprocess_dataset",
    "split_dataset",
    "build_pipeline",
    "save_pipeline",
    "load_pipeline",
    "evaluate_model",
    "print_evaluation_summary",
    "SpamPredictor",
]
