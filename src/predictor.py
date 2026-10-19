"""
Predictor Module
Provides inference service with input validation, probability estimation,
and robust error handling for user messages.
"""

from typing import Any, Dict, Optional
from sklearn.pipeline import Pipeline

from src.pipeline import load_pipeline
from src.preprocessor import validate_input


class SpamPredictor:
    """
    High-level predictor interface for Email/SMS Spam Detection.
    Handles user input sanitization, graceful error handling, and model inference.
    """

    def __init__(self, pipeline: Optional[Pipeline] = None, model_path: Optional[str] = None):
        """
        Initializes predictor with either an in-memory pipeline or path to saved model.

        Args:
            pipeline (Optional[Pipeline]): Pre-trained pipeline object.
            model_path (Optional[str]): Path to serialized .joblib model file.
        """
        if pipeline is not None:
            self.pipeline = pipeline
        elif model_path is not None:
            self.pipeline = load_pipeline(model_path)
        else:
            raise ValueError("Must provide either a trained 'pipeline' or a valid 'model_path'.")

    def predict(self, text: Any) -> Dict[str, Any]:
        """
        Predicts whether a message is SPAM or NOT_SPAM.
        Handles empty messages and non-string inputs gracefully without crashing (TC3 & TC4).

        Args:
            text (Any): The raw input message.

        Returns:
            Dict[str, Any]: Structured prediction result including label, confidence,
                            and probability distribution.
        """
        # Step 1: Input Validation & Sanitization
        is_valid, cleaned_text, error_msg = validate_input(text)

        if not is_valid:
            return {
                "status": "error",
                "is_valid": False,
                "raw_input": text,
                "cleaned_input": cleaned_text,
                "error_message": error_msg,
                "predicted_label": None,
                "confidence": 0.0,
                "probabilities": {
                    "SPAM": 0.0,
                    "NOT_SPAM": 0.0
                }
            }

        # Step 2: Feature vectorization & Model Inference
        # Pipeline receives cleaned text as a single-element list
        pred_label = self.pipeline.predict([cleaned_text])[0]

        # Step 3: Probabilities and confidence
        probabilities = {}
        confidence = 1.0

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba([cleaned_text])[0]
            classes = self.pipeline.classes_
            for cls_name, prob_val in zip(classes, probs):
                probabilities[str(cls_name)] = float(prob_val)
            confidence = float(probabilities.get(pred_label, max(probs)))

        return {
            "status": "success",
            "is_valid": True,
            "raw_input": text,
            "cleaned_input": cleaned_text,
            "error_message": "",
            "predicted_label": pred_label,
            "confidence": confidence,
            "probabilities": probabilities
        }

    def format_prediction_result(self, result: Dict[str, Any]) -> str:
        """Formats prediction output for display to user or terminal."""
        if result["status"] == "error":
            return (
                f"[REJECTED / INVALID INPUT]\n"
                f"  Raw Input: {repr(result['raw_input'])}\n"
                f"  Details  : {result['error_message']}"
            )

        prob_str = ""
        if result["probabilities"]:
            prob_parts = [f"{k}: {v*100:.2f}%" for k, v in result["probabilities"].items()]
            prob_str = f" | Probabilities: ({', '.join(prob_parts)})"

        return (
            f"Input            : {result['raw_input']}\n"
            f"Normalized Text  : {result['cleaned_input']}\n"
            f"Prediction       : {result['predicted_label']}\n"
            f"Confidence       : {result['confidence'] * 100:.2f}%{prob_str}"
        )
