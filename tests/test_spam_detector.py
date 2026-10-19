"""
Automated Test Suite for Email Spam Detection System
Specifically validates Test Cases TC1 through TC5 as per project specification.
"""

import unittest
from pathlib import Path

from src.dataset import load_and_preprocess_dataset, split_dataset
from src.evaluator import evaluate_model
from src.pipeline import build_pipeline, load_pipeline, save_pipeline
from src.predictor import SpamPredictor
from src.preprocessor import clean_text, validate_input

DATASET_PATH = Path(__file__).parent.parent / "data" / "emails.csv"
TEMP_MODEL_PATH = Path(__file__).parent.parent / "models" / "test_model.joblib"


class TestEmailSpamDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Train pipeline once for the test suite."""
        df = load_and_preprocess_dataset(str(DATASET_PATH))
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = split_dataset(
            df, test_size=0.2, random_state=42
        )
        cls.pipeline = build_pipeline(model_type="naive_bayes")
        cls.pipeline.fit(cls.X_train, cls.y_train)
        cls.predictor = SpamPredictor(pipeline=cls.pipeline)

    def test_tc1_spam_pattern_prediction(self):
        """
        TC1: A message containing 'You won a free prize, claim now'
        Expected: Predict SPAM if supported by the trained patterns.
        """
        msg = "You won a free prize, claim now"
        result = self.predictor.predict(msg)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["predicted_label"], "SPAM")
        self.assertGreater(result["confidence"], 0.6)

    def test_tc2_ham_pattern_prediction(self):
        """
        TC2: 'Please attend the team meeting   at 10 AM'
        Expected: Predict NOT_SPAM if supported by the trained patterns.
        """
        msg = "Please attend the team meeting   at 10 AM"
        result = self.predictor.predict(msg)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["predicted_label"], "NOT_SPAM")
        self.assertGreater(result["confidence"], 0.6)

    def test_tc3_empty_message_rejection(self):
        """
        TC3: Empty message
        Expected: Reject or handle gracefully without crashing.
        """
        test_inputs = ["", "   ", "\t  \n  ", None]
        for empty_val in test_inputs:
            result = self.predictor.predict(empty_val)
            self.assertEqual(result["status"], "error")
            self.assertFalse(result["is_valid"])
            self.assertIsNone(result["predicted_label"])
            self.assertTrue(len(result["error_message"]) > 0)

    def test_tc4_uppercase_and_extra_spaces(self):
        """
        TC4: Message with uppercase letters and extra spaces
        Expected: Process without crashing and predict appropriately.
        """
        msg = "   CONGRATULATIONS   YOU   HAVE WON   A FREE PRIZE   "
        result = self.predictor.predict(msg)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["is_valid"])
        self.assertEqual(result["predicted_label"], "SPAM")
        self.assertEqual(result["cleaned_input"], "congratulations you have won a free prize")

    def test_tc5_unseen_message_generalization(self):
        """
        TC5: A message not present in the training dataset
        Expected: Generate a prediction without requiring an exact training-data match.
        """
        msg = "Could we sync up tomorrow regarding the new architecture diagrams?"
        result = self.predictor.predict(msg)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["is_valid"])
        self.assertIn(result["predicted_label"], ["SPAM", "NOT_SPAM"])
        self.assertEqual(result["predicted_label"], "NOT_SPAM")

    def test_preprocessing_functions(self):
        """Tests individual text cleaning and validation functions."""
        self.assertEqual(clean_text("   HELLO   WORLD  ! "), "hello world !")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(12345), "12345")

        is_valid, cleaned, err = validate_input("  Valid Message Here  ")
        self.assertTrue(is_valid)
        self.assertEqual(cleaned, "valid message here")
        self.assertEqual(err, "")

        is_valid_empty, _, err_empty = validate_input("   ")
        self.assertFalse(is_valid_empty)
        self.assertTrue("empty" in err_empty.lower())

    def test_model_evaluation_metrics(self):
        """Tests that evaluation returns valid accuracy, precision, recall, and confusion matrix."""
        eval_res = evaluate_model(self.pipeline, self.X_test, self.y_test)
        self.assertIn("accuracy", eval_res)
        self.assertIn("precision", eval_res)
        self.assertIn("recall", eval_res)
        self.assertIn("confusion_matrix", eval_res)

        self.assertGreaterEqual(eval_res["accuracy"], 0.8)
        self.assertGreaterEqual(eval_res["precision"], 0.8)
        self.assertGreaterEqual(eval_res["recall"], 0.8)

        cm = eval_res["confusion_matrix"]
        self.assertIn("true_negatives", cm)
        self.assertIn("false_positives", cm)
        self.assertIn("false_negatives", cm)
        self.assertIn("true_positives", cm)

    def test_model_persistence(self):
        """Tests saving and reloading model pipeline via joblib."""
        save_pipeline(self.pipeline, str(TEMP_MODEL_PATH))
        self.assertTrue(TEMP_MODEL_PATH.exists())

        loaded_pipe = load_pipeline(str(TEMP_MODEL_PATH))
        new_predictor = SpamPredictor(pipeline=loaded_pipe)
        res = new_predictor.predict("Team meeting tomorrow at 9am")
        self.assertEqual(res["predicted_label"], "NOT_SPAM")

        # Cleanup
        if TEMP_MODEL_PATH.exists():
            TEMP_MODEL_PATH.unlink()


if __name__ == "__main__":
    unittest.main()
