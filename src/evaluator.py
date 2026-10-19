"""
Evaluator Module
Calculates evaluation metrics: Accuracy, Precision, Recall, F1-Score,
and Confusion Matrix with clear visual ASCII rendering.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.pipeline import Pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluates trained pipeline on test data and returns metrics dictionary.

    Args:
        pipeline (Pipeline): Trained scikit-learn pipeline.
        X_test (pd.Series): Test feature texts.
        y_test (pd.Series): True test labels ('SPAM' / 'NOT_SPAM').

    Returns:
        Dict[str, Any]: Structured metrics and confusion matrix values.
    """
    y_pred = pipeline.predict(X_test)

    # Class ordering: ['NOT_SPAM', 'SPAM']
    labels = ["NOT_SPAM", "SPAM"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Extract TN, FP, FN, TP
    # cm layout for labels=['NOT_SPAM', 'SPAM']:
    # [ [TN, FP],
    #   [FN, TP] ]
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label="SPAM", zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label="SPAM", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label="SPAM", zero_division=0))

    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)

    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": labels,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "classification_report": report,
        "test_sample_count": len(y_test),
        "y_true": list(y_test),
        "y_pred": list(y_pred)
    }

    return results


def print_evaluation_summary(results: Dict[str, Any], model_name: str = "Trained Model") -> None:
    """Prints a professional, formatted report of the evaluation metrics and confusion matrix."""
    cm = results["confusion_matrix"]
    tn = cm["true_negatives"]
    fp = cm["false_positives"]
    fn = cm["false_negatives"]
    tp = cm["true_positives"]

    sep = "=" * 64
    subsep = "-" * 64

    print(f"\n{sep}")
    print(f" MODEL EVALUATION REPORT ({model_name.upper()})")
    print(f"{sep}")
    print(f" Test Samples Evaluated: {results['test_sample_count']}")
    print(f" Accuracy : {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
    print(f" Precision: {results['precision']:.4f} (Positive class: SPAM)")
    print(f" Recall   : {results['recall']:.4f} (Positive class: SPAM)")
    print(f" F1-Score : {results['f1_score']:.4f}")
    print(f"{subsep}")
    print(" CONFUSION MATRIX:")
    print("                      Predicted NOT_SPAM    Predicted SPAM")
    print(f"  Actual NOT_SPAM :        {tn:<10}          {fp:<10}  (TN={tn}, FP={fp})")
    print(f"  Actual SPAM     :        {fn:<10}          {tp:<10}  (FN={fn}, TP={tp})")
    print(f"{subsep}")
    print(" DETAILED CLASSIFICATION REPORT:")
    print(results["classification_report"].strip())
    print(f"{sep}\n")
