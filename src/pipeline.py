"""
Pipeline Module
Builds, trains, saves, and loads Scikit-Learn NLP classification pipelines.
Combines TF-IDF Vectorization with Multinomial Naive Bayes or Logistic Regression.
"""

from pathlib import Path
from typing import Literal, Optional
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ModelType = Literal["naive_bayes", "logistic_regression"]


def build_pipeline(
    model_type: ModelType = "naive_bayes",
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    alpha: float = 0.1,
    c_param: float = 1.0,
    random_state: int = 42
) -> Pipeline:
    """
    Constructs an end-to-end NLP classification pipeline:
    TfidfVectorizer -> Classifier (MultinomialNB or LogisticRegression).

    Args:
        model_type (ModelType): Classifier choice ('naive_bayes' or 'logistic_regression').
        ngram_range (tuple): Range of n-grams (unigrams + bigrams for rich context).
        min_df (int): Minimum document frequency.
        alpha (float): Laplace smoothing parameter for MultinomialNB.
        c_param (float): Inverse regularization parameter for LogisticRegression.
        random_state (int): Seed for reproducibility.

    Returns:
        Pipeline: Scikit-learn Pipeline instance ready for training.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True,
        stop_words="english"
    )

    if model_type == "naive_bayes":
        classifier = MultinomialNB(alpha=alpha)
    elif model_type == "logistic_regression":
        classifier = LogisticRegression(
            C=c_param,
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced"
        )
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Choose 'naive_bayes' or 'logistic_regression'."
        )

    pipeline = Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("classifier", classifier)
        ]
    )

    return pipeline


def save_pipeline(pipeline: Pipeline, file_path: str) -> None:
    """Serializes and saves trained pipeline to disk."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(file_path: str) -> Pipeline:
    """Loads a serialized pipeline from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found at: {file_path}")
    pipeline = joblib.load(path)
    return pipeline
