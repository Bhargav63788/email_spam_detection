"""
Dataset Loader and Splitter Module
Handles CSV ingestion, automated column detection, label normalization,
text pre-cleaning, and stratified train-test splitting.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessor import clean_text

MESSAGE_COLUMN_CANDIDATES = [
    "message", "text", "email", "sms", "content", "body", "v2", "sms_message"
]

LABEL_COLUMN_CANDIDATES = [
    "label", "target", "category", "class", "type", "v1", "classification"
]

SPAM_EQUIVALENTS = {"spam", "1", "true", "junk", "phishing", "fraud"}
HAM_EQUIVALENTS = {"ham", "0", "false", "not_spam", "not spam", "legit", "valid", "normal"}


def find_column_by_candidates(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Finds the first matching column name in DataFrame case-insensitively."""
    df_cols_lower = {col.lower().strip(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df_cols_lower:
            return df_cols_lower[candidate]
    return None


def normalize_label(val: any) -> Optional[str]:
    """Standardizes target labels into 'SPAM' or 'NOT_SPAM'."""
    if pd.isna(val):
        return None
    val_str = str(val).strip().lower()
    if val_str in SPAM_EQUIVALENTS:
        return "SPAM"
    if val_str in HAM_EQUIVALENTS:
        return "NOT_SPAM"
    # Default fallback heuristics
    if "spam" in val_str:
        return "SPAM"
    return "NOT_SPAM"


def load_and_preprocess_dataset(
    csv_path: str,
    message_col: Optional[str] = None,
    label_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Loads CSV dataset, validates required columns, cleans text, and normalizes labels.

    Args:
        csv_path (str): File path to CSV.
        message_col (Optional[str]): Explicit message column name.
        label_col (Optional[str]): Explicit label column name.

    Returns:
        pd.DataFrame: Cleaned DataFrame with standard columns ['message', 'label'].
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    # Read CSV supporting UTF-8 and latin-1 fallback
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    if df.empty:
        raise ValueError(f"Dataset at {csv_path} is empty.")

    # Detect message column
    resolved_msg_col = message_col or find_column_by_candidates(df, MESSAGE_COLUMN_CANDIDATES)
    if not resolved_msg_col or resolved_msg_col not in df.columns:
        raise KeyError(
            f"Could not automatically detect message column. "
            f"Available columns: {list(df.columns)}. "
            f"Please specify --message-col."
        )

    # Detect label column
    resolved_label_col = label_col or find_column_by_candidates(df, LABEL_COLUMN_CANDIDATES)
    if not resolved_label_col or resolved_label_col not in df.columns:
        raise KeyError(
            f"Could not automatically detect label column. "
            f"Available columns: {list(df.columns)}. "
            f"Please specify --label-col."
        )

    # Extract relevant columns
    cleaned_df = pd.DataFrame()
    cleaned_df["message"] = df[resolved_msg_col].apply(clean_text)
    cleaned_df["label"] = df[resolved_label_col].apply(normalize_label)

    # Drop missing labels or empty messages
    initial_count = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(subset=["label"])
    cleaned_df = cleaned_df[cleaned_df["message"].str.len() > 0].reset_index(drop=True)

    dropped_count = initial_count - len(cleaned_df)
    if dropped_count > 0:
        print(f"[Dataset] Dropped {dropped_count} invalid or empty rows during loading.")

    if len(cleaned_df) < 5:
        raise ValueError("Dataset has insufficient valid records (< 5) to train.")

    return cleaned_df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits dataset into stratified train and test sets.

    Args:
        df (pd.DataFrame): DataFrame containing 'message' and 'label'.
        test_size (float): Fraction of dataset to reserve for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    X = df["message"]
    y = df["label"]

    # Stratified split to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
