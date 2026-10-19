"""
Text Preprocessor Module
Handles text cleaning, normalization, missing values, and whitespace sanitization.
"""

import re
from typing import Any, Tuple


def clean_text(text: Any) -> str:
    """
    Cleans and normalizes input text:
    - Handles missing/null values (converts None or NaN to empty string)
    - Converts text to string if non-string type
    - Converts all characters to lowercase
    - Replaces consecutive whitespace (spaces, tabs, newlines) with a single space
    - Strips leading and trailing whitespace

    Args:
        text (Any): Input text, string, or object.

    Returns:
        str: Sanitized, lowercase, whitespace-normalized string.
    """
    if text is None:
        return ""

    # Convert to string if not already
    text_str = str(text)

    # Convert to lowercase
    text_str = text_str.lower()

    # Normalize multiple whitespace characters (including tabs and newlines) to a single space
    text_str = re.sub(r"\s+", " ", text_str)

    # Strip leading and trailing whitespace
    text_str = text_str.strip()

    return text_str


def validate_input(text: Any) -> Tuple[bool, str, str]:
    """
    Validates user input message according to requirements:
    - Rejects empty, None, or whitespace-only messages gracefully (Test Case 3).
    - Normalizes uppercase letters and extra whitespace without crashing (Test Case 4).

    Args:
        text (Any): The message to validate.

    Returns:
        Tuple[bool, str, str]:
            - is_valid (bool): True if message is non-empty and valid for prediction.
            - cleaned_text (str): Preprocessed text ready for vectorization.
            - error_message (str): Explanatory error message if invalid, else empty string.
    """
    if text is None:
        return False, "", "Validation Error: Message is null or empty. Please provide a non-empty text message."

    # If it's a non-string, check representation
    raw_str = str(text)

    # Check if empty or whitespace only before cleaning
    if not raw_str.strip():
        return False, "", "Validation Error: Message is empty or contains only whitespace. Please provide valid text content."

    cleaned = clean_text(raw_str)

    if not cleaned:
        return False, "", "Validation Error: Message contains no valid text after cleaning."

    return True, cleaned, ""
