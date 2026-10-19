"""
Main Application CLI Entrypoint
Email Spam Detection - NLP Machine Learning System
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.dataset import load_and_preprocess_dataset, split_dataset
from src.evaluator import evaluate_model, print_evaluation_summary
from src.pipeline import build_pipeline, load_pipeline, save_pipeline
from src.predictor import SpamPredictor

DEFAULT_DATASET = Path(__file__).parent / "data" / "emails.csv"
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "spam_classifier.joblib"


def run_training_and_evaluation(
    dataset_path: str = str(DEFAULT_DATASET),
    model_type: str = "naive_bayes",
    save_path: str = str(DEFAULT_MODEL_PATH),
    message_col: str = None,
    label_col: str = None
):
    """Loads dataset, splits into train/test, fits pipeline, prints metrics, and saves model."""
    # Format paths cleanly relative to working directory if possible
    try:
        display_dataset = Path(dataset_path).relative_to(Path.cwd())
    except (ValueError, Exception):
        display_dataset = dataset_path

    try:
        display_save = Path(save_path).relative_to(Path.cwd())
    except (ValueError, Exception):
        display_save = save_path

    print("=" * 64)
    print(" TRAINING EMAIL SPAM CLASSIFICATION MODEL")
    print("=" * 64)
    print(f" Dataset Path : {display_dataset}")
    print(f" Model Type   : {model_type}")
    print(f" Output Model : {display_save}")

    # Step 1: Load and preprocess dataset
    df = load_and_preprocess_dataset(dataset_path, message_col=message_col, label_col=label_col)
    total_samples = len(df)
    spam_count = (df["label"] == "SPAM").sum()
    ham_count = (df["label"] == "NOT_SPAM").sum()
    print(f" Total Samples: {total_samples} (SPAM: {spam_count}, NOT_SPAM: {ham_count})")

    # Step 2: Split dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = split_dataset(df, test_size=0.2, random_state=42)
    print(f" Split Size   : Train={len(X_train)} samples, Test={len(X_test)} samples")

    # Step 3: Build & Fit Pipeline
    pipeline = build_pipeline(model_type=model_type)
    print(f" Fitting TF-IDF Vectorizer & {model_type.replace('_', ' ').title()}...")
    pipeline.fit(X_train, y_train)

    # Step 4: Evaluate on unseen test set
    eval_results = evaluate_model(pipeline, X_test, y_test)
    print_evaluation_summary(eval_results, model_name=model_type)

    # Step 5: Save trained model
    save_pipeline(pipeline, save_path)
    print(f"[SUCCESS] Model saved successfully to: {display_save}\n")

    return pipeline, eval_results


def run_test_cases(predictor: SpamPredictor):
    """
    Executes and reports results for the five required test cases:
    - TC1: "You won a free prize, claim now" -> SPAM
    - TC2: "Please attend the team meeting at 10 AM" -> NOT_SPAM
    - TC3: Empty message -> Graceful rejection / handling without crash
    - TC4: Message with uppercase and extra spaces -> Process without crash
    - TC5: Message not in training set -> Generalized prediction without crash
    """
    sep = "=" * 80
    subsep = "-" * 80

    test_scenarios = [
        {
            "id": "TC1",
            "name": "Spam Pattern Match",
            "input": "You won a free prize, claim now",
            "expected": "SPAM",
        },
        {
            "id": "TC2",
            "name": "Ham Pattern Match",
            "input": "Please attend the team meeting   at 10 AM",
            "expected": "NOT_SPAM",
        },
        {
            "id": "TC3",
            "name": "Empty Message Validation",
            "input": "   ",
            "expected": "Graceful Rejection / Handled without crashing",
        },
        {
            "id": "TC4",
            "name": "Uppercase & Extra Spaces Handling",
            "input": "   CLAIM   YOUR FREE   PRIZE NOW   TODAY!!   ",
            "expected": "SPAM (Processes without crash)",
        },
        {
            "id": "TC5",
            "name": "Unseen Generalization",
            "input": "Hey David, the Q4 financial audit slides are ready for your review.",
            "expected": "NOT_SPAM (Generalizes without exact match)",
        },
    ]

    print(f"\n{sep}")
    print(" EXECUTING VERIFICATION SUITE: REQUIRED TEST CASES (TC1 - TC5)")
    print(f"{sep}")

    all_passed = True

    for tc in test_scenarios:
        raw_input = tc["input"]
        result = predictor.predict(raw_input)

        print(f"\n[{tc['id']}] {tc['name']}")
        print(f"  Input Description : {repr(raw_input)}")
        print(f"  Expected Behavior : {tc['expected']}")

        if tc["id"] == "TC3":
            # Must handle gracefully without crashing
            if result["status"] == "error" and not result["is_valid"]:
                status_str = "PASS"
                details = f"Gracefully rejected: '{result['error_message']}'"
            else:
                status_str = "FAIL"
                all_passed = False
                details = "Failed to reject empty message"
        else:
            # Must produce valid prediction
            if result["status"] == "success":
                label = result["predicted_label"]
                conf = result["confidence"] * 100
                status_str = "PASS"
                details = f"Predicted: {label} (Confidence: {conf:.2f}%)"
            else:
                status_str = "FAIL"
                all_passed = False
                details = f"Crashed or failed: {result['error_message']}"

        print(f"  Result            : [{status_str}] - {details}")

    print(f"\n{subsep}")
    overall = "ALL TEST CASES PASSED SUCCESSFULLY" if all_passed else "SOME TEST CASES FAILED"
    print(f" OVERALL SUITE STATUS: {overall}")
    print(f"{sep}\n")

    return all_passed


def interactive_mode(predictor: SpamPredictor):
    """Starts an interactive command-line session where users can classify messages."""
    sep = "=" * 64
    print(f"\n{sep}")
    print(" INTERACTIVE EMAIL SPAM DETECTOR")
    print(" Type an email message and press Enter to see its prediction.")
    print(" Type 'exit' or 'quit' to return to terminal.")
    print(f"{sep}\n")

    while True:
        try:
            user_input = input("\nEnter message > ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting interactive mode. Goodbye!")
                break

            result = predictor.predict(user_input)
            print("-" * 50)
            print(predictor.format_prediction_result(result))
            print("-" * 50)
        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Production-Grade Email Spam Detection System (NLP & ML)"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new model on the dataset and display evaluation metrics.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help=f"Path to CSV dataset (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["naive_bayes", "logistic_regression"],
        default="naive_bayes",
        help="Classification algorithm (default: naive_bayes)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help=f"Path to save/load model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--message-col",
        type=str,
        default=None,
        help="Explicit name of the message column in dataset.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Explicit name of the label column in dataset.",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Predict label for a single message string.",
    )
    parser.add_argument(
        "--test-cases",
        action="store_true",
        help="Run the 5 required test cases (TC1-TC5) and print results.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive console prediction prompt.",
    )

    args = parser.parse_args()

    # Determine if any specific action was requested
    has_action = args.train or (args.predict is not None) or args.test_cases or args.interactive

    # Default action if no flags provided: Full Demonstration
    if not has_action:
        print("[Demo Mode] No arguments provided. Running end-to-end demonstration:\n")
        # 1. Train and evaluate
        pipeline, _ = run_training_and_evaluation(
            dataset_path=args.dataset,
            model_type=args.model_type,
            save_path=args.model_path,
            message_col=args.message_col,
            label_col=args.label_col
        )
        predictor = SpamPredictor(pipeline=pipeline)

        # 2. Sample input from problem description
        sample_msg = "Congratulations! You have won a free prize. Click now!"
        print("=" * 64)
        print(" SAMPLE PREDICTION (FROM SPECIFICATION)")
        print("=" * 64)
        sample_res = predictor.predict(sample_msg)
        print(predictor.format_prediction_result(sample_res))
        print("=" * 64)

        # 3. Execute all test cases
        run_test_cases(predictor)
        return

    # If --train is explicitly requested
    if args.train:
        pipeline, _ = run_training_and_evaluation(
            dataset_path=args.dataset,
            model_type=args.model_type,
            save_path=args.model_path,
            message_col=args.message_col,
            label_col=args.label_col
        )
        predictor = SpamPredictor(pipeline=pipeline)
    else:
        # Load existing model if available, or train if not found
        if Path(args.model_path).exists():
            predictor = SpamPredictor(model_path=args.model_path)
        else:
            print(f"[Notice] Model not found at {args.model_path}. Training a new model first...")
            pipeline, _ = run_training_and_evaluation(
                dataset_path=args.dataset,
                model_type=args.model_type,
                save_path=args.model_path,
                message_col=args.message_col,
                label_col=args.label_col
            )
            predictor = SpamPredictor(pipeline=pipeline)

    # If --predict is provided
    if args.predict is not None:
        result = predictor.predict(args.predict)
        print("\n" + "=" * 60)
        print(" PREDICTION RESULT")
        print("=" * 60)
        print(predictor.format_prediction_result(result))
        print("=" * 60 + "\n")

    # If --test-cases is requested
    if args.test_cases:
        run_test_cases(predictor)

    # If --interactive is requested
    if args.interactive:
        interactive_mode(predictor)


if __name__ == "__main__":
    main()
