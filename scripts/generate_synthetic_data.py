"""
Synthetic Dataset Generator Script
Generates balanced, customizable synthetic email datasets for spam classification experiments.
"""

import argparse
import random
from pathlib import Path
import pandas as pd

SPAM_TEMPLATES = [
    "Congratulations! You won a ${amount} {store} gift card. Click {link} now to claim!",
    "URGENT: Your account has been suspended. Verify your credentials at {link} immediately.",
    "Claim your free {item} today! Limited stock available. Call {phone} now.",
    "Earn up to ${amount} per day working from home. No experience needed. Visit {link}.",
    "Exclusive {discount}% discount on luxury designer products! Order now at {link}.",
    "Winner! You have been selected for a ${amount} cash prize. Text WIN to {phone}.",
    "Immediate action required: Unclaimed tax refund of ${amount} waiting. Click {link}.",
    "Double your crypto portfolio with guaranteed 500% returns in 24 hours. Sign up: {link}.",
    "Final notice: Your subscription will be cancelled unless you renew at {link}.",
    "You have 1 pending parcel delivery. Update your address at {link} to receive your package."
]

HAM_TEMPLATES = [
    "Hi team, please find the meeting notes and action items from today's sync attached.",
    "Can you please review the pull request on GitHub before our standup tomorrow?",
    "Reminder: The quarterly planning session is scheduled for {day} at {time}.",
    "Are we still on for lunch today at {time}? Let me know what you prefer.",
    "Please submit your completed expense reports and timesheets by end of week.",
    "Attached is the updated project roadmap and architecture diagram for Q{quarter}.",
    "I will be working remotely on {day} due to a doctor's appointment in the morning.",
    "Thanks for the productive discussion earlier. Let's touch base again on {day}.",
    "The deployment to staging was successful. QA verification can begin shortly.",
    "Could you send me the latest metrics dashboard export for the client presentation?"
]

STORES = ["Amazon", "Walmart", "Target", "Apple", "Costco"]
ITEMS = ["iPhone 15 Pro", "MacBook Air", "PlayStation 5", "Voucher", "Trip to Bahamas"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIMES = ["9:30 AM", "10:00 AM", "11:15 AM", "2:00 PM", "3:30 PM", "4:00 PM"]
LINKS = ["http://promo-win.net", "http://claim-bonus.xyz", "http://verify-login.online", "http://gift-card.top"]
PHONES = ["555-0199", "800-555-2345", "77234", "88210"]


def generate_synthetic_dataset(num_samples: int = 100) -> pd.DataFrame:
    """Generates synthetic spam and ham records."""
    records = []
    half = num_samples // 2

    for _ in range(half):
        template = random.choice(SPAM_TEMPLATES)
        text = template.format(
            amount=random.choice([500, 1000, 2500, 5000, 10000]),
            store=random.choice(STORES),
            item=random.choice(ITEMS),
            discount=random.choice([50, 70, 80, 90]),
            link=random.choice(LINKS),
            phone=random.choice(PHONES)
        )
        records.append({"message": text, "label": "SPAM"})

    for _ in range(num_samples - half):
        template = random.choice(HAM_TEMPLATES)
        text = template.format(
            day=random.choice(DAYS),
            time=random.choice(TIMES),
            quarter=random.choice([1, 2, 3, 4])
        )
        records.append({"message": text, "label": "NOT_SPAM"})

    random.shuffle(records)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic email dataset.")
    parser.add_argument("--count", type=int, default=100, help="Number of records to generate.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent.parent / "data" / "synthetic_emails.csv"),
        help="Target CSV output path."
    )
    args = parser.parse_args()

    df = generate_synthetic_dataset(args.count)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[Synthetic Data] Successfully generated {len(df)} samples at: {args.output}")


if __name__ == "__main__":
    main()
