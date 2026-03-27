#!/usr/bin/env python3
"""
Sentiment Analysis — Command Line Interface
Usage:
    python cli.py                         # interactive mode
    python cli.py --text "Great product!" # single prediction
    python cli.py --train                 # retrain on sample data
    python cli.py --train --csv data.csv  # train on your own CSV
    python cli.py --compare               # compare all model combos
"""

import argparse
import csv
import os
import sys

from sentiment_model import SentimentAnalyser, SAMPLE_DATA

MODEL_PATH = "sentiment_model.pkl"

# ── ANSI Colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

LABEL_COLORS = {
    "positive": GREEN,
    "negative": RED,
    "neutral":  YELLOW,
}

EMOJI = {
    "positive": "😊",
    "negative": "😠",
    "neutral":  "😐",
}

BAR = "█"


def bar_chart(prob_map: dict, width: int = 30) -> str:
    lines = []
    for label, prob in sorted(prob_map.items(), key=lambda x: -x[1]):
        filled = int(prob * width)
        color  = LABEL_COLORS.get(label, CYAN)
        bar    = f"{color}{BAR * filled}{RESET}{'░' * (width - filled)}"
        lines.append(f"  {label:<10} {bar}  {prob*100:5.1f}%")
    return "\n".join(lines)


def display_prediction(result: dict, text: str):
    label = result["label"]
    conf  = result["confidence"]
    color = LABEL_COLORS.get(label, CYAN)
    emoji = EMOJI.get(label, "")

    print(f"\n{'─'*55}")
    print(f"  Input   : {CYAN}{text[:80]}{RESET}")
    print(f"  Result  : {color}{BOLD}{label.upper()} {emoji}{RESET}  "
          f"(confidence: {conf*100:.1f}%)")
    print(f"\n  Class probabilities:")
    print(bar_chart(result["probabilities"]))
    print(f"{'─'*55}\n")


# ── Load / train model ────────────────────────────────────────────────────────

def load_or_train(
    force_train: bool = False,
    csv_path: str | None = None,
    classifier: str = "logistic_regression",
    vectorizer: str = "tfidf",
) -> SentimentAnalyser:

    if not force_train and os.path.exists(MODEL_PATH):
        try:
            return SentimentAnalyser.load(MODEL_PATH)
        except Exception:
            print("⚠  Could not load saved model — retraining …")

    # Load data
    if csv_path:
        texts, labels = load_csv(csv_path)
    else:
        texts  = SAMPLE_DATA["texts"]
        labels = SAMPLE_DATA["labels"]

    model = SentimentAnalyser(classifier=classifier, vectorizer=vectorizer)
    model.train(texts, labels)
    model.save(MODEL_PATH)
    return model


def load_csv(path: str):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Accept common column names
            text  = row.get("text") or row.get("review") or row.get("tweet") or ""
            label = row.get("label") or row.get("sentiment") or row.get("target") or ""
            if text and label:
                texts.append(text.strip())
                labels.append(label.strip().lower())
    print(f"✓ Loaded {len(texts)} rows from '{path}'")
    return texts, labels


# ── Compare all model combinations ────────────────────────────────────────────

def compare_models():
    texts  = SAMPLE_DATA["texts"]
    labels = SAMPLE_DATA["labels"]

    combos = [
        ("naive_bayes",         "tfidf"),
        ("naive_bayes",         "count"),
        ("logistic_regression", "tfidf"),
        ("logistic_regression", "count"),
    ]

    print(f"\n{'='*60}")
    print(f"  Model Comparison")
    print(f"{'='*60}")
    print(f"  {'Classifier':<25} {'Vectorizer':<8} {'Accuracy':>10} {'F1 (W)':>10}")
    print(f"  {'─'*55}")

    best, best_acc = None, 0
    for clf, vec in combos:
        m = SentimentAnalyser(classifier=clf, vectorizer=vec)
        metrics = m.train(texts, labels)
        acc = metrics["accuracy"]
        f1  = metrics["f1_weighted"]
        marker = " ◄ best" if acc > best_acc else ""
        if acc > best_acc:
            best_acc = acc
            best = (clf, vec)
        print(f"  {clf:<25} {vec:<8} {acc:>10.4f} {f1:>10.4f}{marker}")

    print(f"\n  🏆  Best: {best[0]} + {best[1]}  ({best_acc:.4f} accuracy)\n")


# ── Interactive REPL ──────────────────────────────────────────────────────────

def interactive(model: SentimentAnalyser):
    print(f"\n{'='*55}")
    print(f"  {BOLD}Sentiment Analysis — Interactive Mode{RESET}")
    print(f"  Type text and press Enter.  'quit' to exit.")
    print(f"{'='*55}\n")

    while True:
        try:
            user_input = input(f"{CYAN}▶ Enter text:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("\nGoodbye! 👋")
            break

        result = model.predict(user_input)
        display_prediction(result, user_input)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--text",       type=str,  help="Analyse a single text snippet")
    parser.add_argument("--train",      action="store_true", help="Force retrain the model")
    parser.add_argument("--csv",        type=str,  help="Path to CSV file (columns: text, label)")
    parser.add_argument("--classifier", type=str,  default="logistic_regression",
                        choices=["naive_bayes", "logistic_regression"])
    parser.add_argument("--vectorizer", type=str,  default="tfidf",
                        choices=["tfidf", "count"])
    parser.add_argument("--compare",    action="store_true", help="Compare all model combos")
    args = parser.parse_args()

    if args.compare:
        compare_models()
        return

    model = load_or_train(
        force_train=args.train or bool(args.csv),
        csv_path=args.csv,
        classifier=args.classifier,
        vectorizer=args.vectorizer,
    )

    if args.text:
        result = model.predict(args.text)
        display_prediction(result, args.text)
    else:
        interactive(model)


if __name__ == "__main__":
    main()