"""Evaluate a saved UNSW-NB15 classifier on the testing split."""

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTING_DATA = PROJECT_ROOT / "dataset" / "UNSW_NB15_testing-set.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "unsw_nb15_baseline.joblib"


def evaluate(testing_path: Path, model_path: Path) -> None:
    features, target = load_dataset(testing_path)
    model = joblib.load(model_path)
    predictions = model.predict(features)

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(target, predictions))
    print("\nClassification report:")
    print(classification_report(target, predictions, zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_TESTING_DATA, help="Testing CSV path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Trained model path")
    args = parser.parse_args()
    evaluate(args.data, args.model)


if __name__ == "__main__":
    main()
