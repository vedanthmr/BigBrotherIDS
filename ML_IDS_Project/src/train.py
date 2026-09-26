"""Train and persist the baseline UNSW-NB15 classifier."""

import argparse
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_DATA = PROJECT_ROOT / "dataset" / "UNSW_NB15_training-set.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "unsw_nb15_baseline.joblib"


def train(training_path: Path, model_path: Path) -> None:
    features, target = load_dataset(training_path)
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(features)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(features, target)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Trained on {len(features):,} rows; saved model to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_TRAINING_DATA, help="Training CSV path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Output model path")
    args = parser.parse_args()
    train(args.data, args.model)


if __name__ == "__main__":
    main()
