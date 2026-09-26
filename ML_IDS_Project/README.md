# ML IDS Project

A baseline machine-learning workflow for binary intrusion detection with the UNSW-NB15 dataset.

## Layout

- `dataset/`: place `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` here.
- `notebooks/ids_analysis.ipynb`: initial dataset inspection.
- `src/preprocessing.py`: dataset loading and feature preprocessing.
- `src/train.py`: train and save the baseline model.
- `src/evaluate.py`: evaluate the saved model on the testing split.
- `models/`: saved model artifacts.

The CSV files are not included. Obtain them from the official UNSW-NB15 dataset source and place them in `dataset/`.

## Requirements

Python 3.10+ with `pandas` and `scikit-learn` installed.

## Run

From this directory:

```powershell
python -m pip install pandas scikit-learn
python src/train.py
python src/evaluate.py
```

The baseline expects a binary `label` column (`0` for normal traffic and `1` for attacks). It excludes `label`, `attack_cat`, and `id` from model features to avoid target leakage and row-identifier leakage. Categorical columns are one-hot encoded; numeric columns are median-imputed and scaled. The fitted preprocessing and classifier are saved together in `models/unsw_nb15_baseline.joblib`.
