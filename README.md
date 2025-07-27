# Iris Classifier (k-NN)

A simple machine learning project that trains a **k-Nearest Neighbors** classifier on the classic **Iris dataset**.

## What it does
- Loads and preprocesses the Iris dataset
- Splits into train/test sets
- Trains a k-NN model
- Evaluates with accuracy & classification report
- (Optional) Visualizes predictions & confusion matrix

## How to run

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # or activate.bat on CMD
pip install -r requirements.txt
python src/day4_iris_classifier.py
