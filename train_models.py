"""
PREVENT AI - ML Model Training Pipeline
========================================
Trains 3 models (XGBoost, Random Forest, Logistic Regression) for
vector-borne disease outbreak risk prediction.

Models are trained on tabular features:
  - Climate: temperature, rainfall, humidity
  - Geographic: water proximity, vegetation index
  - Demographic: population density
  - Historical: previous month's case count
  - Temporal: month, season (encoded)
  - Disease type (encoded)

Outputs:
  - Trained model files (.joblib) in models/ directory
  - Model comparison metrics saved as JSON
  - Feature importance rankings
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

# Feature columns for the ML models
FEATURE_COLUMNS = [
    "month",
    "avg_temperature_c",
    "rainfall_mm",
    "humidity_pct",
    "water_proximity_km",
    "vegetation_index",
    "population_density",
    "previous_month_cases",
    "disease_encoded",
    "season_encoded",
]

TARGET_COLUMN = "risk_level"
RISK_LABELS = ["Low", "Medium", "High", "Critical"]


def load_and_preprocess(csv_path: str) -> tuple:
    """
    Load data, encode categoricals, and split into features/target.

    Returns:
        (X, y, label_encoders, feature_names)
    """
    print(" Loading data...")
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df):,} records")

    # Encode categorical features
    le_disease = LabelEncoder()
    le_season = LabelEncoder()
    le_target = LabelEncoder()

    df["disease_encoded"] = le_disease.fit_transform(df["disease"])
    df["season_encoded"] = le_season.fit_transform(df["season"])

    # Encode target with defined order
    le_target.fit(RISK_LABELS)
    df["risk_level_encoded"] = le_target.transform(df["risk_level"])

    X = df[FEATURE_COLUMNS].values
    y = df["risk_level_encoded"].values

    encoders = {
        "disease": le_disease,
        "season": le_season,
        "target": le_target,
    }

    print(f"   Features: {len(FEATURE_COLUMNS)}")
    print(f"   Target classes: {RISK_LABELS}")
    print(f"   Class distribution:")
    for label in RISK_LABELS:
        count = (df["risk_level"] == label).sum()
        print(f"     {label}: {count:,} ({count/len(df)*100:.1f}%)")

    return X, y, encoders, FEATURE_COLUMNS


def train_xgboost(X_train, y_train, X_test, y_test, n_classes: int) -> dict:
    """Train XGBoost classifier."""
    print("\n Training XGBoost...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")

    return {
        "name": "XGBoost",
        "model": model,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "y_pred": y_pred,
        "feature_importances": model.feature_importances_,
    }


def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    """Train Random Forest classifier."""
    print("\n Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")

    return {
        "name": "Random Forest",
        "model": model,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "y_pred": y_pred,
        "feature_importances": model.feature_importances_,
    }


def train_logistic_regression(X_train, y_train, X_test, y_test) -> dict:
    """Train Logistic Regression (as the 'Linear' model baseline)."""
    print("\n Training Logistic Regression...")

    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")

    # Use absolute coefficient magnitudes as proxy for importance
    importances = np.mean(np.abs(model.coef_), axis=0)
    importances = importances / importances.sum()

    return {
        "name": "Logistic Regression",
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "y_pred": y_pred,
        "feature_importances": importances,
    }


def save_models(results: list, encoders: dict, feature_names: list):
    """Save all trained models and metadata."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    for result in results:
        name_slug = result["name"].lower().replace(" ", "_")
        model_path = os.path.join(MODEL_DIR, f"{name_slug}.joblib")
        joblib.dump(result["model"], model_path)
        print(f"   Saved: {model_path}")

        # Save scaler for logistic regression
        if "scaler" in result:
            scaler_path = os.path.join(MODEL_DIR, "logistic_scaler.joblib")
            joblib.dump(result["scaler"], scaler_path)
            print(f"   Saved: {scaler_path}")

    # Save encoders
    encoders_path = os.path.join(MODEL_DIR, "encoders.joblib")
    joblib.dump(encoders, encoders_path)
    print(f"   Saved: {encoders_path}")

    # Save feature names
    meta = {
        "feature_columns": feature_names,
        "target_labels": RISK_LABELS,
        "trained_at": datetime.now().isoformat(),
    }
    meta_path = os.path.join(MODEL_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"   Saved: {meta_path}")

    # Save comparison metrics
    comparison = {}
    for result in results:
        comparison[result["name"]] = {
            "accuracy": round(result["accuracy"], 4),
            "f1_score": round(result["f1_score"], 4),
            "precision": round(result["precision"], 4),
            "recall": round(result["recall"], 4),
            "feature_importance": {
                name: round(float(imp), 4)
                for name, imp in zip(feature_names, result["feature_importances"])
            },
        }
    comparison_path = os.path.join(MODEL_DIR, "model_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"   Saved: {comparison_path}")


def main():
    """Full training pipeline."""
    print("=" * 60)
    print("  PREVENT AI - Model Training Pipeline")
    print("=" * 60)

    # Load data
    csv_path = os.path.join(DATA_DIR, "disease_data.csv")
    if not os.path.exists(csv_path):
        print("[ERROR] Data not found! Run generate_data.py first.")
        return

    X, y, encoders, feature_names = load_and_preprocess(csv_path)

    # Split
    print("\n Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Train all models
    n_classes = len(RISK_LABELS)
    results = [
        train_xgboost(X_train, y_train, X_test, y_test, n_classes),
        train_random_forest(X_train, y_train, X_test, y_test),
        train_logistic_regression(X_train, y_train, X_test, y_test),
    ]

    # Save everything
    print("\n Saving models & metadata...")
    save_models(results, encoders, feature_names)

    # Print final comparison
    print("\n" + "=" * 60)
    print("   MODEL COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("  " + "-" * 65)
    for r in results:
        print(
            f"  {r['name']:<25} {r['accuracy']:>10.4f} {r['f1_score']:>10.4f} "
            f"{r['precision']:>10.4f} {r['recall']:>10.4f}"
        )

    best = max(results, key=lambda x: x["f1_score"])
    print(f"\n   Best Model: {best['name']} (F1: {best['f1_score']:.4f})")

    # Feature importance for best model
    print(f"\n   Feature Importance ({best['name']}):")
    importances = list(zip(feature_names, best["feature_importances"]))
    importances.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importances:
        bar = "#" * int(imp * 50)
        print(f"    {name:<25} {imp:.4f}  {bar}")

    print(f"\n{'=' * 60}")
    print("  Training complete! Run: streamlit run app.py")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
