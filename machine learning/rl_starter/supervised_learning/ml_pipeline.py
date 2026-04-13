"""
ml_pipeline.py — End-to-end supervised learning pipeline.

Trains and compares multiple algorithms, tunes the best one,
and produces evaluation plots.  Works on any sklearn built-in
dataset or CSV file.

Usage:
    python ml_pipeline.py --dataset iris
    python ml_pipeline.py --csv mydata.csv --target label
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
    load_digits,
    load_diabetes,
    load_iris,
    load_wine,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

BUILTIN_DATASETS = {
    "iris": (load_iris, "classification"),
    "wine": (load_wine, "classification"),
    "digits": (load_digits, "classification"),
    "diabetes": (load_diabetes, "regression"),
}

OUTPUT_DIR = Path("outputs")


# ── data loading ────────────────────────────────────────────────


def load_builtin(name: str) -> tuple[pd.DataFrame, str, str]:
    loader, task = BUILTIN_DATASETS[name]
    bunch = loader()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    return df, "target", task


def load_csv(path: str, target: str) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path)
    if target not in df.columns:
        sys.exit(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    nunique = df[target].nunique()
    task = "classification" if nunique <= 20 or df[target].dtype == object else "regression"
    return df, target, task


# ── preprocessing ───────────────────────────────────────────────


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))
    return ColumnTransformer(transformers, remainder="passthrough")


# ── models ──────────────────────────────────────────────────────


def get_models(task: str) -> dict[str, object]:
    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
            "SVM (RBF)": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBClassifier(
                n_estimators=200, use_label_encoder=False, eval_metric="mlogloss", random_state=42
            )
    else:
        models = {
            "Ridge Regression": Ridge(),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
            "SVR (RBF)": SVR(),
            "KNN": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
        }
        if HAS_XGB:
            models["XGBoost"] = XGBRegressor(n_estimators=200, random_state=42)
    return models


PARAM_GRIDS: dict[str, dict] = {
    "SVM (RBF)": {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", "auto"]},
    "Random Forest": {"model__n_estimators": [100, 300], "model__max_depth": [None, 10, 20]},
    "Gradient Boosting": {
        "model__n_estimators": [100, 300],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__max_depth": [3, 5],
    },
    "SVR (RBF)": {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", "auto"]},
    "Ridge Regression": {"model__alpha": [0.01, 0.1, 1, 10, 100]},
}


# ── evaluation ──────────────────────────────────────────────────


def cross_validate_models(
    models: dict, preprocessor: ColumnTransformer, X: pd.DataFrame, y, task: str
) -> dict[str, tuple[float, float]]:
    scoring = "accuracy" if task == "classification" else "r2"
    cv = StratifiedKFold(5, shuffle=True, random_state=42) if task == "classification" else 5
    results: dict[str, tuple[float, float]] = {}
    for name, model in models.items():
        pipe = Pipeline([("pre", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = (scores.mean(), scores.std())
    return results


def print_results(results: dict[str, tuple[float, float]], task: str) -> str:
    metric_name = "accuracy" if task == "classification" else "R²"
    best_name = max(results, key=lambda k: results[k][0])
    print(f"\nCross-Validation Results (5-fold {metric_name}):")
    for name, (mean, std) in sorted(results.items(), key=lambda x: -x[1][0]):
        star = "  ★ best" if name == best_name else ""
        print(f"  {name:<28s} {mean:.4f} ± {std:.4f}{star}")
    return best_name


def tune_best(
    best_name: str,
    models: dict,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train,
    task: str,
) -> Pipeline:
    pipe = Pipeline([("pre", preprocessor), ("model", models[best_name])])
    grid = PARAM_GRIDS.get(best_name)
    if grid:
        scoring = "accuracy" if task == "classification" else "r2"
        cv = StratifiedKFold(5, shuffle=True, random_state=42) if task == "classification" else 5
        print(f"\nTuning {best_name} …")
        gs = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        print(f"  Best params: {gs.best_params_}")
        return gs.best_estimator_
    pipe.fit(X_train, y_train)
    return pipe


# ── plots ───────────────────────────────────────────────────────


def save_confusion_matrix(pipe: Pipeline, X_test, y_test, labels, tag: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, display_labels=labels, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / f'{tag}_confusion_matrix.png'}")


def save_roc_curves(pipe: Pipeline, X_test, y_test, labels, tag: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    if len(labels) > 2 and not hasattr(pipe, "predict_proba"):
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_roc.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / f'{tag}_roc.png'}")


def save_feature_importance(pipe: Pipeline, feature_names: list[str], tag: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        return
    # get transformed feature names
    pre = pipe.named_steps["pre"]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = feature_names
    if len(names) != len(importances):
        names = [f"f{i}" for i in range(len(importances))]
    idx = np.argsort(importances)[-20:]  # top 20
    fig, ax = plt.subplots(figsize=(8, max(4, len(idx) * 0.35)))
    ax.barh(range(len(idx)), importances[idx], color="#2bb0a7")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([names[i] for i in idx])
    ax.set_title("Feature Importance (top 20)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / f'{tag}_feature_importance.png'}")


def save_comparison_chart(results: dict[str, tuple[float, float]], task: str, tag: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    names = list(results.keys())
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, means, xerr=stds, color="#2bb0a7", capsize=4)
    metric = "Accuracy" if task == "classification" else "R²"
    ax.set_xlabel(metric)
    ax.set_title("Model Comparison (5-fold CV)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / f'{tag}_comparison.png'}")


# ── main ────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised ML Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=list(BUILTIN_DATASETS.keys()), help="Built-in dataset")
    group.add_argument("--csv", type=str, help="Path to CSV file")
    parser.add_argument("--target", type=str, default="target", help="Target column name (for CSV)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # load data
    if args.dataset:
        df, target, task = load_builtin(args.dataset)
        tag = args.dataset
    else:
        df, target, task = load_csv(args.csv, args.target)
        tag = Path(args.csv).stem

    X = df.drop(columns=[target])
    y = df[target].values

    if task == "classification" and y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    print("═" * 55)
    print(f"  ML Pipeline — {tag} ({task})")
    print("═" * 55)
    n_classes = len(np.unique(y)) if task == "classification" else None
    info = f"Dataset: {len(df)} samples, {X.shape[1]} features"
    if n_classes:
        info += f", {n_classes} classes"
    print(info)

    # split
    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )

    preprocessor = build_preprocessor(X_train)
    models = get_models(task)

    # cross-validate all
    results = cross_validate_models(models, preprocessor, X_train, y_train, task)
    best_name = print_results(results, task)

    # tune & evaluate
    best_pipe = tune_best(best_name, models, preprocessor, X_train, y_train, task)
    y_pred = best_pipe.predict(X_test)

    print(f"\nTest Evaluation ({best_name}):")
    if task == "classification":
        print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"\n{classification_report(y_test, y_pred)}")
        labels = BUILTIN_DATASETS.get(tag, (None, None))[0]
        label_names = labels().target_names if labels else [str(c) for c in np.unique(y)]
        save_confusion_matrix(best_pipe, X_test, y_test, label_names, tag)
        save_roc_curves(best_pipe, X_test, y_test, label_names, tag)
    else:
        print(f"  R²:  {r2_score(y_test, y_pred):.4f}")
        print(f"  MAE: {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    save_feature_importance(best_pipe, X.columns.tolist(), tag)
    save_comparison_chart(results, task, tag)
    print("\nDone ✓")


if __name__ == "__main__":
    main()
