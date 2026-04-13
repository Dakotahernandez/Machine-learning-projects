"""
ensemble_stacking.py — Advanced ensemble methods: stacking & voting.

Demonstrates stacking classifiers/regressors with a meta-learner
and compares them to voting ensembles and individual models.

Usage:
    python ensemble_stacking.py --dataset iris
    python ensemble_stacking.py --dataset diabetes
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris, load_wine, load_digits, load_diabetes
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score

DATASETS = {
    "iris": (load_iris, "classification"),
    "wine": (load_wine, "classification"),
    "digits": (load_digits, "classification"),
    "diabetes": (load_diabetes, "regression"),
}

OUTPUT_DIR = Path("outputs")


def build_classifiers():
    base = [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("svm", Pipeline([("scale", StandardScaler()), ("svc", SVC(probability=True, random_state=42))])),
        ("knn", Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier())])),
    ]
    stacker = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1,
    )
    voter_hard = VotingClassifier(estimators=base, voting="hard", n_jobs=-1)
    voter_soft = VotingClassifier(estimators=base, voting="soft", n_jobs=-1)
    return {
        "Random Forest": base[0][1],
        "Gradient Boosting": base[1][1],
        "SVM": base[2][1],
        "KNN": base[3][1],
        "Stacking (LR meta)": stacker,
        "Voting (hard)": voter_hard,
        "Voting (soft)": voter_soft,
    }


def build_regressors():
    base = [
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ("svr", Pipeline([("scale", StandardScaler()), ("svr", SVR())])),
        ("knn", Pipeline([("scale", StandardScaler()), ("knn", KNeighborsRegressor())])),
    ]
    stacker = StackingRegressor(
        estimators=base,
        final_estimator=Ridge(),
        cv=5,
        n_jobs=-1,
    )
    voter = VotingRegressor(estimators=base, n_jobs=-1)
    return {
        "Random Forest": base[0][1],
        "Gradient Boosting": base[1][1],
        "SVR": base[2][1],
        "KNN": base[3][1],
        "Stacking (Ridge meta)": stacker,
        "Voting": voter,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble Stacking Comparison")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loader, task = DATASETS[args.dataset]
    data = loader()
    X, y = data.data, data.target

    if task == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed, stratify=y
        )
        models = build_classifiers()
        scoring = "accuracy"
        cv = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=args.seed
        )
        models = build_regressors()
        scoring = "r2"
        cv = 5

    print("═" * 55)
    print(f"  Ensemble Comparison — {args.dataset} ({task})")
    print("═" * 55)

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = (scores.mean(), scores.std())
        marker = "  ★" if "Stacking" in name else ""
        print(f"  {name:<28s} {scores.mean():.4f} ± {scores.std():.4f}{marker}")

    # fit best ensemble on full train, evaluate on test
    best_name = max(results, key=lambda k: results[k][0])
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print(f"\nBest: {best_name}")
    if task == "classification":
        print(f"  Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    else:
        print(f"  Test R²: {r2_score(y_test, y_pred):.4f}")

    # plot
    OUTPUT_DIR.mkdir(exist_ok=True)
    names = list(results.keys())
    means = [results[n][0] for n in names]
    stds = [results[n][1] for n in names]
    colors = ["#f2b14c" if "Stacking" in n or "Voting" in n else "#2bb0a7" for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(names, means, xerr=stds, color=colors, capsize=4)
    metric = "Accuracy" if task == "classification" else "R²"
    ax.set_xlabel(metric)
    ax.set_title(f"Individual vs Ensemble ({args.dataset})")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc="#2bb0a7", label="Individual"),
            plt.Rectangle((0, 0), 1, 1, fc="#f2b14c", label="Ensemble"),
        ],
        labels=["Individual", "Ensemble"],
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{args.dataset}_ensemble.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved {OUTPUT_DIR / f'{args.dataset}_ensemble.png'}")
    print("Done ✓")


if __name__ == "__main__":
    main()
