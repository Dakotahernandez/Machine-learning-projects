# Supervised Learning — Classic ML Pipeline

End-to-end supervised learning with **scikit-learn**: automated data preprocessing, model comparison, hyperparameter tuning, and evaluation across classification and regression tasks.

## What's Inside

### `ml_pipeline.py` — Full ML Pipeline
- Automated feature engineering (scaling, encoding, imputation)
- Trains & compares **8 algorithms** side-by-side:
  - Logistic Regression / Ridge Regression
  - Random Forest, Gradient Boosting, XGBoost
  - SVM, KNN, Decision Tree
- Stratified K-Fold cross-validation
- Grid search hyperparameter tuning on the best model
- Confusion matrix, ROC curves, feature importance plots
- Works on **any** CSV dataset (auto-detects classification vs regression)

### `ensemble_stacking.py` — Advanced Ensemble Methods
- Stacking classifier/regressor with meta-learner
- Voting ensembles (hard + soft)
- Compares ensemble vs individual model performance

## Quick Start

```powershell
pip install -r requirements.txt

# Run full pipeline on built-in datasets
python ml_pipeline.py --dataset iris
python ml_pipeline.py --dataset wine
python ml_pipeline.py --dataset digits
python ml_pipeline.py --dataset diabetes    # regression

# Run on your own CSV
python ml_pipeline.py --csv data.csv --target price

# Stacking ensembles
python ensemble_stacking.py --dataset iris
```

## Sample Output

```
═══════════════════════════════════════════════════
  ML Pipeline — iris (classification)
═══════════════════════════════════════════════════
Dataset: 150 samples, 4 features, 3 classes

Cross-Validation Results (5-fold):
  Logistic Regression    0.967 ± 0.021
  Random Forest          0.960 ± 0.033
  Gradient Boosting      0.953 ± 0.028
  SVM (RBF)              0.973 ± 0.018  ★ best
  KNN                    0.960 ± 0.033
  Decision Tree          0.947 ± 0.040
  XGBoost                0.960 ± 0.033

Tuning best model (SVM) ...
  Best params: {'C': 10, 'gamma': 'scale'}
  Test accuracy: 0.978
```
