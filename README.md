# 🚢 Titanic Survival Prediction

Binary classification predicting Titanic passenger survival using scikit-learn. Includes full EDA, feature engineering, and multi-model ROC analysis.

## Overview
| Detail | Value |
|--------|-------|
| Type | Binary Classification |
| Dataset | Titanic (seaborn built-in, 891 passengers) |
| Framework | scikit-learn |
| Models | Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN |

## Getting Started
```bash
git clone https://github.com/Dnshitobu/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
python titanic_survival.py
```

## What It Does
1. EDA — survival rates by class, sex, age, embarkation
2. Feature engineering: family_size, is_alone, fare_per_person, age_class
3. Trains 5 classifiers with stratified CV
4. Evaluates with accuracy, ROC-AUC, confusion matrix
5. Plots ROC curves and feature importances

## Results
| Model | Accuracy | ROC-AUC |
|-------|:---:|:---:|
| **Gradient Boosting** | **~0.83** | **~0.88** |
| Random Forest | ~0.82 | ~0.87 |

## Concepts Covered
Missing value imputation · Label encoding · Feature engineering · ROC-AUC · Stratified splits
