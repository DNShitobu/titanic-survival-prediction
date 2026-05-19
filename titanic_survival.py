"""
Titanic Survival Prediction
============================
Binary classification predicting passenger survival using seaborn Titanic dataset.
Covers EDA, feature engineering, multiple classifiers, ROC curves.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

df = sns.load_dataset("titanic")
print("=" * 60)
print("TITANIC SURVIVAL PREDICTION")
print("=" * 60)
print(f"Shape: {df.shape}  |  Survival rate: {df['survived'].mean():.2%}")

# EDA
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
df["survived"].value_counts().plot(kind="bar", ax=axes[0,0], color=["salmon","steelblue"], rot=0)
axes[0,0].set_title("Survival Count")
df.groupby("pclass")["survived"].mean().plot(kind="bar", ax=axes[0,1], color="mediumseagreen", rot=0)
axes[0,1].set_title("Survival Rate by Class")
df.groupby("sex")["survived"].mean().plot(kind="bar", ax=axes[0,2], color=["coral","steelblue"], rot=0)
axes[0,2].set_title("Survival Rate by Sex")
axes[1,0].hist(df[df["survived"]==0]["age"].dropna(), bins=25, alpha=0.6, label="Not Survived", color="salmon")
axes[1,0].hist(df[df["survived"]==1]["age"].dropna(), bins=25, alpha=0.6, label="Survived", color="steelblue")
axes[1,0].set_title("Age by Survival")
axes[1,0].legend()
df.groupby("embark_town")["survived"].mean().plot(kind="bar", ax=axes[1,1], color="orchid", rot=0)
axes[1,1].set_title("Survival by Embarkation")
plt.suptitle("Titanic EDA", fontsize=14)
plt.tight_layout()
plt.savefig("eda_titanic.png", dpi=150, bbox_inches="tight")
plt.close()

df = df.drop(columns=["deck","alive","who","adult_male","class","embark_town"])
df["age"].fillna(df["age"].median(), inplace=True)
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)
df["sex"] = LabelEncoder().fit_transform(df["sex"])
df["embarked"] = LabelEncoder().fit_transform(df["embarked"].astype(str))
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["is_alone"] = (df["family_size"] == 1).astype(int)
df["fare_per_person"] = df["fare"] / df["family_size"]
df["age_class"] = df["age"] * df["pclass"]

feature_cols = ["pclass","sex","age","sibsp","parch","fare","embarked","family_size","is_alone","fare_per_person","age_class"]
X, y = df[feature_cols], df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
}

plt.figure(figsize=(7, 6))
results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {"acc": acc, "auc": auc}
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    print(f"{name}: Acc={acc:.4f} AUC={auc:.4f}")

plt.plot([0,1],[0,1],"k--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves")
plt.legend(fontsize=9); plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight"); plt.close()

best = max(results, key=lambda k: results[k]["auc"])
print(f"\nBest: {best} (AUC={results[best]['auc']:.4f})")
print("\n✅ Done!")
