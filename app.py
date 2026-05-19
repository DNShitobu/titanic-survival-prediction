"""
Titanic Survival Predictor - Streamlit App
"""
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

@st.cache_resource(show_spinner="Training on Titanic dataset...")
def load_model():
    df = sns.load_dataset("titanic")
    df = df[["survived","pclass","sex","age","sibsp","parch","fare","embarked"]].dropna()
    df["sex"] = LabelEncoder().fit_transform(df["sex"])
    df["embarked"] = LabelEncoder().fit_transform(df["embarked"].astype(str))
    feature_cols = ["pclass","sex","age","sibsp","parch","fare","embarked"]
    X = df[feature_cols]; y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train); X_te = scaler.transform(X_test)
    models = {
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=500, random_state=42),
    }
    trained = {}; metrics = {}
    for name, m in models.items():
        m.fit(X_tr, y_train)
        trained[name] = m
        metrics[name] = {"acc": accuracy_score(y_test, m.predict(X_te)), "auc": roc_auc_score(y_test, m.predict_proba(X_te)[:,1])}
    return scaler, trained, metrics, feature_cols

scaler, trained_models, metrics, feature_cols = load_model()

st.title("🚢 Titanic Survival Predictor")
st.caption("Gradient Boosting · Random Forest · Logistic Regression")

cols = st.columns(3)
for col, (name, m) in zip(cols, metrics.items()):
    col.metric(name, f"{m['acc']:.1%} acc", f"AUC {m['auc']:.3f}")

st.divider()
st.subheader("Predict Survival")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger Class", [1,2,3], format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} class")
    sex = st.radio("Sex", ["Male","Female"], horizontal=True)
    age = st.slider("Age", 1, 80, 28)
    embarked = st.selectbox("Port of Embarkation", ["Southampton","Cherbourg","Queenstown"])
with col2:
    sibsp = st.number_input("Siblings/Spouses aboard", 0, 8, 0)
    parch = st.number_input("Parents/Children aboard", 0, 6, 0)
    fare = st.number_input("Ticket fare ($)", 0.0, 520.0, 32.0, step=1.0)

model_choice = st.selectbox("Classifier", list(trained_models.keys()))
sex_enc = 1 if sex == "Male" else 0
embarked_enc = {"Southampton":2,"Cherbourg":0,"Queenstown":1}[embarked]
passenger = pd.DataFrame([[pclass, sex_enc, age, sibsp, parch, fare, embarked_enc]], columns=feature_cols)

if st.button("🔮 Predict Survival", type="primary", use_container_width=True):
    model = trained_models[model_choice]
    X_in = scaler.transform(passenger)
    pred = model.predict(X_in)[0]
    proba = model.predict_proba(X_in)[0]
    if pred == 1:
        st.success(f"✅ SURVIVED — {proba[1]:.1%} confidence")
        st.balloons()
    else:
        st.error(f"💀 DID NOT SURVIVE — {proba[0]:.1%} confidence")
    st.progress(float(proba[1]), text=f"Survival probability: {proba[1]:.1%}")

st.divider()
st.markdown("Built by [Dnshitobu](https://github.com/Dnshitobu) · [Source](https://github.com/Dnshitobu/titanic-survival-prediction)", unsafe_allow_html=True)
