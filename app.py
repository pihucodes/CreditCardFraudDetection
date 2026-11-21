import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

MODEL_FILE = "credit_fraud_random_forest.pkl"
FEATURE_FILE = "model_features.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE), joblib.load(FEATURE_FILE)


st.title("Credit Card Fraud Detection Dashboard")
st.caption("Random Forest model trained on 284,807 transactions. Upload transactions, set a threshold, and see which ones get flagged.")

if not (os.path.exists(MODEL_FILE) and os.path.exists(FEATURE_FILE)):
    st.error(f"Put {MODEL_FILE} and {FEATURE_FILE} in the same folder as app.py, then reload.")
    st.stop()

model, features = load_model()

# ---------------- Sidebar ----------------
st.sidebar.header("Controls")
threshold = st.sidebar.slider(
    "Fraud threshold", 0.05, 0.95, 0.50, 0.05,
    help="Flag a transaction as fraud when its probability is at least this value. "
         "Lower = catch more fraud but more false alarms. Higher = fewer false alarms but more missed fraud.")
uploaded = st.sidebar.file_uploader("Upload transactions (CSV)", type=["csv"])
use_demo = st.sidebar.button("Try a demo sample from creditcard.csv",
                             disabled=not os.path.exists("creditcard.csv"))
st.sidebar.caption("Needed columns: " + ", ".join(features[:3]) + " ... " + features[-1])

data = None
if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.session_state["demo"] = False
elif use_demo:
    full = pd.read_csv("creditcard.csv")
    data = pd.concat([full[full.Class == 1].sample(60, random_state=1),
                      full[full.Class == 0].sample(1940, random_state=1)]).sample(frac=1, random_state=1)
    st.session_state["demo_df"] = data
    st.session_state["demo"] = True
elif st.session_state.get("demo"):
    data = st.session_state["demo_df"]

if data is None:
    st.info("Upload a CSV in the sidebar to begin. Use test_valid.csv, or click the demo button.")
    st.stop()

# ---------------- Validation ----------------
missing = [c for c in features if c not in data.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

X = data[features].apply(pd.to_numeric, errors="coerce")
bad = X.isnull().any(axis=1)
if bad.any():
    st.warning(f"{int(bad.sum())} row(s) had missing or non-numeric values and were skipped (rows: {list(X.index[bad][:10])}).")
X = X[~bad]
data = data.loc[X.index].copy()
if X.empty:
    st.error("No valid rows left to score.")
    st.stop()
if st.session_state.get("demo"):
    st.caption("Demo sample: includes rows the model trained on and extra fraud cases, so scores look better than real-world performance.")

# ---------------- Scoring ----------------
data["Fraud Probability"] = model.predict_proba(X)[:, 1]
data["Prediction"] = np.where(data["Fraud Probability"] >= threshold, "Fraudulent", "Legitimate")
flagged = data[data["Prediction"] == "Fraudulent"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions scored", f"{len(data):,}")
c2.metric("Flagged as fraud", f"{len(flagged):,}")
c3.metric("Flag rate", f"{len(flagged) / len(data):.2%}")
if "Amount" in data.columns:
    c4.metric("Amount flagged", f"{flagged['Amount'].sum():,.2f}")

tab1, tab2, tab3 = st.tabs(["Flagged transactions", "Charts", "Accuracy check"])

with tab1:
    st.subheader("Highest-risk transactions first")
    view = flagged.sort_values("Fraud Probability", ascending=False)
    show = ["Fraud Probability", "Prediction"] + [c for c in ["Time", "Amount", "Class"] if c in view.columns]
    st.dataframe(view[show].style.format({"Fraud Probability": "{:.3f}"}), use_container_width=True)
    st.download_button("Download all results", data.to_csv(index=False), "fraud_predictions.csv")

with tab2:
    a, b = st.columns(2)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(data["Fraud Probability"], bins=20, color="#0f766e")
    ax.axvline(threshold, color="#c2362b", linestyle="--", label=f"Threshold {threshold:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("Fraud probability")
    ax.set_ylabel("Transactions (log scale)")
    ax.legend()
    a.pyplot(fig)
    imp = pd.Series(model.feature_importances_, index=features).sort_values().tail(10)
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    imp.plot(kind="barh", ax=ax2, color="#0f766e")
    ax2.set_xlabel("Feature importance")
    fig2.tight_layout()
    b.pyplot(fig2)

with tab3:
    if "Class" in data.columns:
        y = pd.to_numeric(data["Class"], errors="coerce")
        pred = (data["Fraud Probability"] >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{prec:.2%}"); m2.metric("Recall", f"{rec:.2%}"); m3.metric("F1", f"{f1:.3f}")
        st.table(pd.DataFrame([[tn, fp], [fn, tp]],
                              index=["Actually legitimate", "Actually fraud"],
                              columns=["Predicted legitimate", "Predicted fraud"]))
        st.caption("Change the sidebar threshold and watch these update.")
    else:
        st.info("Your file has no 'Class' column, so accuracy can't be checked. This is normal for new, unlabeled transactions.")
