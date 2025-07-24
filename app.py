# ===============================================
# STUDENT PERFORMANCE PREDICTION SYSTEM (ALL-IN-ONE)
# ===============================================

# -------------------- Imports --------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import os

# -------------------- File Paths --------------------
DATA_PATH = "student_mat.csv"
LABEL_ENCODER_FILE = "label_encoders.pkl"
SCALER_FILE = "scaler.pkl"
REG_MODEL_FILE = "regression_model.pkl"
CLF_MODEL_FILE = "classification_model.pkl"

# -------------------- 1. Data Loading & Preprocessing --------------------
@st.cache_resource
def load_and_train_models():
    df = pd.read_csv(DATA_PATH, sep=';')
    df['pass_fail'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

    features = ['sex', 'G1', 'G2', 'absences']
    X = df[features]
    y_reg = df['G3']
    y_clf = df['pass_fail']

    le_sex = LabelEncoder()
    X['sex'] = le_sex.fit_transform(X['sex'])

    label_encoders = {'sex': le_sex}
    joblib.dump(label_encoders, LABEL_ENCODER_FILE)

    numeric_cols = ['G1', 'G2', 'absences']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    joblib.dump(scaler, SCALER_FILE)

    # Split
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    # Models
    reg_model = LinearRegression()
    reg_model.fit(X_train_reg, y_train_reg)
    joblib.dump(reg_model, REG_MODEL_FILE)

    clf_model = LogisticRegression()
    clf_model.fit(X_train_clf, y_train_clf)
    joblib.dump(clf_model, CLF_MODEL_FILE)

    return reg_model, clf_model, scaler, label_encoders

# Load or train models
if not all(os.path.exists(f) for f in [REG_MODEL_FILE, CLF_MODEL_FILE, SCALER_FILE, LABEL_ENCODER_FILE]):
    reg_model, clf_model, scaler, label_encoders = load_and_train_models()
else:
    reg_model = joblib.load(REG_MODEL_FILE)
    clf_model = joblib.load(CLF_MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    label_encoders = joblib.load(LABEL_ENCODER_FILE)

# -------------------- 2. Streamlit UI --------------------
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.markdown("Predict **final grade (G3)** and **pass/fail** from key student info.")

with st.form("input_form"):
    name = st.text_input("👤 Student Name", placeholder="e.g., Sunny Kumar")
    sex = st.selectbox("👩‍🎓 Gender", ["M", "F"])

    col1, col2 = st.columns(2)
    with col1:
        G1 = st.slider("📘 First Period Grade (G1)", 0, 20, 10)
    with col2:
        G2 = st.slider("📘 Second Period Grade (G2)", 0, 20, 10)

    absences = st.slider("📅 Number of Absences", 0, 50, 4)
    submitted = st.form_submit_button("Predict")

if submitted:
    if not name.strip():
        st.warning("⚠️ Please enter the student name to get predictions.")
    else:
        input_dict = {
            'sex': sex,
            'G1': G1,
            'G2': G2,
            'absences': absences
        }

        features = ['sex', 'G1', 'G2', 'absences']
        input_df = pd.DataFrame([input_dict], columns=features)

        # Encode and scale
        input_df['sex'] = label_encoders['sex'].transform(input_df['sex'])
        input_df[['G1', 'G2', 'absences']] = scaler.transform(input_df[['G1', 'G2', 'absences']])

        # Predictions
        predicted_grade = reg_model.predict(input_df)[0]
        pass_fail = clf_model.predict(input_df)[0]

        st.success(f"### Prediction for {name}:")
        st.markdown(f"**Predicted Final Grade (G3):** {predicted_grade:.2f}")
        st.markdown(f"**Result:** {'✅ Pass 🎉' if pass_fail == 1 else '❌ Fail'}")

        # Grade comparison graph
        st.markdown("#### Grade Comparison")
        grades = {'G1': G1, 'G2': G2, 'Predicted G3': predicted_grade}
        fig, ax = plt.subplots()
        ax.bar(grades.keys(), grades.values(), color=['skyblue', 'lightgreen', 'salmon'])
        ax.set_ylim(0, 20)
        ax.set_ylabel("Grade")
        ax.set_title("Grades Comparison (Scale: 0-20)")
        st.pyplot(fig)

        # Feature importance
        if hasattr(reg_model, 'coef_'):
            st.markdown("#### Feature Contribution (%) to Final Grade Prediction")
            coefs = reg_model.coef_

            abs_coefs = np.abs(coefs)
            total = np.sum(abs_coefs)
            contrib_perc = 100 * abs_coefs / total

            feat_contrib = dict(zip(features, contrib_perc))
            sorted_feats = dict(sorted(feat_contrib.items(), key=lambda item: item[1], reverse=True))

            fig2, ax2 = plt.subplots()
            ax2.bar(sorted_feats.keys(), sorted_feats.values(), color='mediumseagreen')
            ax2.set_ylabel("Contribution %")
            ax2.set_ylim(0, 100)
            ax2.set_title("Feature Contribution to Prediction")
            st.pyplot(fig2)
        else:
            st.info("Feature importance not available for this model type.")
