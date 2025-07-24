import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load models and preprocessors
reg_model = joblib.load("regression_model.pkl")
clf_model = joblib.load("classification_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")  # {'sex': LabelEncoder()}

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

        # Encode sex
        input_df['sex'] = label_encoders['sex'].transform(input_df['sex'])

        # Scale numeric features
        numeric_cols = ['G1', 'G2', 'absences']
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

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

        # Feature importance (if available)
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
