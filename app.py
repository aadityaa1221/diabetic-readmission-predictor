import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load model and features
model    = joblib.load('readmission_model.pkl')
features = joblib.load('model_features.pkl')

st.set_page_config(page_title="Hospital Readmission Predictor", page_icon="🏥", layout="wide")

st.title("🏥 Hospital Readmission Risk Predictor")
st.markdown("Predict whether a diabetic patient is at risk of being readmitted within 30 days.")
st.markdown("---")

# ── Sidebar inputs ──────────────────────────────────────────────────────────
st.sidebar.header("🧾 Patient Information")

age = st.sidebar.selectbox("Age Group", [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
])

time_in_hospital = st.sidebar.slider("Days in Hospital",               1,  14,  3)
n_lab_procedures = st.sidebar.slider("Number of Lab Procedures",       1, 132, 40)
n_procedures     = st.sidebar.slider("Number of Procedures",           0,   6,  1)
n_medications    = st.sidebar.slider("Number of Medications",          1,  81, 15)
n_outpatient     = st.sidebar.slider("Outpatient Visits (past year)",  0,  42,  0)
n_inpatient      = st.sidebar.slider("Inpatient Visits (past year)",   0,  21,  0)
n_emergency      = st.sidebar.slider("Emergency Visits (past year)",   0,  76,  0)

medical_specialty = st.sidebar.selectbox("Medical Specialty", [
    "InternalMedicine", "Family/GeneralPractice", "Cardiology",
    "Emergency/Trauma", "Surgery", "Other", "Missing"
])

glucose_test = st.sidebar.selectbox("Glucose Serum Test Result", ["no", "normal", "high"])
A1Ctest      = st.sidebar.selectbox("HbA1c Test Result",         ["no", "normal", "high"])
change       = st.sidebar.selectbox("Change in Diabetes Medication?", ["no", "yes"])
diabetes_med = st.sidebar.selectbox("On Diabetes Medication?",        ["yes", "no"])

# ── Exact encoding maps from training ───────────────────────────────────────
age_order = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
]

specialty_map = {
    "Cardiology": 0, "Emergency/Trauma": 1, "Family/GeneralPractice": 2,
    "InternalMedicine": 3, "Missing": 4, "Other": 5, "Surgery": 6
}
glucose_map  = {"high": 0, "no": 1, "normal": 2}
a1c_map      = {"high": 0, "no": 1, "normal": 2}
change_map   = {"no": 0, "yes": 1}
med_map      = {"no": 0, "yes": 1}

# ── Build input dataframe ───────────────────────────────────────────────────
input_data = pd.DataFrame([{
    'age':               age_order.index(age),
    'time_in_hospital':  time_in_hospital,
    'n_lab_procedures':  n_lab_procedures,
    'n_procedures':      n_procedures,
    'n_medications':     n_medications,
    'n_outpatient':      n_outpatient,
    'n_inpatient':       n_inpatient,
    'n_emergency':       n_emergency,
    'medical_specialty': specialty_map[medical_specialty],
    'glucose_test':      glucose_map[glucose_test],
    'A1Ctest':           a1c_map[A1Ctest],
    'change':            change_map[change],
    'diabetes_med':      med_map[diabetes_med],
}])

input_data = input_data[features]

# ── Prediction ───────────────────────────────────────────────────────────────
st.markdown("### 🔍 Prediction Result")

if st.button("Predict Readmission Risk", use_container_width=True):

    prob         = model.predict_proba(input_data)[0][1]
    risk_percent = round(prob * 100, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Readmission Risk Score", value=f"{risk_percent}%")

        if prob >= 0.5:
            st.error("⚠️ HIGH RISK — This patient may need follow-up care after discharge.")
        else:
            st.success("✅ LOW RISK — Patient is unlikely to be readmitted within 30 days.")

        # Risk gauge
        st.markdown("#### 📊 Risk Gauge")
        fig, ax = plt.subplots(figsize=(5, 1))
        bar_color = "#F44336" if prob >= 0.5 else "#4CAF50"
        ax.barh(["Risk"], [prob],         color=bar_color,  height=0.4)
        ax.barh(["Risk"], [1 - prob],     left=[prob],
                color="#E0E0E0", height=0.4)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"Risk: {risk_percent}%", fontsize=12)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### 📋 Patient Feature Overview")

        feature_labels = {
            'age':               f"Age Group ({age})",
            'time_in_hospital':  "Days in Hospital",
            'n_lab_procedures':  "Lab Procedures",
            'n_procedures':      "Procedures",
            'n_medications':     "Medications",
            'n_outpatient':      "Outpatient Visits",
            'n_inpatient':       "Inpatient Visits",
            'n_emergency':       "Emergency Visits",
            'medical_specialty': f"Specialty ({medical_specialty})",
            'glucose_test':      f"Glucose Test ({glucose_test})",
            'A1Ctest':           f"HbA1c Test ({A1Ctest})",
            'change':            f"Medication Change ({change})",
            'diabetes_med':      f"Diabetes Med ({diabetes_med})",
        }

        raw_values    = input_data.iloc[0].to_dict()
        display_labels = [feature_labels[f] for f in features]
        display_values = [raw_values[f]      for f in features]

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        colors = ["#2196F3" if v > 0 else "#B0BEC5" for v in display_values]
        ax2.barh(display_labels, display_values, color=colors, edgecolor='white')
        ax2.set_xlabel("Encoded Value")
        ax2.set_title("Input Feature Values", fontsize=12)
        ax2.spines[['top', 'right']].set_visible(False)

        active_patch = mpatches.Patch(color='#2196F3', label='Active / Present')
        zero_patch   = mpatches.Patch(color='#B0BEC5', label='Zero / Absent')
        ax2.legend(handles=[active_patch, zero_patch], fontsize=8)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

else:
    st.info("👈 Fill in the patient details on the left sidebar and click **Predict**.")