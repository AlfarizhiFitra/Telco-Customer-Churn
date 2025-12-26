import streamlit as st
import pandas as pd
import joblib
import uuid

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# =========================
# Load Model
# =========================
model = joblib.load("telco_churn_model.pkl")

# =========================
# Header
# =========================
st.title("Telco Customer Churn Prediction")
st.markdown(
    "Aplikasi ini memprediksi **kemungkinan pelanggan melakukan churn** "
    "berdasarkan data layanan dan pelanggan."
)

st.divider()

# =========================
# Form Input
# =========================
with st.form("churn_form"):
    st.subheader("Data Pelanggan")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (bulan)", 0, 72, 1)

        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    st.subheader("Kontrak & Pembayaran")

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    st.subheader("Biaya Dalam USD")

    monthly = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
    total = st.number_input("Total Charges", min_value=0.0, step=1.0)

    submit = st.form_submit_button("🔍 Prediksi Churn")

# =========================
# Prediction
# =========================
if submit:
    input_data = pd.DataFrame({
        "customerID": [str(uuid.uuid4())],  # dummy ID
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone],
        "MultipleLines": [multiple],
        "InternetService": [internet],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

st.divider()
st.subheader("Hasil Prediksi")

if probability < 0.5:
    st.error("⚠️ Pelanggan **BERPOTENSI CHURN**")
else:
    st.success("✅ Pelanggan **TIDAK CHURN**")

st.write(f"**Probabilitas Churn:** `{probability:.2%}`")

