import streamlit as st
import numpy as np
import joblib
import xgboost as xgb
from auth import *

create_user_table()

st.set_page_config(
    page_title="Diabetes Prediction Web App",
    page_icon="🩺"
)

# Session
if "user" not in st.session_state:
    st.session_state.user = None
    st.session_state.role = None

# Sidebar menu
menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

# ---------------- REGISTER ----------------
if choice == "Register":
    st.title("📝 User Registration")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        if register_user(email, password):
            st.success("Registered successfully! Please login.")
        else:
            st.error("User already exists")

# ---------------- LOGIN ----------------
elif choice == "Login":
    st.title("🔐 User Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        result = login_user(email, password)
        if result:
            st.session_state.user = email
            st.session_state.role = result[0]
            st.success("Login successful")
        else:
            st.error("Invalid email or password")

# ---------------- AFTER LOGIN ----------------
if st.session_state.user:
    st.sidebar.success(f"Logged in as {st.session_state.user}")

    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.experimental_rerun()

    st.title("🩺 Diabetes Prediction System")

    # Load model & preprocessors
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_model.json")

    scaler = joblib.load("models/scaler.pkl")
    imputer = joblib.load("models/imputer.pkl")

    # Inputs
    preg = st.number_input("Pregnancies", 0, 20, 1)
    glu = st.number_input("Glucose", 50, 250, 120)
    bp = st.number_input("Blood Pressure", 40, 150, 80)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin", 0, 400, 100)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)

    if st.button("Predict"):
        data = np.array([[preg, glu, bp, skin, insulin, bmi, dpf, age]])
        data = imputer.transform(data)
        data = scaler.transform(data)

        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1] * 100

        if pred == 1:
            st.error(f"⚠️ Diabetic (Risk: {prob:.2f}%)")
        else:
            st.success(f"✅ Non-Diabetic (Risk: {prob:.2f}%)")
