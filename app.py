import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the model
def load_model():
    with open("Titanic_model.pkl", "rb") as model_file:
        return pickle.load(model_file)

model = load_model()

# Titanic Theme Colors
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🌊", layout="centered")
st.markdown(
    """
    
    """,
    unsafe_allow_html=True,
)

# Titanic Header
st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_column_width=True)
st.markdown("# 🌊 Titanic Survival Prediction")
st.write("### Enter passenger details to check survival probability")

# Form for Input
with st.form(key="titanic_form"):
    pclass = st.selectbox("Passenger Class", ["1st Class", "2nd Class", "3rd Class"], index=2)
    sex = st.selectbox("Sex", ["Male", "Female"], index=0)
    age = st.slider("Age", min_value=0, max_value=100, value=30, step=1)
    sibsp = st.selectbox("Siblings/Spouses Aboard", list(range(11)), index=0)
    parch = st.selectbox("Parents/Children Aboard", list(range(11)), index=0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], index=2)  # C = Cherbourg, Q = Queenstown, S = Southampton
    submit = st.form_submit_button("Predict Survival")

# Prediction Logic
if submit:
    pclass_encoded = {"1st Class": 1, "2nd Class": 2, "3rd Class": 3}[pclass]
    sex_encoded = 1 if sex == "Female" else 0  # Encoding: Female = 1, Male = 0
    embarked_encoded = {"C": 0, "Q": 1, "S": 2}[embarked]  # Encoding ports
    features = np.array([[pclass_encoded, sex_encoded, age, sibsp, parch, embarked_encoded]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100  # Probability of survival

    if prediction == 1:
        st.success(f"Survival Probability: {probability:.2f}% - You are likely to survive! 🌟")
    else:
        st.error(f"Survival Probability: {probability:.2f}% - Survival is uncertain. 💦")
