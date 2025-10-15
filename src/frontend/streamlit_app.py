import streamlit as st
import requests

# --- Page setup ---
st.set_page_config(page_title="🚢 Titanic Survival Predictor", page_icon="🧭", layout="centered")

st.title("🚢 Titanic Survival Prediction")
st.markdown("#### Enter passenger details below to check survival probability")

# --- Input fields ---
Pclass = st.selectbox("🎟️ Passenger Class", [1, 2, 3])
Sex = st.radio("👫 Gender", ["Male", "Female"])
Sex = 0 if Sex == "Male" else 1
Age = st.slider("🎂 Age", 0.0, 100.0, 25.0)
SibSp = st.number_input("👨‍👩‍👧 Siblings/Spouses aboard", min_value=0, max_value=10, value=0)
Parch = st.number_input("👶 Parents/Children aboard", min_value=0, max_value=10, value=0)
Fare = st.slider("💰 Ticket Fare (£)", 0.0, 550.0, 30.0)
Embarked = st.selectbox("⚓ Port of Embarkation", ["S", "C", "Q"])

# Encode embarkation
emb_q = 1 if Embarked == "Q" else 0
emb_s = 1 if Embarked == "S" else 0

# --- Prediction button ---
if st.button("🔍 Predict Survival"):
    payload = {
        "Pclass": int(Pclass),
        "Sex": int(Sex),
        "Age": float(Age),
        "SibSp": int(SibSp),
        "Parch": int(Parch),
        "Fare": float(Fare),
        "Embarked_Q": emb_q,
        "Embarked_S": emb_s
    }
    backend = "http://localhost:8000/predict"

    try:
        res = requests.post(backend, json=payload, timeout=5)
        result = res.json()

        st.markdown("---")
        st.subheader("🧠 Prediction Result")

        pred = result.get("prediction", None)
        proba = result.get("probabilities", [0, 0])

        if pred == 1:
            st.success(f"✅ **Survived!** Probability: {proba[1]*100:.2f}%")
        elif pred == 0:
            st.error(f"❌ **Did Not Survive** Probability: {proba[0]*100:.2f}%")
        else:
            st.warning("⚠️ Could not determine prediction.")

        # Add progress bar visualization
        st.progress(proba[1])
        st.caption("⬆️ Bar shows the predicted chance of survival")

    except Exception as e:
        st.error(f"🚨 Error connecting to backend: {e}")
