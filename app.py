import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
models = {
    "KMeans": joblib.load("kMeans.pkl"),
    "Birch": joblib.load("Brich.pkl"),
    "Gaussian Mixture": joblib.load("GaussianMixture.pkl")
}
st.title("🛍️ Mall Customer Data Input Form")
genre = st.radio("Select Gender:", ("Male", "Female"))
age = st.slider("Age", min_value=18, max_value=70, value=30)
annual_income = st.slider("Annual Income (k$)", min_value=15, max_value=126, value=50)
spending_score = st.slider("Spending Score (1–100)", min_value=1, max_value=100, value=50)
st.subheader("Your Input Data:")
st.write(f"**Gender:** {genre}")
st.write(f"**Age:** {age}")
st.write(f"**Annual Income (k$):** {annual_income}")
st.write(f"**Spending Score:** {spending_score}")
if st.button("🔮 Predict Customer Cluster", type="primary"):
    gender_encoder = LabelEncoder()
    gender_encoder.fit(["Male", "Female"])
    genre_encoded = gender_encoder.transform([genre])[0]
    input_data = np.array([[age, annual_income, spending_score]])
    input_scaled = scaler.transform(input_data)
    input_pca = pca.transform(input_data)
    input_with_genre = np.array([[genre_encoded, age, annual_income, spending_score]])
    
    st.subheader("📊 Model Predictions")
    cols = st.columns(2)
    for idx, (name, model) in enumerate(models.items()):
            cluster = model.predict(input_with_genre)[0]
            with cols[idx % 2]:
                st.success(f"**{name}:** Cluster {cluster}")
    