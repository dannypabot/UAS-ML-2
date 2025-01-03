import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Judul Aplikasi
st.title('Aplikasi Prediksi Jenis Buah Berbasis Streamlit')

# Load model dan encoder
with open('frfruit.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('encode.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Fungsi untuk prediksi
def predict_fruit(diameter, weight, red, green, blue):
    features = np.array([[diameter, weight, red, green, blue]])
    prediction = model.predict(features)
    return label_encoder.inverse_transform(prediction)[0]

# Judul aplikasi
st.title("Aplikasi Prediksi Buah")

# Input dari pengguna
diameter = st.number_input("Masukkan Diameter (cm):", min_value=0.0)
weight = st.number_input("Masukkan Berat (gram):", min_value=0.0)
red = st.number_input("Masukkan Nilai Merah (0-255):", min_value=0, max_value=255)
green = st.number_input("Masukkan Nilai Hijau (0-255):", min_value=0, max_value=255)
blue = st.number_input("Masukkan Nilai Biru (0-255):", min_value=0, max_value=255)

# Tombol untuk prediksi
if st.button("Prediksi"):
    result = predict_fruit(diameter, weight, red, green, blue)
    st.success(f"Buah yang diprediksi: {result}")

    # Menampilkan gambar sesuai dengan hasil prediksi
    if result == "orange":
        st.image("https://cimaung-cikeusal.desa.id/wp-content/uploads/2021/09/istockphoto-1140677637-612x612-1.jpg", caption="Ini adalah Jeruk", use_container_width=True)
    elif result == "grapefruit":
        st.image("https://res.cloudinary.com/dk0z4ums3/image/upload/v1693464270/attached_image/8-manfaat-anggur-merah-untuk-kesehatan.jpg", caption="Ini adalah Grapefruit", use_container_width=True)