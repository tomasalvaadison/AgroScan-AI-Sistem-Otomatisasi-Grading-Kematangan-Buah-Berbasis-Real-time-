import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AgroScan AI", layout="wide", page_icon="🍎")

# 1. LOAD MODEL HASIL TRAINING (best.pt)
@st.cache_resource
def load_trained_model():
    # Pastikan file best.pt ada di folder yang sama dengan app.py
    return YOLO('best.pt') 

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Gagal memuat model 'best.pt'. Pastikan file sudah dipindahkan ke folder proyek. Error: {e}")

# --- KUSTOMISASI TAMPILAN (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button {
        width: 100%; border-radius: 10px; height: 3em;
        background-color: #2e7d32; color: white;
    }
    .result-card {
        padding: 20px; background-color: white;
        border-radius: 15px; box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER & SIDEBAR ---
st.title("🌱 AgroScan AI: Smart Fruit Grading")
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1865/1865222.png", width=100)
    st.title("Control Panel")
    mode = st.radio("Navigasi", ["🏠 Beranda", "📸 Scan Buah", "📊 Laporan Riwayat"])
    st.divider()
    st.info("Sistem ini mendeteksi kematangan menggunakan AI YOLOv8 hasil training custom.")

# --- LOGIKA NAVIGASI ---

if mode == "🏠 Beranda":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Selamat Datang di Smart Farming System")
        st.write("Aplikasi ini menggunakan AI untuk mendeteksi kualitas apel secara real-time.")
        if st.button("Mulai Scanning"):
            st.info("Pilih menu 'Scan Buah' di sidebar.")
    with col2:
        st.image("https://img.freepik.com/free-vector/smart-farming-concept-illustration_114360-7527.jpg")

elif mode == "📸 Scan Buah":
    st.subheader("Pusat Deteksi Real-time")
    tab1, tab2 = st.tabs(["Kamera Aktif", "Analisis Detail"])
    
    with tab1:
        col_cam, col_info = st.columns([2, 1])
        with col_cam:
            img_file = st.camera_input("Scanner")
            
        with col_info:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.write("### 📋 Hasil Analisis")
            
            if img_file:
                with st.spinner("AI sedang menganalisis..."):
                    # Proses Gambar
                    image = Image.open(img_file)
                    frame = np.array(image)
                    
                    # 2. PROSES DETEKSI DENGAN AI ASLI
                    results = model.predict(frame, conf=0.5)
                    
                    # Ambil informasi hasil deteksi
                    detected_objects = results[0].boxes
                    
                    if len(detected_objects) > 0:
                        # Ambil label pertama yang terdeteksi (misal: 'matang')
                        label_id = int(detected_objects[0].cls[0])
                        label_name = model.names[label_id]
                        prob = detected_objects[0].conf[0]
                        
                        # TAMPILKAN HASIL NYATA
                        st.metric(label="Status Kematangan", value=label_name.upper(), delta=f"{prob:.2%} Akurasi")
                        
                        # Logika Grade Sederhana
                        grade = "Grade A" if "matang" in label_name.lower() else "Grade C"
                        st.metric(label="Prediksi Grade", value=grade)
                        
                        # Tampilkan gambar hasil deteksi kotak-kotak
                        annotated_image = results[0].plot()
                        st.image(annotated_image, caption="Visualisasi AI")
                    else:
                        st.warning("Buah tidak terdeteksi. Coba arahkan lebih dekat.")
            else:
                st.warning("Menunggu input kamera...")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        if img_file:
            st.write("### Perbandingan Visual")
            gray = cv2.cvtColor(np.array(Image.open(img_file)), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            st.image(edges, caption="Analisis Garis Tepi (Struktur Kulit)", width=400)
        else:
            st.info("Lakukan scan terlebih dahulu.")

elif mode == "📊 Laporan Riwayat":
    st.subheader("Data Statistik Panen")
    st.bar_chart({"Matang": 45, "Mentah": 20})