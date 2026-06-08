import streamlit as st
import cv2
import requests
import numpy as np
import time
import base64

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AgroScan AI Real-time", layout="wide", page_icon="🍎")

# URL API Backend FastAPI
API_URL = "http://127.0.0.1:8000/predict_live"

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
st.title("🌱 AgroScan AI: Real-time Smart Fruit Grading")
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1865/1865222.png", width=100)
    st.title("Control Panel")
    mode = st.radio("Navigasi", ["🏠 Beranda", "📸 Scan Buah Real-time", "📊 Laporan Riwayat"])
    st.divider()
    st.info("Sistem ini mendeteksi kematangan menggunakan Video Stream API Berbasis Cloud.")

# --- LOGIKA NAVIGASI ---

if mode == "🏠 Beranda":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Selamat Datang di Smart Farming System")
        st.write("""
            Aplikasi ini mendeteksi kualitas buah secara otomatis dan real-time lewat video kamera tanpa perlu menekan tombol jepret foto.
        """)
    with col2:
        st.image("https://img.freepik.com/free-vector/smart-farming-concept-illustration_114360-7527.jpg")

elif mode == "📸 Scan Buah Real-time":
    st.subheader("Pusat Deteksi Video Live")
    
    col_cam, col_info = st.columns([2, 1])
    
    with col_cam:
        run_camera = st.checkbox("Nyalakan Kamera Scanner", value=False)
        FRAME_WINDOW = st.image([]) 
        
    with col_info:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.write("### 📋 Hasil Analisis NLP & ML")
        info_placeholder = st.empty()
        audio_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    if run_camera:
        # Menggunakan cv2.CAP_DSHOW agar kamera Windows stabil tidak error status -1072873821
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        last_voice_time = 0 
        
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal mengakses kamera.")
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode frame gambar menjadi format JPEG bytes untuk dikirim via API
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()
            
            # Tampilkan gambar mentah ke layar dlu
            FRAME_WINDOW.image(frame_rgb, channels="RGB")
            
            try:
                # Tembak frame kamera ke Backend API
                response = requests.post(API_URL, files={"file": ("frame.jpg", img_bytes, "image/jpeg")})
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("detected"):
                        label_name = result["label_name"]
                        prob = result["confidence"]
                        naratif = result["naratif"]
                        
                        with info_placeholder.container():
                            st.metric(label="Status Objek", value=label_name.upper())
                            st.metric(label="Akurasi", value=f"{prob:.2%}")
                            st.info(naratif)
                        
                        # Atur jeda pemutaran suara otomatis (7 detik)
                        current_time = time.time()
                        if current_time - last_voice_time > 7:
                            audio_b64 = result["audio_b64"]
                            audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
                            audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                            last_voice_time = current_time
                    else:
                        with info_placeholder.container():
                            st.warning("Menunggu buah dihadapkan ke kamera...")
            except requests.exceptions.ConnectionError:
                with info_placeholder.container():
                    st.error("Koneksi ke Backend API Terputus! Nyalakan Uvicorn Anda.")
            
            time.sleep(0.03)
            
        cap.release()
    else:
        FRAME_WINDOW.image("https://dummyimage.com/600x400/000/fff&text=Kamera+Mati")

elif mode == "📊 Laporan Riwayat":
    st.subheader("Data Statistik Panen")
    st.bar_chart({"Matang": 45, "Mentah": 20})