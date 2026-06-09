import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
import os
import base64
from gtts import gTTS

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AgroScan AI Real-time", layout="wide", page_icon="🍎")

# 1. LOAD MODEL HASIL TRAINING (best.pt)
@st.cache_resource
def load_trained_model():
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
st.title("🌱 AgroScan AI: Real-time Smart Fruit Grading")
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1865/1865222.png", width=100)
    st.title("Control Panel")
    mode = st.radio("Navigasi", ["🏠 Beranda", "📸 Scan Buah Real-time", "📊 Laporan Riwayat"])
    st.divider()
    st.info("Sistem ini mendeteksi kematangan menggunakan Video Stream Real-time.")

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
        # Tombol untuk menyalakan/mematikan kamera
        run_camera = st.checkbox("Nyalakan Kamera Scanner", value=False)
        FRAME_WINDOW = st.image([]) # Tempat menaruh feed video Streamlit
        
    with col_info:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.write("### 📋 Hasil Analisis NLP & ML")
        info_placeholder = st.empty()
        audio_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    # Inisialisasi kamera menggunakan OpenCV
    if run_camera:
        cap = cv2.VideoCapture(0) # 0 adalah ID kamera internal laptop
        
        last_voice_time = 0 # Mengatur jeda suara agar tidak berisik berulang-ulang
        
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal mengakses kamera.")
                break
                
            # OpenCV membaca BGR, Streamlit butuh RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Jalankan prediksi YOLOv8 pada frame video yang sedang berjalan
            results = model.predict(frame_rgb, conf=0.6, verbose=False) 
            detected_objects = results[0].boxes
            
            # Tampilkan kotak deteksi pada gambar video
            annotated_frame = results[0].plot()
            FRAME_WINDOW.image(annotated_frame, channels="RGB")
            
            # Logika jika buah dihadapkan ke kamera dan terdeteksi
            if len(detected_objects) > 0:
                label_id = int(detected_objects[0].cls[0])
                label_name = model.names[label_id]
                prob = detected_objects[0].conf[0]
                
                # Buat kalimat naratif (NLG - NLP)
                if "matang" in label_name.lower():
                    naratif = f"Terdeteksi buah {label_name.upper()} dengan akurasi {prob:.2%}. Buah siap dipanen."
                else:
                    naratif = f"Terdeteksi buah {label_name.upper()} dengan akurasi {prob:.2%}. Tunda pemanenan."
                
                # Tampilkan teks hasil analisis secara real-time di kolom kanan
                with info_placeholder.container():
                    st.metric(label="Status Objek", value=label_name.upper())
                    st.metric(label="Akurasi", value=f"{prob:.2%}")
                    st.info(naratif)
                
                # FOTO & SUARA OTOMATIS: 
                # Jika terdeteksi kuat dan sudah lewat 5 detik dari suara terakhir (supaya tidak spam suara)
                current_time = time.time()
                if current_time - last_voice_time > 7: 
                    try:
                        tts = gTTS(text=naratif, lang='id')
                        tts.save("live_hasil.mp3")
                        
                        with open("live_hasil.mp3", "rb") as f:
                            data = f.read()
                            b64 = base64.b64encode(data).decode()
                            audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
                            audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                        
                        last_voice_time = current_time
                    except Exception as e:
                        pass
            else:
                with info_placeholder.container():
                    st.warning("Menunggu buah dihadapkan ke kamera...")
                    
            time.sleep(0.03) # Menjaga fps agar laptop tidak lag
            
        cap.release() # Matikan kamera saat checkbox dimatikan
    else:
        FRAME_WINDOW.image("https://dummyimage.com/600x400/000/fff&text=Kamera+Mati")

elif mode == "📊 Laporan Riwayat":
    st.subheader("Data Statistik Panen")
    st.bar_chart({"Matang": 45, "Mentah": 20})