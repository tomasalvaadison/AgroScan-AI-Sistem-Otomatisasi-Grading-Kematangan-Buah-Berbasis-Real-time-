import streamlit as st
import requests
import base64

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AgroScan AI Real-time", layout="wide", page_icon="🍎")

# URL API Backend FastAPI Produksi di Hugging Face
API_URL = "https://thhmmzz-agroscan-backend-api.hf.space/predict_live"

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
            Aplikasi ini mendeteksi kualitas buah secara otomatis dan real-time lewat video kamera secara responsif.
        """)
    with col2:
        st.image("https://img.freepik.com/free-vector/smart-farming-concept-illustration_114360-7527.jpg")

elif mode == "📸 Scan Buah Real-time":
    st.subheader("Pusat Deteksi Kamera HP Real-time")
    
    col_cam, col_info = st.columns([2, 1])
    
    with col_cam:
        # Menggunakan camera_input resmi agar kompatibel 100% dengan SSL HTTPS di Android/iOS/PC
        img_file_buffer = st.camera_input("Arahkan Buah Apel ke Kamera")
        
    with col_info:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.write("### 📋 Hasil Analisis NLP & ML")
        
        info_placeholder = st.empty()
        audio_placeholder = st.empty()
        
        if img_file_buffer is not None:
            # Ambil bytes gambar secara real-time dari tangkapan kamera browser
            img_bytes = img_file_buffer.getvalue()
            
            try:
                # Tembak langsung ke Hugging Face Backend Anda
                response = requests.post(API_URL, files={"file": ("frame.jpg", img_bytes, "image/jpeg")}, timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get("detected"):
                        label_name = result["label_name"]
                        prob = result["confidence"]
                        naratif = result["naratif"]
                        
                        info_placeholder.metric(label="Status Objek", value=label_name.upper())
                        st.metric(label="Akurasi", value=f"{prob:.2%}")
                        st.info(naratif)
                        
                        # Putar audio otomatis tanpa intervensi klik tambahan
                        audio_b64 = result["audio_b64"]
                        audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
                        audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                    else:
                        info_placeholder.warning("Buah belum terdeteksi dengan jelas, posisikan ulang di depan kamera.")
                else:
                    info_placeholder.error(f"Server Backend membalas dengan status: {response.status_code}")
            except Exception as e:
                info_placeholder.error("Gagal berkomunikasi dengan core AI di Hugging Face.")
        else:
            info_placeholder.warning("Kamera aktif. Menunggu objek buah dihadapkan ke lensa...")
            
        st.markdown('</div>', unsafe_allow_html=True)

elif mode == "📊 Laporan Riwayat":
    st.subheader("Data Statistik Panen")
    st.bar_chart({"Matang": 45, "Mentah": 20})