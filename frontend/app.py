import streamlit as st
import requests
import base64
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AgroScan AI Real-time", layout="wide", page_icon="🍎")

# URL API Backend FastAPI Produksi di Hugging Face
API_URL = "https://thhmmzz-agroscan-backend-api.hf.space/predict_live"

# --- KUSTOMISASI TAMPILAN (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .result-card {
        padding: 20px; background-color: white;
        border-radius: 15px; box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state untuk menampung hasil prediksi antar frame
if "api_result" not in st.session_state:
    st.session_state.api_result = {"detected": False, "label_name": "-", "confidence": 0.0, "naratif": "", "audio_b64": ""}

# --- KELAS PEMROSES FRAME VIDEO REAL-TIME ---
class VideoAPIProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame):
        # Ubah format frame video dari browser ke ndarray OpenCV
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Kirim ke API setiap 10 frame sekali agar tidak membebani server/lagging
        if self.frame_count % 10 == 0:
            _, img_encoded = cv2.imencode('.jpg', img)
            img_bytes = img_encoded.tobytes()
            
            try:
                response = requests.post(API_URL, files={"file": ("frame.jpg", img_bytes, "image/jpeg")}, timeout=2)
                if response.status_code == 200:
                    result = response.json()
                    # Simpan hasil ke session state agar bisa dirender oleh Streamlit UI
                    st.session_state.api_result = result
            except Exception:
                pass
                
        # Kembalikan frame asli untuk ditampilkan di layar browser
        return frame

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
    st.subheader("Pusat Deteksi Kamera HP Real-time")
    
    col_cam, col_info = st.columns([2, 1])
    
    with col_cam:
        # Menggunakan WebRTC agar streaming video otomatis berjalan lancar di HTTPS Android/iOS
        ctx = webrtc_streamer(
            key="agroscan-stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoAPIProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": False},
        )
        
    with col_info:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.write("### 📋 Hasil Analisis NLP & ML")
        
        # Baca data dari hasil pemrosesan frame otomatis
        res = st.session_state.api_result
        
        if res.get("detected"):
            st.metric(label="Status Objek", value=res["label_name"].upper())
            st.metric(label="Akurasi", value=f"{res['confidence']:.2%}")
            st.info(res["naratif"])
            
            # Putar audio otomatis tanpa klik tombol jika ada data base64 baru
            if res.get("audio_b64"):
                audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{res["audio_b64"]}" type="audio/mp3"></audio>'
                st.markdown(audio_html, unsafe_allow_html=True)
        else:
            st.warning("Menunggu buah dihadapkan ke kamera...")
            
        st.markdown('</div>', unsafe_allow_html=True)

elif mode == "📊 Laporan Riwayat":
    st.subheader("Data Statistik Panen")
    st.bar_chart({"Matang": 45, "Mentah": 20})