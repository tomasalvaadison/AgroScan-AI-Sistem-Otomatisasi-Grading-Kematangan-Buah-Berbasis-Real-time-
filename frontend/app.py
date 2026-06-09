import streamlit as st
import cv2
import requests
import time
import base64

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="AgroScan AI Real-time", layout="wide", page_icon="🍎")

# URL API LOCALHOST (Untuk run di laptop) & CLOUD HUGGING FACE
# Jika Anda menjalankan Uvicorn di laptop, ganti URL ke "http://127.0.0.1:8000/predict_live"
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
    st.info("Sistem ini mendeteksi kematangan menggunakan API Multidevice.")

# --- LOGIKA NAVIGASI ---

if mode == "🏠 Beranda":
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Selamat Datang di Smart Farming System")
        st.write("""
            Aplikasi ini mendukung pemindaian otomatis real-time lewat laptop serta fitur ambil foto praktis lewat handphone.
        """)
    with col2:
        st.image("https://img.freepik.com/free-vector/smart-farming-concept-illustration_114360-7527.jpg")

elif mode == "📸 Scan Buah Real-time":
    st.subheader("Pusat Deteksi Kamera Multidevice")
    
    # PILIHAN METODE SCAN
    metode_scan = st.radio(
        "Pilih Perangkat Keras Anda:", 
        ["💻 Scan Otomatis / Live (Laptop/PC)", "📱 Ambil Foto (Handphone)"],
        horizontal=True
    )
    
    col_cam, col_info = st.columns([2, 1])
    
    with col_info:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.write("### 📋 Hasil Analisis NLP & ML")
        info_placeholder = st.empty()
        audio_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- MODE 1: LAPTOP / PC (REAL-TIME VIDEO LIVE - BEBAS LAG) ---
    if metode_scan == "💻 Scan Otomatis / Live (Laptop/PC)":
        with col_cam:
            run_camera = st.checkbox("Nyalakan Kamera Scanner (Laptop)", value=False)
            FRAME_WINDOW = st.image([]) 
            
        if run_camera:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if "last_detected_label" not in st.session_state:
                st.session_state.last_detected_label = ""
            
            # Penghitung frame untuk mencegah macet/lag internet
            frame_counter = 0
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    info_placeholder.error("Gagal mengakses webcam laptop.")
                    break
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb, channels="RGB") # Tampilan lokal dijamin lancar murni
                
                frame_counter += 1
                
                # Optimasi: Hanya kirim ke AI setiap 15 frame sekali (~0.5 detik)
                if frame_counter % 15 == 0:
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    img_bytes = img_encoded.tobytes()
                    
                    try:
                        response = requests.post(API_URL, files={"file": ("frame.jpg", img_bytes, "image/jpeg")}, timeout=2)
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
                                
                                # Trigger suara cerdas jika objek berubah
                                if st.session_state.last_detected_label != label_name:
                                    audio_b64 = result["audio_b64"]
                                    audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
                                    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                                    st.session_state.last_detected_label = label_name
                            else:
                                with info_placeholder.container():
                                    st.warning("Menunggu buah dihadapkan ke kamera...")
                                st.session_state.last_detected_label = ""
                    except Exception:
                        pass
                time.sleep(0.01)
            cap.release()
        else:
            FRAME_WINDOW.image("https://dummyimage.com/600x400/000/fff&text=Kamera+Laptop+Mati")
            st.session_state.last_detected_label = ""

    # --- MODE 2: HANDPHONE (TAKE PHOTO VIA BROWSER CLOUD) ---
    else:
        with col_cam:
            img_file_buffer = st.camera_input("Arahkan kamera HP Anda ke buah apel")
            
        if img_file_buffer is not None:
            img_bytes = img_file_buffer.getvalue()
            try:
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
                        
                        # Mode HP selalu berbunyi saat foto berhasil dijepret
                        audio_b64 = result["audio_b64"]
                        audio_html = f'<audio autoplay="true"><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'
                        audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                    else:
                        with info_placeholder.container():
                            st.warning("Buah tidak terdeteksi, silakan ambil foto ulang.")
            except Exception:
                info_placeholder.error("Gagal terhubung ke server backend AI.")
        else:
            info_placeholder.warning("Silakan klik tombol jepret di HP untuk menganalisis.")

elif mode == "📊 Laporan Riwayat":
    st.subheader("Data Statistik Panen")
    st.bar_chart({"Matang": 45, "Mentah": 20})