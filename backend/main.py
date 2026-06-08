from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
from gtts import gTTS
import shutil
import os
import time

app = FastAPI(
    title="AgroScan AI - Core API",
    description="API untuk Deteksi Kematangan Buah Apel Real-time",
    version="1.0.0"
)

# Load model
model = YOLO('best.pt')

@app.post("/predict_live")
async def predict_live(file: UploadFile = File(...)):
    temp_path = f"live_{time.time()}.jpg"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. Proses Machine Learning (YOLOv8)
        # Gunakan cv2.CAP_DSHOW atau menaikkan conf diatur di server ini
        results = model.predict(temp_path, conf=0.75, verbose=False)
        detected_objects = results[0].boxes
        
        if len(detected_objects) == 0:
            if os.path.exists(temp_path): os.remove(temp_path)
            return {"detected": False}
            
        label_id = int(detected_objects[0].cls[0])
        label_name = model.names[label_id]
        prob = float(detected_objects[0].conf[0])
        
        # 2. Proses NLP (NLG)
        if "matang" in label_name.lower():
            naratif = f"Terdeteksi buah {label_name.upper()} dengan akurasi {prob:.2%}. Buah siap dipanen."
        else:
            naratif = f"Terdeteksi buah {label_name.upper()} dengan akurasi {prob:.2%}. Tunda pemanenan."
            
        # 3. Buat File Audio di Sisi Server Backend
        audio_filename = "temp_voice.mp3"
        tts = gTTS(text=naratif, lang='id')
        tts.save(audio_filename)
        
        # Baca audio menjadi byte untuk dikirim di JSON
        import base64
        with open(audio_filename, "rb") as f:
            audio_bytes = base64.b64encode(f.read()).decode()
            
        if os.path.exists(temp_path): os.remove(temp_path)
        if os.path.exists(audio_filename): os.remove(audio_filename)
        
        return {
            "detected": True,
            "label_name": label_name,
            "confidence": prob,
            "naratif": naratif,
            "audio_b64": audio_bytes
        }
        
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})