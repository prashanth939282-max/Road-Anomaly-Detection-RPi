import cv2
import os
import socket
import threading
import time
import csv
import numpy as np
from datetime import datetime

# ==========================================
# 1. HARDWARE & TESTING CONFIGURATION
# ==========================================
# SIMULATION_MODE: Set to True to test on PC without the .tflite model file
SIMULATION_MODE = True 

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

MODEL_PATH = "../models/road_model_quantized.tflite"
LOG_PATH = "../logs/road_anomalies.csv"            
DATA_DIR = "../data/snapshots"                     
UDP_PORT = 5000                                    
CONFIDENCE_MIN = 0.5                               

# ==========================================
# 2. GLOBAL STATE & THREAD SAFETY
# ==========================================
state_lock = threading.Lock()
current_gps = {"lat": "0.0000", "lon": "0.0000", "active": False}

def gps_listener():
    """
    Background Thread: Listens for 'lat,lon' UDP packets.
    Works with gps_sim.py or a phone app.
    """
    global current_gps
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        # Binding to 0.0.0.0 allows receiving from localhost AND external devices (phone)
        sock.bind(("0.0.0.0", UDP_PORT))
        print(f"[GPS] Socket bound to port {UDP_PORT}")
        
        while True:
            data, _ = sock.recvfrom(1024)
            raw_msg = data.decode('utf-8').strip()
            
            if "," in raw_msg:
                parts = raw_msg.split(",")
                if len(parts) >= 2:
                    with state_lock:
                        current_gps["lat"] = parts[0].strip()
                        current_gps["lon"] = parts[1].strip()
                        current_gps["active"] = True
    except Exception as e:
        print(f"[GPS ERROR] {e}")

# ==========================================
# 3. STORAGE LOGIC
# ==========================================
def initialize_system():
    """Creates folders and the CSV log file if missing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs("../logs", exist_ok=True)
    
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Anomaly_Type", "Conf", "Latitude", "Longitude", "Image_File"])
        print("[SYSTEM] Folders and CSV ready.")

def save_anomaly_data(frame, label, score):
    """Saves image and appends a row to the CSV log."""
    now = datetime.now()
    img_name = f"road_{now.strftime('%H%M%S_%f')}.jpg"
    img_save_path = os.path.join(DATA_DIR, img_name)
    
    # Save the current camera frame as a JPG
    cv2.imwrite(img_save_path, frame)
    
    with state_lock:
        lat, lon = current_gps["lat"], current_gps["lon"]

    # Open CSV in 'append' mode to keep previous logs
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([now.isoformat(), label, f"{score:.2f}", lat, lon, img_name])
    
    print(f"\n[DETECTED] {label} at {lat}, {lon} (Saved: {img_name})")

# ==========================================
# 4. ENGINE
# ==========================================
def main_loop():
    initialize_system()
    
    interpreter = None
    input_size = 320 # Default fallback
    
    # Only try to load the model if NOT in simulation mode
    if not SIMULATION_MODE:
        try:
            interpreter = Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            input_size = input_details[0]['shape'][1]
            print(f"[AI] Model loaded. Input size: {input_size}")
        except Exception as e:
            print(f"[AI ERROR] Model failed to load: {e}")
            return
    else:
        print("[TEST MODE] AI Model is disabled. Simulating detections...")

    cap = cv2.VideoCapture(0) # Use 0 for laptop webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_sim_time = time.time()

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break

        # --- DETECTION PHASE ---
        max_conf = 0.0
        
        if SIMULATION_MODE:
            # Randomly "detect" a pothole every 7 seconds for testing logs
            if time.time() - last_sim_time > 7.0:
                max_conf = 0.95
                last_sim_time = time.time()
        else:
            # REAL AI INFERENCE CODE
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (input_size, input_size))
            input_data = np.expand_dims(img_resized, axis=0)
            
            # Check model type
            if interpreter.get_input_details()[0]['dtype'] == np.uint8:
                input_data = input_data.astype(np.uint8)
            else:
                input_data = (input_data / 255.0).astype(np.float32)

            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            max_conf = np.max(output)

        # --- LOGGING PHASE ---
        if max_conf > CONFIDENCE_MIN:
            save_anomaly_data(frame, "Pothole", max_conf)

        # --- UI PHASE ---
        fps = 1.0 / (time.time() - start_time)
        with state_lock:
            gps_disp = f"GPS: {current_gps['lat']}, {current_gps['lon']}"
            gps_active = current_gps["active"]

        # Visual overlays
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, gps_disp, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if gps_active else (0, 0, 255), 2)
        if SIMULATION_MODE:
            cv2.putText(frame, "SIMULATION ACTIVE", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        cv2.imshow("Road Edge AI Debugger", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    t = threading.Thread(target=gps_listener, daemon=True)
    t.start()
    main_loop()