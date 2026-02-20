import cv2
import os
import socket
import threading
import time
import csv
import numpy as np
from datetime import datetime
from collections import deque

# ==========================================
# 1. SYSTEM & DIRECTORY CONFIGURATION
# ==========================================
# Toggle between live hardware deployment and PC-based testing
SIMULATION_MODE = True  
MODEL_PATH = "../models/road_model_quantized.tflite"
LOG_PATH = "../logs/road_anomalies.csv"            
DATA_DIR = "../data/snapshots"   
VIDEO_DIR = "../data/video_clips" 

# Detection and Logging parameters
CONFIDENCE_MIN = 0.6       
LOG_COOLDOWN = 5.0         # Prevention of duplicate logs for the same anomaly
VIDEO_BUFFER_SEC = 3       # Pre-detection footage duration
FPS_ESTIMATE = 10          # Targeted frame rate for video assembly

# ==========================================
# 2. GLOBAL STATE & CONCURRENCY CONTROL
# ==========================================
# Lock ensures thread-safe access to shared variables
state_lock = threading.Lock()
current_gps = {"lat": "0.0000", "lon": "0.0000", "active": False}
system_stats = {"cpu_temp": "N/A"}

# Ring buffer to store the most recent frames in memory
video_buffer = deque(maxlen=VIDEO_BUFFER_SEC * FPS_ESTIMATE)

def gps_listener():
    """
    Background listener for GPS coordinates sent via UDP.
    Updates the global state when a valid 'lat,lon' string is received.
    """
    global current_gps
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", 5000))
        while True:
            data, _ = sock.recvfrom(1024)
            parts = data.decode('utf-8').strip().split(",")
            if len(parts) >= 2:
                with state_lock:
                    current_gps["lat"], current_gps["lon"] = parts[0], parts[1]
                    current_gps["active"] = True
    except: pass

def hardware_monitor():
    """
    Continuous check of system temperature.
    Reads from the Raspberry Pi thermal zone or defaults to PC mode.
    """
    global system_stats
    while True:
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp = int(f.read()) / 1000.0
                with state_lock:
                    system_stats["cpu_temp"] = f"{temp:.1f}C"
        except:
            with state_lock: system_stats["cpu_temp"] = "PC-Mode"
        time.sleep(5)

# ==========================================
# 3. DATA PERSISTENCE & VIDEO EXPORT
# ==========================================
def save_video_clip(frames_to_save, filename):
    """
    Writes a list of frames to an AVI file.
    Executed in a separate thread to avoid blocking the main vision pipeline.
    """
    if not frames_to_save: return
    
    h, w, _ = frames_to_save[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(VIDEO_DIR, filename), fourcc, 10.0, (w, h))
    
    for f in frames_to_save:
        out.write(f)
    out.release()

def initialize_system():
    """
    Verified required directories exist and initializes the CSV log file 
    with standard headers if it is not already present.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs("../logs", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Type", "Conf", "Lat", "Lon", "Img_File", "Vid_File"])

# ==========================================
# 4. AI INFERENCE WRAPPER
# ==========================================
def process_inference(interpreter, frame, input_size):
    """
    Performs image preprocessing and executes the TFLite model.
    Returns the maximum confidence score detected in the current frame.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (input_size, input_size))
    input_data = np.expand_dims(img_resized, axis=0)

    # Handle float32 vs uint8 (quantized) models
    if interpreter.get_input_details()[0]['dtype'] == np.float32:
        input_data = (input_data / 255.0).astype(np.float32)
    
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    return np.max(output) 

# ==========================================
# 5. MAIN PROCESSING LOOP
# ==========================================
def main_loop():
    initialize_system()
    last_log_time = 0
    
    # Interpreter initialization logic
    interpreter = None
    input_size = 320
    if not SIMULATION_MODE:
        try:
            from tflite_runtime.interpreter import Interpreter
            interpreter = Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            input_size = interpreter.get_input_details()[0]['shape'][1]
        except Exception as e:
            print(f"Inference Initialization Error: {e}")
            return

    # Video stream configuration
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Add current frame to the pre-detection buffer
        video_buffer.append(frame.copy())
        
        # Step 1: Execute Detection
        confidence = 0.0
        if SIMULATION_MODE:
            # Manual trigger for debugging purposes
            if cv2.waitKey(1) & 0xFF == ord('s'): confidence = 0.99
        else:
            confidence = process_inference(interpreter, frame, input_size)

        # Step 2: Anomaly Event Handling
        current_time = time.time()
        if confidence > CONFIDENCE_MIN and (current_time - last_log_time) > LOG_COOLDOWN:
            now = datetime.now()
            file_stamp = now.strftime('%H%M%S')
            img_name = f"img_{file_stamp}.jpg"
            vid_name = f"clip_{file_stamp}.avi"
            
            # Save static image snapshot
            cv2.imwrite(os.path.join(DATA_DIR, img_name), frame)
            
            # Initiate background video writing of the stored buffer
            frames_to_write = list(video_buffer)
            threading.Thread(target=save_video_clip, args=(frames_to_write, vid_name)).start()
            
            # Capture current GPS coordinates
            with state_lock:
                lat, lon = current_gps["lat"], current_gps["lon"]
            
            # Append entry to central log file
            with open(LOG_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([now.isoformat(), "Pothole", f"{confidence:.2f}", lat, lon, img_name, vid_name])
            
            last_log_time = current_time
            print(f"[EVENT] Detection recorded at {lat}, {lon}")

        # Step 3: Heads-Up Display (HUD)
        with state_lock:
            temp = system_stats["cpu_temp"]
            gps_str = f"GPS: {current_gps['lat']}, {current_gps['lon']}"
            gps_col = (0, 255, 0) if current_gps["active"] else (0, 0, 255)

        # Draw overlays on the frame
        cv2.putText(frame, f"TEMP: {temp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, gps_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, gps_col, 2)
        
        # Visual notification for detection event
        if (current_time - last_log_time) < 1.0:
            cv2.rectangle(frame, (0,0), (640,480), (0, 0, 255), 10) 

        cv2.imshow("Road Anomaly Edge AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Resource release
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Launch parallel tasks for GPS and Hardware monitoring
    threading.Thread(target=gps_listener, daemon=True).start()
    threading.Thread(target=hardware_monitor, daemon=True).start()
    main_loop()