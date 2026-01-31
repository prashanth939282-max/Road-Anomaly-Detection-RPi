import cv2
import pandas as pd
import os
import socket
import threading
import time
from datetime import datetime

# ==========================================
# CONFIGURATION (Software Lead Settings)
# ==========================================
LOG_PATH = "../logs/road_anomalies.csv"
DATA_DIR = "../data"
UDP_PORT = 5000  # Port for Phone GPS
DETECTION_INTERVAL = 1.0  # Seconds between each detection attempt

# Global state
current_location = {"lat": "Waiting...", "lon": "Waiting..."}
last_detection_time = 0

def gps_receiver_thread():
    """
    Listens for GPS data from the phone over the Wi-Fi hotspot.
    Expected format from phone app: 'latitude,longitude'
    """
    global current_location
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", UDP_PORT))
        print(f"[GPS] Listener active on Port {UDP_PORT}")
        while True:
            data, _ = sock.recvfrom(1024)
            message = data.decode('utf-8').strip()
            if "," in message:
                parts = message.split(",")
                current_location["lat"] = parts[0][:10]
                current_location["lon"] = parts[1][:10]
    except Exception as e:
        print(f"[GPS ERROR] {e}")

def setup_environment():
    """Ensures folders and log files are ready."""
    for path in [DATA_DIR, "../logs"]:
        if not os.path.exists(path):
            os.makedirs(path)
    
    if not os.path.exists(LOG_PATH):
        headers = ["Timestamp", "Anomaly_Type", "Confidence", "Latitude", "Longitude", "Image_Path"]
        pd.DataFrame(columns=headers).to_csv(LOG_PATH, index=False)
        print("[INFO] Environment Initialized.")

def log_anomaly(frame, anomaly_type="Pothole", confidence=0.98):
    """Saves the evidence image and records GPS/Time to CSV."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    file_time = now.strftime("%H%M%S_%f")
    
    image_name = f"detect_{file_time}.jpg"
    image_path = os.path.join(DATA_DIR, image_name)
    
    # Save high-res snapshot
    cv2.imwrite(image_path, frame)
    
    # Update CSV
    new_entry = {
        "Timestamp": timestamp,
        "Anomaly_Type": anomaly_type,
        "Confidence": confidence,
        "Latitude": current_location["lat"],
        "Longitude": current_location["lon"],
        "Image_Path": os.path.abspath(image_path)
    }
    
    df = pd.read_csv(LOG_PATH)
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    
    print(f"\n[!!!] ANOMALY DETECTED [!!!]")
    print(f"Location: {current_location['lat']}, {current_location['lon']}")
    print(f"Logged to: {LOG_PATH}")

def start_application():
    """Main loop with autonomous timed detection."""
    global last_detection_time
    cap = cv2.VideoCapture(0)
    
    print("\n--- PROTOTYPE STARTING ---")
    print(f"Auto-Detection Frequency: Every {DETECTION_INTERVAL} seconds")
    print("Press 'q' to stop application.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()

        # AUTONOMOUS DETECTION LOGIC
        # Every 'DETECTION_INTERVAL' seconds, the system "thinks"
        if (current_time - last_detection_time) >= DETECTION_INTERVAL:
            
            # --- WEEK 3 INTEGRATION POINT ---
            # This is where Member B's AI model will run.
            # For now, we simulate a 'detection' every 10 seconds 
            # or when you press 's'.
            is_anomaly_found = False # This will be: results.confidence > 0.8
            
            # (SIMULATION ONLY: We auto-detect every 10 seconds for testing)
            if int(current_time) % 10 == 0:
                is_anomaly_found = True
            
            if is_anomaly_found:
                log_anomaly(frame)
            
            last_detection_time = current_time

        # UI Overlay (Live Feedback)
        status_color = (0, 255, 0) if current_location["lat"] != "Waiting..." else (0, 0, 255)
        cv2.putText(frame, f"GPS: {current_location['lat']}, {current_location['lon']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, "STATUS: SCANNING ROAD...", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Road Anomaly Edge AI", frame)

        # Keyboard interrupts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('s'): # Manual override/test
            log_anomaly(frame, "Manual_Trigger")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    setup_environment()
    
    # Start the GPS thread so it doesn't block the camera
    gps_thread = threading.Thread(target=gps_receiver_thread, daemon=True)
    gps_thread.start()
    
    start_application()