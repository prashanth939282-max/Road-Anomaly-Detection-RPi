import cv2          
import pandas as pd 
import os          
from datetime import datetime 


# Defining where logs and images go
LOG_PATH = "../logs/road_anomalies.csv"
DATA_DIR = "../data"

def setup_project_environment():
    """
    Ensures all necessary folders and files exist before starting.
   
    """
    # Create 'data' folder for evidence snapshots if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"[INFO] Created directory: {DATA_DIR}")

    # Create 'logs' folder if it doesn't exist
    if not os.path.exists("../logs"):
        os.makedirs("../logs")

    # Initialize the CSV file with headers if it's missing
    if not os.path.exists(LOG_PATH):
        headers = ["Timestamp", "Anomaly_Type", "Confidence", "Image_Path"]
        df = pd.DataFrame(columns=headers)
        df.to_csv(LOG_PATH, index=False)
        print(f"[INFO] Initialized new log file at: {LOG_PATH}")

def log_detection(anomaly_type, confidence, frame):
    """
    Saves a record of a detection and stores an image snapshot.
    """
    # 1. Generate unique timestamp and filename
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    file_safe_time = now.strftime("%Y%m%d_%H%M%S")
    
    image_name = f"{anomaly_type}_{file_safe_time}.jpg"
    image_full_path = os.path.join(DATA_DIR, image_name)

    # 2. Save the image frame (Evidence Snapshot)
    cv2.imwrite(image_full_path, frame)

    # 3. Update the CSV log using Pandas
    new_entry = {
        "Timestamp": timestamp_str,
        "Anomaly_Type": anomaly_type,
        "Confidence": f"{confidence:.2f}",
        "Image_Path": image_full_path
    }
    
    # Read existing, append new, and save back
    df = pd.read_csv(LOG_PATH)
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)
    
    print(f"[DETECTION] {anomaly_type} found! Logged to CSV and saved image.")

def run_detection_loop():
    """
    Main loop to capture video. Member B will add AI logic here in Week 3.
    """
    # Initialize camera (0 is usually the default RPi camera or USB web cam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open camera. Check hardware connection.")
        return

    print("[STATUS] Application Skeleton Running. Press 's' to simulate a detection, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # --- PLACEHOLDER FOR WEEK 3 (Member B's Model) ---
        # Currently, we just show the camera feed. 
        # Later, 'frame' will be passed to the TFLite model.
        
        # Display the live video feed
        cv2.imshow("Road Anomaly Detection - Prototype", frame)

        # KEYBOARD COMMANDS FOR TESTING
        key = cv2.waitKey(1) & 0xFF
        
        # Simulate a detection manually for testing the logger (Press 's')
        if key == ord('s'):
            log_detection("Pothole_Test", 0.95, frame)

        # Quit the application (Press 'q')
        if key == ord('q'):
            print("[STATUS] Closing application...")
            break

    # Cleanup hardware resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Start the setup, then the loop
    setup_project_environment()
    run_detection_loop()