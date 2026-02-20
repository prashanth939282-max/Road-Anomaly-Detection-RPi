# Road-Anomaly-Detection-RPi
An Edge AI application for the Raspberry Pi 4 designed to detect road anomalies (potholes and cracks) in real-time (> 5 FPS) using a quantized YOLOv5/MobileNet model.
# Real-Time Road Anomaly Detection (Edge AI)

## 📌 Project Overview
[cite_start]The goal of this project is to develop an Edge AI application that runs on a **Raspberry Pi 4** to identify road damage, such as potholes and cracks, in real-time[cite: 649]. [cite_start]The system aims for an inference speed of at least **5 FPS** using a lightweight, quantized Deep Learning model[cite: 649].

## 👥 Team Roles & Responsibilities
| Role | Member | Key Responsibilities |
| :--- | :--- | :--- |
| **Hardware Lead** | Member A | [cite_start]RPi OS setup, Camera integration, System optimization, Thermal management. |
| **AI/Data Lead** | Member B | [cite_start]Dataset acquisition (Roboflow), Model training (Colab), TFLite Quantization. |
| **Software Lead** | Member C | [cite_start]Python application logic, Data logging (CSV), Logic integration, Repository management. |

## 🛠️ System Architecture
1. [cite_start]**Input**: Real-time video stream (30 FPS) captured via Raspberry Pi Camera Module[cite: 663, 664].
2. [cite_start]**Inference**: Frames are processed by a **YOLOv5 Nano (Int8 Quantized)** TFLite model.
3. **Output**: 
   - [cite_start]Real-time display with detection bounding boxes[cite: 665].
   - [cite_start]**CSV Logger**: Detections are saved to a log file with timestamps[cite: 663, 665].
   - [cite_start]**Snapshots**: High-confidence detection images are saved as evidence[cite: 664, 665].

## 📂 Project Structure
- [cite_start]`/models`: Contains `.pt` model files.
- [cite_start]`/scripts`: Main Python application logic and utilities[cite: 663, 665].
- [cite_start]`/data`: Evidence snapshots of detected anomalies.
- [cite_start]`/logs`: CSV files recording detection history[cite: 663].

## 📅 Target Deadline
- [cite_start]**Submission Date**: February 20, 2026[cite: 646].
