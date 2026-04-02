# 3rdAI | Project Intelligence & Debug Guide

This document provides a concise but complete understanding of the **3rdAI Video Analytics** project.

## 🛠 Project Core (The "Big Picture")

**What is it?**  
3rdAI is an AI-powered surveillance and traffic enforcement platform. It processes video streams (RTSP cameras or files) and uses Deep Learning to detect vehicles, identify violations, and record number plates (ANPR).

**The Tech Stack:**
- **AI Brain**: YOLOv8 (Ultralytics) — The fastest real-time object detection model.
- **Vision Engine**: OpenCV — Handles frame capture, resizing, and drawing boxes.
- **Backend**: FastAPI (Python) — A high-performance, asynchronous web server.
- **OCR Engine**: Plate Recognizer Cloud API + EasyOCR — Reads license plates into text.
- **Data Persistence**: PostgreSQL (Database) & Cloudflare R2 (Evidence Storage).
- **Video Delivery**: WebSocket (Real-time frames) & FFmpeg (H.264 Playback).

---

## 🧠 The Models (The AI Brains)

The system is modular, meaning it loads only the "brain" it needs for a specific task:

| Model Name | Purpose | Why this model? |
| :--- | :--- | :--- |
| **ANPR.pt** | Number Plate Detection | Specialized in finding tiny rectangular license plates. |
| **helmet.pt** | Helmet Violation | Detects if a motorcycle rider is safe or not. |
| **triple.pt** | Triple Riding | Identifies if more than 2 people are on a motorcycle. |
| **seatbelt.pt** | Seatbelt Detection | High-precision model looking for shoulder straps in cars. |
| **yolov8n.pt** | Vehicle Detection | A tiny, super-fast model used as a fallback. |

---

## 🔍 How it Works (Function Flow)

### 1. Data Acquisition
The system connects to a camera using several methods (TCP/UDP) to ensure a stable feed. This is the **"Eyes"** of the project.

### 2. Intelligent Tracking
Instead of just seeing a car once, we use **Persistant Tracking**. Every car gets a unique **Track ID** (e.g., `Vehicle #42`). This prevents the system from triggering the same alarm multiple times for the same vehicle.

### 3. OCR & Verification
When a plate is found, we don't just "read" it once. We use a **Regex pattern** (`MH12AB1234`) to ensure it's a valid Indian number plate before saving it to the database.

### 4. Evidence Archive
Every time a violation is "Detected":
- A **Tight Crop** is saved for the plate.
- A **Wide Crop** is saved for the vehicle color/type.
- Both are uploaded to the **Cloud (R2)** permanently.

---

## 🐞 Deep Debugging (How to track the system)

If you need to explain "how it works" to a developer, or find out why something isn't working, follow these **DEBUG** steps in the terminal:

### 1. Connection Diagnostic
- `DEBUG: Trying connection via FFmpeg TCP... SUCCESS!`  
If this fails, it means the RTSP camera is offline or the URL is wrong.

### 2. Detection Verification
- `DEBUG: Loaded model ANPR.pt for Number Plate Detection`  
Confirms the AI model has been successfully loaded into GPU/RAM.

### 3. Database Check
- `DEBUG: SUCCESS! Saved Detection to Database via SQLAlchemy`  
Confirms the internet connection to the PostgreSQL database is healthy.

### 4. Cloud Evidence Sync
- `DEBUG: SUCCESS! Saved to R2: https://...`  
Confirms your S3/R2 credentials are correct and storage is working.

---

## 🎤 How to explain to anyone (Elevator Pitch)

> "3rdAI is an automated traffic police assistant. It watches cameras for you and uses AI (YOLOv8) to instantly detect when someone isn't wearing a helmet or a seatbelt. It even reads their license plate automatically and saves a photo as evidence in the cloud. It turns a regular camera into a smart city enforcement tool."

---

## 📂 Key Files to Show Others
- `utils/detection.py`: The core AI engine.
- `main.py`: The server and API hub.
- `PROJECT_EXPLAINER.html`: The beautiful visual guide.
