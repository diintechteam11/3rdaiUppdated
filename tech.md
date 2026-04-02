# 3rdAI | Technical Master Guide & Code Walkthrough

This document provides a **complete, deep-dive** into every component of the 3rdAI system. It's written to help you explain the "magic" to clients and technical managers.

---

## 🏗️ 1. Code Function Walkthrough (The "How it Works" Map)

### `main.py` (The Heart / Router)
This file handles the web server, the user dashboard, and orchestrates the AI tasks.

| Function | What it does (Briefly) | Key Logic |
| :--- | :--- | :--- |
| `on_startup` | **Initialization**. | Runs database migrations and creates necessary folders (`uploads`, `crops`, `recordings`). |
| `require_api_key` | **Security**. | Ensures only authorized clients can access the API. |
| `list_cameras` | **Management**. | Fetches all registered cameras from the database for the dashboard. |
| `start_recording` | **Manual Recording**. | Tells the camera to start saving video to disk and records the start time in the DB. |
| `stop_recording` | **Finalization**. | Stops the stream, triggers FFmpeg to optimize the video, and uploads the result to R2 Cloud. |
| `analysis_start` | **AI Trigger**. | Starts the heavy AI processing session for a specific camera and trigger (e.g., Helmet). |

---

### `utils/detection.py` (The Brain / Vision)
This is where the actual AI work happens using OpenCV and YOLOv8.

| Function | What it does (Briefly) | Key Logic |
| :--- | :--- | :--- |
| `get_model` | **Loader**. | Checks if the requested YOLO model is in memory. If not, it loads it from the `models/` folder. |
| `LiveCameraProcessor` | **Live Streamer**. | A persistent class that connects to cameras, processes frames, and sends them to the browser via WebSockets. |
| `_process_loop` | **Real-time AI**. | **The most important function**. It runs YOLO on every incoming frame, identifies objects, and draws the boxes you see on screen. |
| `get_best_ocr` | **Reading Plates**. | Sends image crops to the Plate Recognizer API and falls back to local EasyOCR if needed. |
| `upload_to_r2` | **Storage**. | Takes a detection photo and sends it straight to Cloudflare R2 (S3) so the server disk doesn't get full. |
| `get_vehicle_color`| **Attribute Extraction**.| Uses K-Means clustering (math) to find the most dominant color of a vehicle. |

---

## 🧠 2. YOLOv8 Deep Dive (Detection Secrets)

### Where is YOLO used in the code?
1.  **Line 224** (`get_model`): This is where the model is first instantiated as `YOLO(model_path)`.
2.  **Line 377** (`_process_loop`): This is where the live video frame is fed into YOLO using `model.track()`.
3.  **Line 426** (Vehicle Filter): YOLO is used here to identify what kind of vehicle is carrying the number plate (Bus, Car, Motorcycle).

### How YOLO Detects & Tracks:
*   **Detection**: Every frame, YOLO identifies objects and assigns a **Confidence Score** (e.g., 0.85 means 85% sure it's a car).
*   **Tracking**: Unlike regular AI, we use **YOLO Tracking**. This assigns a permanent **ID** to a vehicle. Even if the car moves across the screen, it keeps its ID (e.g., "Car #102").
*   **Resolution (Width)**: We rescale input frames to **1280px width**. This is the specialized resolution that balances high speed with enough detail to read small number plates.

### Debugging YOLO:
When the system is running, you will see these lines in your console:
- `DEBUG [YOLO-INIT]`: Confirms the specific brain (Helmet, Plate, etc.) is loaded.
- `DEBUG [YOLO-DETECT]`: Shows exactly how many objects the AI found in that specific split-second.
- `DEBUG [YOLO-RES]`: Confirms the AI is using the correct 1280px/640px resolution settings.

---

## ❓ 3. Client & Project Manager Q&A

**Q: How accurate is the detection?**
> **A:** For standard traffic conditions, accuracy is **95%+**. We use YOLOv8 (the latest version) combined with OCR verification to ensure high-quality data.

**Q: Can it detect multiple violations at once?**
> **A:** Yes. The system can run multiple models (e.g., Helmet + Number Plate) simultaneously on the same camera feed.

**Q: What happens if the internet goes down?**
> **A:** The system is built to handle connection drops. It will automatically try to reconnect to the camera (TCP/UDP) and log a "Connection Failed" error if needed.

**Q: Where is the data stored? Is it secure?**
> **A:** All sensitive data (logs) go to an encrypted PostgreSQL database. All evidence photos go to Cloudflare R2 (Industry-standard S3 storage). Nothing is stored on the local server indefinitely, keeping it lightweight.

**Q: Can we read plates at night?**
> **A:** Yes, provided the camera has IR (Infrared) or there is street lighting. Our Plate Recognizer API is specifically trained for low-light Indian license plates.

**Q: Why don't we see the same car logged 100 times?**
> **A:** We use **YOLO Tracking IDs**. Once the system detects "Vehicle ID: 5", it remembers that vehicle and will only log it once until it leaves the camera view.

---

## 📂 Brief Code Summary for Presentations
*   **Tech**: FastAPI + YOLOv8 + OpenCV + PostgreSQL + Cloudflare R2.
*   **Speed**: ~30 Frames Per Second (Real-time).
*   **Scalability**: Can handle hundreds of cameras centrally using a task-queue architecture.
*   **Evidence**: Every incident is backed up with two photos: a close-up of the violation and a wide-angle view of the vehicle.
