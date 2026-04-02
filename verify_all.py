from utils.db import SessionLocal, RecordingSession, Camera
from sqlalchemy import desc
import os

def verify():
    db = SessionLocal()
    print("--- RECENT RECORDING SESSIONS ---")
    recs = db.query(RecordingSession).order_by(desc(RecordingSession.started_at)).limit(5).all()
    
    if not recs:
        print("No recordings found in DB.")
        return

    for r in recs:
        print(f"\nID: {r.id}")
        print(f"Name: {r.video_name}")
        print(f"Path: {r.file_path}")
        print(f"Started: {r.started_at}")
        print(f"Duration: {r.duration_secs}s")
        
        # Check if local file exists
        if r.file_path and not r.file_path.startswith("http"):
            # Construct absolute path similar to main.py
            # /static/recordings/camera_id/filename
            parts = r.file_path.split("/")
            if len(parts) >= 4:
                # e.g. static/recordings/cam-id/file.mp4
                local_path = os.path.join(os.getcwd(), "static", "recordings", parts[3], parts[4])
                print(f"Checked Local Path: {local_path}")
                if os.path.exists(local_path):
                    print("Status: ✅ Local File Exists")
                else:
                    print(f"Status: ❌ Local File Missing (Expected at {local_path})")
                    # Check for _raw version
                    raw = local_path.replace(".mp4", "_raw.mp4")
                    if os.path.exists(raw):
                        print("Status: ⚠️ Found RAW version only! (Transcode failed?)")
        elif r.file_path and r.file_path.startswith("http"):
             print(f"Status: ☁️ Cloud URL detected (R2)")
    
    db.close()

if __name__ == "__main__":
    verify()
