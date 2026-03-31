from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request, Form, WebSocket, WebSocketDisconnect, Depends, Header, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone
import os, uuid, json, re, time
from pathlib import Path
import cv2

from utils.detection import process_video, LiveCameraProcessor
from utils.db import init_db, get_db, Camera, RecordingSession, Schedule, AnalysisSession, Detection
from utils.r2 import upload_video_to_r2, delete_from_r2, build_r2_key
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, desc, asc, text
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AI Camera Recording & Analytics API", version="2.0")

# ─── Startup ────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    init_db()
    _migrate_db()


def _migrate_db():
    """Run any needed ALTER TABLE migrations safely (idempotent)."""
    from utils.db import engine
    migrations = [
        # Add deleted_at to recording_sessions if missing
        "ALTER TABLE recording_sessions ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP WITH TIME ZONE;",
        # Add updated_at / created_at if missing
        "ALTER TABLE recording_sessions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();",
        "ALTER TABLE cameras ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();",
    ]
    with engine.connect() as conn:
        for sql in migrations:
            try:
                conn.execute(text(sql))
                conn.commit()
            except Exception as e:
                print(f"[Migration] Skipped (already exists or error): {e}")

# ─── Directories ────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
UPLOAD_DIR     = BASE_DIR / "static" / "uploads"
OUTPUT_DIR     = BASE_DIR / "static" / "outputs"
CROPS_DIR      = BASE_DIR / "static" / "crops"
RECORDINGS_DIR = BASE_DIR / "static" / "recordings"
TEMPLATES_DIR  = BASE_DIR / "templates"
API_KEYS_FILE  = BASE_DIR / "api_keys.json"

for d in [UPLOAD_DIR, OUTPUT_DIR, CROPS_DIR, RECORDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ─── In-memory state (existing AI detection system) ─────────────────────────
processing_tasks: dict = {}
camera_processes: dict = {}

# ─── Helpers ────────────────────────────────────────────────────────────────
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def iso(dt) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def load_keys():
    if not API_KEYS_FILE.exists() or API_KEYS_FILE.stat().st_size == 0:
        return {}
    try:
        with open(API_KEYS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def save_keys(keys):
    with open(API_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=4)

if not API_KEYS_FILE.exists() or API_KEYS_FILE.stat().st_size == 0:
    save_keys({})

def err(code: str, message: str, status: int = 400):
    return JSONResponse(
        status_code=status,
        content={"error": True, "code": code, "message": message, "timestamp": iso(utc_now())}
    )

# ─── Auth ────────────────────────────────────────────────────────────────────
async def require_api_key(x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key. Include X-API-Key header.")
    keys = load_keys()
    if x_api_key not in keys:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    keys[x_api_key]["usage"] = keys[x_api_key].get("usage", 0) + 1
    save_keys(keys)
    return x_api_key

async def optional_api_key(x_api_key: str = Header(None)):
    """For browser-facing pages: don't block if no key."""
    if x_api_key:
        keys = load_keys()
        if x_api_key not in keys:
            raise HTTPException(status_code=403, detail="Invalid API key.")
    return x_api_key

# ─── Pydantic Schemas ────────────────────────────────────────────────────────
class RegisterCameraBody(BaseModel):
    name: str
    ip: str
    location: str
    brand: Optional[str] = None

class StartRecordingBody(BaseModel):
    initiated_by: str
    note: Optional[str] = None

class StopRecordingBody(BaseModel):
    stopped_by: str
    name: Optional[str] = None
    description: Optional[str] = None

class ScheduleBody(BaseModel):
    mode: str
    is_enabled: bool
    custom_start_time: Optional[str] = None
    custom_end_time: Optional[str] = None
    timezone: str = "UTC"
    days_of_week: Optional[List[str]] = None
    created_by: str

class AnalysisStartBody(BaseModel):
    analysis_type: str
    triggered_by: str
    analysis_id: Optional[str] = None

class AnalysisStopBody(BaseModel):
    analysis_session_id: str
    analysis_result: Optional[str] = None
    video_name: Optional[str] = None
    stopped_by: str

# ─── Page routes ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/cameras-dashboard", response_class=HTMLResponse)
async def cameras_dashboard(request: Request):
    return templates.TemplateResponse(request, "cameras.html")

@app.get("/docs-page", response_class=HTMLResponse)
async def documentation(request: Request):
    return templates.TemplateResponse(request, "docs.html")

@app.get("/api-access", response_class=HTMLResponse)
async def api_access(request: Request):
    return templates.TemplateResponse(request, "api.html")

@app.get("/playground", response_class=HTMLResponse)
async def playground(request: Request):
    return templates.TemplateResponse(request, "playground.html")

@app.post("/generate-api-key")
async def generate_api_key():
    new_key = f"sk-{uuid.uuid4().hex}"
    keys = load_keys()
    keys[new_key] = {"created_at": iso(utc_now()), "usage": 0}
    save_keys(keys)
    return {"api_key": new_key}

# ═══════════════════════════════════════════════════════════════════════════════
# CAMERA MANAGEMENT  /api/v1/cameras
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/cameras")
async def list_cameras(
    search: Optional[str] = None,
    status: Optional[str] = None,
    location: Optional[str] = None,
    brand: Optional[str] = None,
    sort: str = "latest",
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    _key=Depends(require_api_key),
):
    page_size = min(page_size, 100)
    q = db.query(Camera)
    if search:
        q = q.filter(or_(Camera.name.ilike(f"%{search}%"), Camera.location.ilike(f"%{search}%")))
    if status:
        q = q.filter(Camera.status == status)
    if location:
        q = q.filter(Camera.location.ilike(f"%{location}%"))
    if brand:
        q = q.filter(Camera.brand.ilike(f"%{brand}%"))
    if sort == "oldest":
        q = q.order_by(asc(Camera.created_at))
    elif sort == "name_asc":
        q = q.order_by(asc(Camera.name))
    elif sort == "name_desc":
        q = q.order_by(desc(Camera.name))
    else:
        q = q.order_by(desc(Camera.created_at))
    total = q.count()
    cameras = q.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "cameras": [_fmt_camera(c) for c in cameras],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.get("/api/v1/cameras/{camera_id}")
async def get_camera(camera_id: str, db: Session = Depends(get_db), _key=Depends(require_api_key)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        return err("CAMERA_NOT_FOUND", f"Camera {camera_id} not found.", 404)
    return _fmt_camera(cam)


@app.post("/api/v1/cameras", status_code=201)
async def register_camera(body: RegisterCameraBody, db: Session = Depends(get_db), _key=Depends(require_api_key)):
    cam = Camera(
        id=str(uuid.uuid4()),
        name=body.name,
        ip=body.ip,
        location=body.location,
        brand=body.brand,
        status="active",
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return _fmt_camera(cam)


class UpdateCameraBody(BaseModel):
    name: Optional[str] = None
    ip: Optional[str] = None
    location: Optional[str] = None
    brand: Optional[str] = None


@app.put("/api/v1/cameras/{camera_id}")
async def update_camera(camera_id: str, body: UpdateCameraBody, db: Session = Depends(get_db), _key=Depends(require_api_key)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        return err("CAMERA_NOT_FOUND", f"Camera {camera_id} not found.", 404)
    if body.name is not None:
        cam.name = body.name
    if body.ip is not None:
        cam.ip = body.ip
    if body.location is not None:
        cam.location = body.location
    if body.brand is not None:
        cam.brand = body.brand
    db.commit()
    db.refresh(cam)
    return _fmt_camera(cam)


@app.delete("/api/v1/cameras/{camera_id}", status_code=204)
async def delete_camera(camera_id: str, db: Session = Depends(get_db), _key=Depends(require_api_key)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        return err("CAMERA_NOT_FOUND", f"Camera {camera_id} not found.", 404)
    # Stop any active recording
    if cam.is_recording:
        cam.is_recording = False
        cam.status = "inactive"
    db.delete(cam)
    db.commit()
    return JSONResponse(status_code=204, content=None)


def _fmt_camera(c: Camera) -> dict:
    return {
        "id": c.id,
        "name": c.name,
        "location": c.location,
        "status": c.status,
        "ip": c.ip,
        "brand": c.brand,
        "is_recording": c.is_recording,
        "created_at": iso(c.created_at),
        "updated_at": iso(c.updated_at),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RECORDING CONTROL
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/cameras/{camera_id}/recording/start", status_code=201)
async def start_recording(camera_id: str, body: StartRecordingBody, db: Session = Depends(get_db), _key=Depends(require_api_key)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        return err("CAMERA_NOT_FOUND", f"Camera {camera_id} not found.", 404)
    if cam.is_recording:
        return err("ALREADY_RECORDING", "Camera is already recording.", 409)

    session_id = str(uuid.uuid4())
    now = utc_now()

    session = RecordingSession(
        id=session_id,
        camera_id=camera_id,
        source="manual",
        initiated_by=body.initiated_by,
        description=body.note,
        started_at=now,
    )
    db.add(session)
    cam.is_recording = True
    cam.status = "recording"
    db.commit()

    # Also kick off live processor if camera is in camera_processes
    if camera_id in camera_processes:
        camera_processes[camera_id]["processor"].start_recording(
            initiated_by=body.initiated_by, note=body.note, source="manual"
        )

    return {
        "session_id": session_id,
        "camera_id": camera_id,
        "status": "recording",
        "started_at": iso(now),
        "initiated_by": body.initiated_by,
    }


@app.post("/api/v1/cameras/{camera_id}/recording/stop")
async def stop_recording(camera_id: str, body: StopRecordingBody, db: Session = Depends(get_db), _key=Depends(require_api_key)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        return err("CAMERA_NOT_FOUND", f"Camera {camera_id} not found.", 404)
    if not cam.is_recording:
        return err("NOT_RECORDING", "Camera is not currently recording.", 409)

    session = (
        db.query(RecordingSession)
        .filter(RecordingSession.camera_id == camera_id, RecordingSession.stopped_at == None)
        .order_by(desc(RecordingSession.started_at))
        .first()
    )
    if not session:
        return err("SESSION_NOT_FOUND", "No active recording session found.", 404)

    now = utc_now()
    duration = int((now - session.started_at.replace(tzinfo=timezone.utc)).total_seconds()) if session.started_at else 0

    auto_name = _auto_filename(cam.name)
    video_name = body.name or auto_name

    r2_path = None
    local_path = RECORDINGS_DIR / camera_id / auto_name
    if local_path.exists():
        try:
            key = build_r2_key(cam.name, video_name)
            r2_path = upload_video_to_r2(str(local_path), key)
        except Exception as e:
            print(f"R2 upload failed: {e}")

    session.stopped_at = now
    session.saved_at = now
    session.duration_secs = duration
    session.stopped_by = body.stopped_by
    session.video_name = video_name
    session.file_path = r2_path or f"/static/recordings/{camera_id}/{auto_name}"
    if body.description:
        session.description = body.description

    cam.is_recording = False
    cam.status = "active"
    db.commit()

    if camera_id in camera_processes:
        camera_processes[camera_id]["processor"].stop_recording(stopped_by=body.stopped_by)

    return {
        "recording_id": session.id,
        "camera_id": camera_id,
        "video_name": video_name,
        "file_path": session.file_path,
        "duration_secs": duration,
        "started_at": iso(session.started_at),
        "stopped_at": iso(now),
        "saved_at": iso(now),
        "stopped_by": body.stopped_by,
    }


@app.get("/api/v1/cameras/{camera_id}/recording/status")
async def recording_status(camera_id: str, db: Session = Depends(get_db), _key=Depends(require_api_key)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        return err("CAMERA_NOT_FOUND", f"Camera {camera_id} not found.", 404)

    active = None
    if cam.is_recording:
        active = (
            db.query(RecordingSession)
            .filter(RecordingSession.camera_id == camera_id, RecordingSession.stopped_at == None)
            .order_by(desc(RecordingSession.started_at))
            .first()
        )

    elapsed = None
    if active and active.started_at:
        elapsed = int((utc_now() - active.started_at.replace(tzinfo=timezone.utc)).total_seconds())

    return {
        "camera_id": camera_id,
        "is_recording": cam.is_recording,
        "session_id": active.id if active else None,
        "started_at": iso(active.started_at) if active else None,
        "duration_secs": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEO / RECORDING MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/v1/cameras/{camera_id}/recordings")
async def list_camera_recordings(
    camera_id: str,
    search: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort: str = "latest",
    duration_min: Optional[int] = None,
    duration_max: Optional[int] = None,
    source: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    _key=Depends(require_api_key),
):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        return err("CAMERA_NOT_FOUND", f"Camera {camera_id} not found.", 404)
    return _query_recordings(db, camera_id=camera_id, search=search, date_from=date_from,
                              date_to=date_to, sort=sort, duration_min=duration_min,
                              duration_max=duration_max, source=source, page=page, page_size=page_size)


@app.get("/api/v1/recordings")
async def list_all_recordings(
    camera_id: Optional[str] = None,
    search: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort: str = "latest",
    source: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    _key=Depends(require_api_key),
):
    return _query_recordings(db, camera_id=camera_id, search=search, date_from=date_from,
                              date_to=date_to, sort=sort, source=source, page=page, page_size=page_size)


@app.get("/api/v1/recordings/{recording_id}")
async def get_recording(recording_id: str, db: Session = Depends(get_db), _key=Depends(require_api_key)):
    rec = db.query(RecordingSession).filter(RecordingSession.id == recording_id).first()
    if not rec:
        return err("RECORDING_NOT_FOUND", f"Recording {recording_id} not found.", 404)
    # Soft-delete check: if deleted_at column exists, filter it
    try:
        if rec.deleted_at is not None:
            return err("RECORDING_NOT_FOUND", f"Recording {recording_id} not found.", 404)
    except Exception:
        pass
    cam = db.query(Camera).filter(Camera.id == rec.camera_id).first()
    return {**_fmt_recording(rec), "camera_name": cam.name if cam else None, "description": rec.description}


@app.delete("/api/v1/recordings/{recording_id}", status_code=204)
async def delete_recording(
    recording_id: str,
    delete_file: bool = False,
    db: Session = Depends(get_db),
    _key=Depends(require_api_key),
):
    rec = db.query(RecordingSession).filter(RecordingSession.id == recording_id).first()
    if not rec:
        return err("RECORDING_NOT_FOUND", f"Recording {recording_id} not found.", 404)
    if delete_file and rec.file_path and rec.file_path.startswith("http"):
        key = "/".join(rec.file_path.split("/")[-3:])
        delete_from_r2(key)
    try:
        rec.deleted_at = utc_now()
    except Exception:
        pass
    try:
        db.delete(rec)
    except Exception:
        pass
    db.commit()
    return JSONResponse(status_code=204, content=None)


def _query_recordings(db, camera_id=None, search=None, date_from=None, date_to=None,
                       sort="latest", duration_min=None, duration_max=None, source=None,
                       page=1, page_size=20):
    page_size = min(page_size, 100)
    q = db.query(RecordingSession)
    # Safely filter out soft-deleted records if column exists
    try:
        q = q.filter(RecordingSession.deleted_at == None)
    except Exception:
        pass
    if camera_id:
        q = q.filter(RecordingSession.camera_id == camera_id)
    if search:
        q = q.filter(or_(RecordingSession.video_name.ilike(f"%{search}%"), RecordingSession.description.ilike(f"%{search}%")))
    if date_from:
        q = q.filter(RecordingSession.started_at >= date_from)
    if date_to:
        q = q.filter(RecordingSession.started_at <= date_to + "T23:59:59Z")
    if duration_min is not None:
        q = q.filter(RecordingSession.duration_secs >= duration_min)
    if duration_max is not None:
        q = q.filter(RecordingSession.duration_secs <= duration_max)
    if source:
        q = q.filter(RecordingSession.source == source)
    q = q.order_by(asc(RecordingSession.started_at) if sort == "oldest" else desc(RecordingSession.started_at))
    total = q.count()
    recs = q.offset((page - 1) * page_size).limit(page_size).all()
    return {"recordings": [_fmt_recording(r) for r in recs], "total": total, "page": page, "page_size": page_size}


def _fmt_recording(r: RecordingSession) -> dict:
    return {
        "id": r.id,
        "camera_id": r.camera_id,
        "video_name": r.video_name,
        "file_path": r.file_path,
        "duration_secs": r.duration_secs,
        "source": r.source,
        "started_at": iso(r.started_at),
        "stopped_at": iso(r.stopped_at),
        "saved_at": iso(r.saved_at),
        "created_at": iso(r.created_at),
        "updated_at": iso(r.updated_at),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULING  /api/v1/cameras/{id}/schedule
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/cameras/{camera_id}/schedule", status_code=201)
async def create_or_update_schedule(camera_id: str, body: ScheduleBody, db: Session = Depends(get_db), _key=Depends(require_api_key)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        return err("CAMERA_NOT_FOUND", f"Camera {camera_id} not found.", 404)
    if body.mode == "custom" and (not body.custom_start_time or not body.custom_end_time):
        return err("VALIDATION_ERROR", "custom_start_time and custom_end_time are required when mode=custom.", 422)

    sched = db.query(Schedule).filter(Schedule.camera_id == camera_id).first()
    if sched:
        sched.mode = body.mode
        sched.is_enabled = body.is_enabled
        sched.custom_start_time = body.custom_start_time if body.mode == "custom" else None
        sched.custom_end_time = body.custom_end_time if body.mode == "custom" else None
        sched.timezone = body.timezone
        sched.days_of_week = json.dumps(body.days_of_week) if body.days_of_week else None
    else:
        sched = Schedule(
            id=str(uuid.uuid4()),
            camera_id=camera_id,
            mode=body.mode,
            is_enabled=body.is_enabled,
            custom_start_time=body.custom_start_time if body.mode == "custom" else None,
            custom_end_time=body.custom_end_time if body.mode == "custom" else None,
            timezone=body.timezone,
            days_of_week=json.dumps(body.days_of_week) if body.days_of_week else None,
            created_by=body.created_by,
        )
        db.add(sched)
    db.commit()
    db.refresh(sched)
    return _fmt_schedule(sched)


@app.get("/api/v1/cameras/{camera_id}/schedule")
async def get_schedule(camera_id: str, db: Session = Depends(get_db), _key=Depends(require_api_key)):
    sched = db.query(Schedule).filter(Schedule.camera_id == camera_id).first()
    if not sched:
        return err("SCHEDULE_NOT_FOUND", f"No schedule found for camera {camera_id}.", 404)
    return _fmt_schedule(sched)


@app.delete("/api/v1/cameras/{camera_id}/schedule", status_code=204)
async def delete_schedule(camera_id: str, db: Session = Depends(get_db), _key=Depends(require_api_key)):
    sched = db.query(Schedule).filter(Schedule.camera_id == camera_id).first()
    if not sched:
        return err("SCHEDULE_NOT_FOUND", f"No schedule found for camera {camera_id}.", 404)
    db.delete(sched)
    db.commit()
    return JSONResponse(status_code=204, content=None)


@app.get("/api/v1/schedules")
async def list_all_schedules(
    mode: Optional[str] = None,
    is_enabled: Optional[bool] = None,
    camera_id: Optional[str] = None,
    db: Session = Depends(get_db),
    _key=Depends(require_api_key),
):
    q = db.query(Schedule)
    if mode:
        q = q.filter(Schedule.mode == mode)
    if is_enabled is not None:
        q = q.filter(Schedule.is_enabled == is_enabled)
    if camera_id:
        q = q.filter(Schedule.camera_id == camera_id)
    return {"schedules": [_fmt_schedule(s) for s in q.all()]}


def _fmt_schedule(s: Schedule) -> dict:
    return {
        "schedule_id": s.id,
        "camera_id": s.camera_id,
        "mode": s.mode,
        "is_enabled": s.is_enabled,
        "custom_start_time": s.custom_start_time,
        "custom_end_time": s.custom_end_time,
        "timezone": s.timezone,
        "days_of_week": json.loads(s.days_of_week) if s.days_of_week else None,
        "created_by": s.created_by,
        "created_at": iso(s.created_at),
        "updated_at": iso(s.updated_at),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# AI ANALYSIS TRIGGER
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/cameras/{camera_id}/analysis/start", status_code=201)
async def analysis_start(camera_id: str, body: AnalysisStartBody, db: Session = Depends(get_db), _key=Depends(require_api_key)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        return err("CAMERA_NOT_FOUND", f"Camera {camera_id} not found.", 404)
    if cam.is_recording:
        return err("ALREADY_RECORDING", "Camera is already recording.", 409)

    now = utc_now()
    session_id = str(uuid.uuid4())
    analysis_id = body.analysis_id or str(uuid.uuid4())

    # Create analysis session
    analysis = AnalysisSession(
        id=analysis_id,
        camera_id=camera_id,
        analysis_type=body.analysis_type,
        triggered_by=body.triggered_by,
        capture_started_at=now,
    )
    db.add(analysis)

    # Create linked recording session
    rec = RecordingSession(
        id=session_id,
        camera_id=camera_id,
        source="analysis",
        initiated_by=body.triggered_by,
        started_at=now,
    )
    db.add(rec)
    db.flush()
    analysis.recording_session_id = session_id
    cam.is_recording = True
    cam.status = "recording"
    db.commit()

    if camera_id in camera_processes:
        camera_processes[camera_id]["processor"].start_recording(
            initiated_by=body.triggered_by, source="analysis", analysis_session_id=analysis_id
        )

    return {
        "session_id": session_id,
        "analysis_session_id": analysis_id,
        "camera_id": camera_id,
        "status": "capturing",
        "analysis_type": body.analysis_type,
        "capture_started_at": iso(now),
    }


@app.post("/api/v1/cameras/{camera_id}/analysis/stop")
async def analysis_stop(camera_id: str, body: AnalysisStopBody, db: Session = Depends(get_db), _key=Depends(require_api_key)):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        return err("CAMERA_NOT_FOUND", f"Camera {camera_id} not found.", 404)

    analysis = db.query(AnalysisSession).filter(AnalysisSession.id == body.analysis_session_id).first()
    if not analysis:
        return err("ANALYSIS_NOT_FOUND", f"Analysis session {body.analysis_session_id} not found.", 404)

    now = utc_now()
    rec = db.query(RecordingSession).filter(RecordingSession.id == analysis.recording_session_id).first()
    duration = 0
    if rec and rec.started_at:
        duration = int((now - rec.started_at.replace(tzinfo=timezone.utc)).total_seconds())
        auto_name = _auto_filename(cam.name)
        video_name = body.video_name or auto_name

        r2_path = None
        local_path = RECORDINGS_DIR / camera_id / auto_name
        if local_path.exists():
            try:
                key = build_r2_key(cam.name, video_name)
                r2_path = upload_video_to_r2(str(local_path), key)
            except Exception as e:
                print(f"R2 upload failed: {e}")

        rec.stopped_at = now
        rec.saved_at = now
        rec.duration_secs = duration
        rec.stopped_by = body.stopped_by
        rec.video_name = video_name
        rec.file_path = r2_path or f"/static/recordings/{camera_id}/{auto_name}"

    analysis.capture_ended_at = now
    analysis.stopped_by = body.stopped_by
    if body.analysis_result:
        analysis.analysis_result = body.analysis_result

    cam.is_recording = False
    cam.status = "active"
    db.commit()

    if camera_id in camera_processes:
        camera_processes[camera_id]["processor"].stop_recording(stopped_by=body.stopped_by)

    return {
        "recording_id": rec.id if rec else None,
        "analysis_session_id": analysis.id,
        "video_name": rec.video_name if rec else None,
        "capture_started_at": iso(analysis.capture_started_at),
        "capture_ended_at": iso(now),
        "duration_secs": duration,
        "source": "analysis",
        "analysis_result": body.analysis_result,
        "saved_at": iso(now),
    }


@app.get("/api/v1/cameras/{camera_id}/analysis/sessions")
async def list_analysis_sessions(
    camera_id: str,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort: str = "latest",
    analysis_type: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    _key=Depends(require_api_key),
):
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        return err("CAMERA_NOT_FOUND", f"Camera {camera_id} not found.", 404)
    page_size = min(page_size, 100)
    q = db.query(AnalysisSession).filter(AnalysisSession.camera_id == camera_id)
    if date_from:
        q = q.filter(AnalysisSession.capture_started_at >= date_from)
    if date_to:
        q = q.filter(AnalysisSession.capture_started_at <= date_to + "T23:59:59Z")
    if analysis_type:
        q = q.filter(AnalysisSession.analysis_type == analysis_type)
    q = q.order_by(asc(AnalysisSession.capture_started_at) if sort == "oldest" else desc(AnalysisSession.capture_started_at))
    total = q.count()
    sessions = q.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "sessions": [_fmt_analysis(s) for s in sessions],
        "total": total, "page": page, "page_size": page_size,
    }


def _fmt_analysis(a: AnalysisSession) -> dict:
    return {
        "id": a.id,
        "camera_id": a.camera_id,
        "recording_session_id": a.recording_session_id,
        "analysis_type": a.analysis_type,
        "analysis_result": a.analysis_result,
        "capture_started_at": iso(a.capture_started_at),
        "capture_ended_at": iso(a.capture_ended_at),
        "triggered_by": a.triggered_by,
        "stopped_by": a.stopped_by,
        "created_at": iso(a.created_at),
        "updated_at": iso(a.updated_at),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY — existing AI detection & video upload (unchanged functionality)
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_filename(camera_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]", "_", camera_name.lower())
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{safe}_{ts}.mp4"


def _bg_process_video(task_id, input_path, output_path, selected_triggers):
    try:
        processing_tasks[task_id]["status"] = "processing"
        logs = process_video(task_id, input_path, output_path, selected_triggers)
        processing_tasks[task_id].update({
            "status": "completed",
            "logs": logs,
            "video_url": f"/static/outputs/{os.path.basename(output_path)}",
        })
    except Exception as e:
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["error"] = str(e)


@app.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    triggers: str = Form(""),
    x_api_key: str = Header(None),
):
    await optional_api_key(x_api_key)
    if not video_file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        return JSONResponse({"error": "Invalid video format"}, status_code=400)
    selected_triggers = [t.strip() for t in triggers.split(",") if t.strip()]
    if not selected_triggers:
        return JSONResponse({"error": "No triggers selected"}, status_code=400)
    task_id = str(uuid.uuid4())
    input_path = str(UPLOAD_DIR / f"{task_id}_{video_file.filename}")
    output_path = str(OUTPUT_DIR / f"processed_{task_id}.mp4")
    with open(input_path, "wb") as f:
        f.write(await video_file.read())
    processing_tasks[task_id] = {"status": "queued", "video_url": None, "logs": [], "id": task_id, "filename": video_file.filename}
    background_tasks.add_task(_bg_process_video, task_id, input_path, output_path, selected_triggers)
    return {"message": "Upload successful, processing started.", "task_id": task_id}


@app.get("/video-result/{task_id}")
async def get_video_result(task_id: str):
    task = processing_tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return task


@app.get("/logs/{task_id}")
async def get_logs(task_id: str):
    task = processing_tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    return {"logs": task.get("logs", [])}


@app.post("/connect-camera")
async def connect_camera(name: str = Form(...), link: str = Form(...), triggers: str = Form("")):
    camera_id = f"cam-{uuid.uuid4().hex[:8]}"
    selected_triggers = [t.strip() for t in triggers.split(",") if t.strip()]
    if not selected_triggers:
        return JSONResponse({"error": "No triggers selected"}, status_code=400)
    processor = LiveCameraProcessor(camera_id, link, selected_triggers)
    camera_processes[camera_id] = {"id": camera_id, "name": name, "link": link, "triggers": selected_triggers, "processor": processor}
    return {"message": "Camera connection initiated", "camera_id": camera_id}


@app.get("/camera-status/{camera_id}")
async def get_camera_status(camera_id: str):
    if camera_id not in camera_processes:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return {"status": camera_processes[camera_id]["processor"].status}


@app.get("/camera-stream/{camera_id}")
async def camera_stream(camera_id: str):
    from fastapi.responses import StreamingResponse
    if camera_id not in camera_processes:
        raise HTTPException(status_code=404, detail="Camera not found")
    processor = camera_processes[camera_id]["processor"]
    def generate():
        while True:
            if processor.latest_jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + processor.latest_jpeg + b"\r\n"
            time.sleep(0.04)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws-camera/{camera_id}")
async def ws_camera_stream(websocket: WebSocket, camera_id: str):
    import asyncio
    if camera_id not in camera_processes:
        # Accept then close gracefully with a reason code instead of 403
        await websocket.accept()
        await websocket.send_text('{"error": "Camera session not found. Please reconnect."}')
        await websocket.close(code=1008)
        return
    await websocket.accept()
    processor = camera_processes[camera_id]["processor"]
    try:
        while True:
            if processor.latest_jpeg:
                await websocket.send_bytes(processor.latest_jpeg)
            await asyncio.sleep(0.04)
    except (WebSocketDisconnect, Exception):
        pass


@app.get("/camera-logs/{camera_id}")
async def get_camera_logs(camera_id: str):
    if camera_id not in camera_processes:
        # Return empty logs instead of 404 to avoid frontend errors
        return {"logs": [], "status": "session_lost"}
    return {"logs": camera_processes[camera_id]["processor"].logs}


@app.post("/disconnect-camera/{camera_id}")
async def disconnect_camera(camera_id: str):
    if camera_id in camera_processes:
        camera_processes[camera_id]["processor"].stop()
        del camera_processes[camera_id]
        return {"message": "Camera disconnected"}
    return {"message": "Camera already disconnected or not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
