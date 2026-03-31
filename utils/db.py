from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

# DB Configuration — all values come from .env, no fallback secrets
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

if not all([DB_HOST, DB_NAME, DB_USER, DB_PASS]):
    raise RuntimeError("Missing required DB environment variables: DB_HOST, DB_NAME, DB_USER, DB_PASS")

SQLALCHEMY_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    ip = Column(String, nullable=False)
    location = Column(String)
    brand = Column(String)
    status = Column(Enum("active", "inactive", "recording", name="camera_status"), default="active")
    is_recording = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    recordings = relationship("RecordingSession", back_populates="camera")
    schedules = relationship("Schedule", back_populates="camera", uselist=False)
    analysis_sessions = relationship("AnalysisSession", back_populates="camera")


class RecordingSession(Base):
    __tablename__ = "recording_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String, ForeignKey("cameras.id"))
    video_name = Column(String)
    file_path = Column(String)
    duration_secs = Column(Integer)
    source = Column(Enum("manual", "auto", "analysis", name="recording_source"), default="manual")
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    stopped_at = Column(DateTime(timezone=True))
    saved_at = Column(DateTime(timezone=True))
    initiated_by = Column(String)
    stopped_by = Column(String)
    description = Column(Text)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    camera = relationship("Camera", back_populates="recordings")
    analysis_session = relationship("AnalysisSession", back_populates="recording_session", uselist=False)


class Schedule(Base):
    __tablename__ = "schedules"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String, ForeignKey("cameras.id"), unique=True)
    mode = Column(Enum("24_7", "custom", name="schedule_mode"), default="24_7")
    is_enabled = Column(Boolean, default=True)
    custom_start_time = Column(String)
    custom_end_time = Column(String)
    timezone = Column(String, default="UTC")
    days_of_week = Column(Text)
    created_by = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    camera = relationship("Camera", back_populates="schedules")


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    camera_id = Column(String, ForeignKey("cameras.id"))
    recording_session_id = Column(String, ForeignKey("recording_sessions.id"), nullable=True)
    analysis_type = Column(String)
    analysis_result = Column(Text)
    capture_started_at = Column(DateTime(timezone=True))
    capture_ended_at = Column(DateTime(timezone=True))
    triggered_by = Column(String)
    stopped_by = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    camera = relationship("Camera", back_populates="analysis_sessions")
    recording_session = relationship("RecordingSession", back_populates="analysis_session")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String)
    filename = Column(String)
    timestamp = Column(Float)
    trigger = Column(String)
    event = Column(Text)
    image_plate_url = Column(String)
    image_object_url = Column(String)
    plate_number = Column(String)
    vehicle_color = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
