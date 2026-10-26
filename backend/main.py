
import os
import sys
import uuid
import io
import re
import hashlib
import asyncio
import subprocess
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional, Literal

from database import Base, engine, get_db
from models import Violation
from schemas import ViolationCreate, ViolationResponse, ViolationStatusUpdate

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Intelligent Traffic Monitoring System",
    description="Traffic violation detection and monitoring backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "*"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

EVIDENCE_DIR = Path(__file__).resolve().parent / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "dataset" / "uploads"
PROCESSED_VIDEO_DIR = PROJECT_ROOT / "dataset" / "processed"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# In-memory deduplication registry to prevent duplicate records if a video upload is retried
PROCESSED_VIDEO_HASHES: set[str] = set()


@app.get("/")
def home():
    return {"message": "Traffic Monitoring Backend is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# 1. Create a violation
@app.post("/api/violations", response_model=ViolationResponse, status_code=201)
def create_violation(
    violation_data: ViolationCreate,
    db: Session = Depends(get_db)
):
    violation = Violation(**violation_data.model_dump())

    try:
        db.add(violation)
        db.commit()
        db.refresh(violation)
        return violation
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to save violation"
        )


# 2. Retrieve violations with optional filters
@app.get("/api/violations", response_model=list[ViolationResponse])
def get_violations(
    vehicle_number: Optional[str] = None,
    violation_type: Optional[
        Literal["no_helmet", "triple_riding", "red_light", "stop_line"]
    ] = None,
    status: Optional[
        Literal["pending", "approved", "rejected"]
    ] = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    query = db.query(Violation)

    if vehicle_number:
        query = query.filter(
            Violation.vehicle_number.ilike(f"%{vehicle_number}%")
        )

    if violation_type:
        query = query.filter(
            Violation.violation_type == violation_type
        )

    if status:
        query = query.filter(Violation.status == status)

    return query.order_by(
        Violation.id.desc()
    ).limit(limit).all()


# 3. Retrieve one violation
@app.get("/api/violations/{violation_id}", response_model=ViolationResponse)
def get_violation(
    violation_id: int,
    db: Session = Depends(get_db)
):
    violation = db.query(Violation).filter(
        Violation.id == violation_id
    ).first()

    if violation is None:
        raise HTTPException(status_code=404, detail="Violation not found")

    return violation


# 4. Approve or reject a violation
@app.patch("/api/violations/{violation_id}/status",
           response_model=ViolationResponse)
def update_violation_status(
    violation_id: int,
    update_data: ViolationStatusUpdate,
    db: Session = Depends(get_db)
):
    violation = db.query(Violation).filter(
        Violation.id == violation_id
    ).first()

    if violation is None:
        raise HTTPException(status_code=404, detail="Violation not found")

    violation.status = update_data.status
    try:
        db.commit()
        db.refresh(violation)
        return violation
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to update violation status"
        )


# 5. Summary statistics
@app.get("/api/analytics")
def get_analytics(db: Session = Depends(get_db)):
    violations = db.query(Violation).all()

    total = len(violations)
    pending = sum(v.status == "pending" for v in violations)
    approved = sum(v.status == "approved" for v in violations)
    rejected = sum(v.status == "rejected" for v in violations)

    by_type = {}
    for violation in violations:
        by_type[violation.violation_type] = (
            by_type.get(violation.violation_type, 0) + 1
        )

    return {
        "total": total,
        "total_violations": total,
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "by_type": by_type
    }


# 6. Upload violation evidence image
@app.post("/api/violations/{violation_id}/evidence")
async def upload_evidence(
    violation_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    violation = db.query(Violation).filter(
        Violation.id == violation_id
    ).first()
    if violation is None:
        raise HTTPException(
            status_code=404,
            detail="Violation not found"
        )

    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are allowed"
        )

    # Read at most 5 MB + 1 byte to enforce the limit
    content = await file.read(MAX_FILE_SIZE + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 5 MB limit"
        )

    # Verify that the uploaded file is a valid image using Pillow
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file or corrupted image"
        )

    # Calculate SHA-256 fingerprint from the validated content bytes
    evidence_hash = hashlib.sha256(content).hexdigest()

    # Save to evidence directory with a unique filename
    filename = f"{uuid.uuid4().hex}{extension}"
    file_path = EVIDENCE_DIR / filename
    with open(file_path, "wb") as f:
        f.write(content)

    # Save relative path and hash to DB
    violation.evidence_path = f"evidence/{filename}"
    violation.evidence_hash = evidence_hash

    try:
        db.commit()
        db.refresh(violation)
    except SQLAlchemyError:
        db.rollback()
        if file_path.is_file():
            file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail="Failed to save evidence to database"
        )

    return {
        "message": "Evidence uploaded successfully",
        "violation_id": violation.id,
        "evidence_path": violation.evidence_path,
        "evidence_hash": violation.evidence_hash
    }


# 7. Retrieve violation evidence image
@app.get("/api/violations/{violation_id}/evidence")
def get_evidence(
    violation_id: int,
    db: Session = Depends(get_db)
):
    violation = db.query(Violation).filter(
        Violation.id == violation_id
    ).first()
    if violation is None:
        raise HTTPException(
            status_code=404,
            detail="Violation not found"
        )
    if not violation.evidence_path:
        raise HTTPException(
            status_code=404,
            detail="No evidence associated with this violation"
        )

    # Resolve only the filename inside the evidence directory
    filename = Path(violation.evidence_path).name
    file_path = (EVIDENCE_DIR / filename).resolve()

    if file_path.parent != EVIDENCE_DIR.resolve():
        raise HTTPException(
            status_code=400,
            detail="Invalid evidence path"
        )
    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Evidence file not found"
        )

    return FileResponse(file_path)


# 8. Verify evidence integrity
@app.get("/api/violations/{violation_id}/evidence-integrity")
def verify_evidence_integrity(
    violation_id: int,
    db: Session = Depends(get_db),
):
    violation = db.query(Violation).filter(
        Violation.id == violation_id
    ).first()
    if not violation:
        raise HTTPException(
            status_code=404,
            detail="Violation not found",
        )
    if not violation.evidence_path or not violation.evidence_hash:
        raise HTTPException(
            status_code=400,
            detail="Evidence or its hash is missing",
        )
    filename = Path(violation.evidence_path).name
    file_path = EVIDENCE_DIR / filename
    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Evidence file not found",
        )
    current_hash = hashlib.sha256(
        file_path.read_bytes()
    ).hexdigest()
    intact = current_hash == violation.evidence_hash
    return {
        "violation_id": violation_id,
        "integrity": "intact" if intact else "mismatch",
        "intact": intact,
        "verified": intact,
        "stored_hash": violation.evidence_hash,
        "current_hash": current_hash,
    }


# 9. Upload & Process Traffic Video
@app.post("/api/video/process")
async def process_traffic_video(
    file: UploadFile = File(...),
    max_frames: Optional[int] = Form(default=300)
):
    """
    Ingest a traffic video, validate constraints, run the multi-model vision pipeline,
    and generate tamper-evident violation records.
    """
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{extension}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
        )

    # Read video content with size limit enforcement (max 100 MB)
    content = await file.read(MAX_VIDEO_SIZE + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Empty video file")
    if len(content) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=413, detail="Video exceeds 100 MB maximum size limit")

    # Compute SHA-256 fingerprint of the uploaded video stream
    video_hash = hashlib.sha256(content).hexdigest()
    output_filename = f"annotated_{video_hash[:12]}.mp4"
    output_file_path = PROCESSED_VIDEO_DIR / output_filename

    # Deduplication guard: prevent duplicate records if job is retried
    if video_hash in PROCESSED_VIDEO_HASHES and output_file_path.is_file():
        return {
            "status": "already_processed",
            "message": "This video stream has already been processed. Duplicate record creation was prevented.",
            "video_hash": video_hash,
            "annotated_video_url": f"/api/video/processed/{output_filename}",
            "filename": file.filename,
            "events_logged": 0
        }

    # Save uploaded video safely to designated staging directory
    input_filename = f"upload_{video_hash[:12]}{extension}"
    input_file_path = UPLOAD_DIR / input_filename
    input_file_path.write_bytes(content)

    # Invoke vision/detector.py with explicit input/output paths
    detector_script = PROJECT_ROOT / "vision" / "detector.py"
    if not detector_script.is_file():
        raise HTTPException(status_code=500, detail="Vision detector script not found on server")

    cmd = [
        sys.executable,
        str(detector_script),
        "--video", str(input_file_path),
        "--output", str(output_file_path),
        "--api-url", "http://127.0.0.1:8000",
        "--camera-id", f"CAM_FEED_{video_hash[:6].upper()}",
        "--custom-conf", "0.25"
    ]
    if max_frames and max_frames > 0:
        cmd.extend(["--max-frames", str(max_frames)])

    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Video processing timed out after 600 seconds")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to execute vision detector: {exc}")

    if proc.returncode != 0:
        err_msg = proc.stderr[-400:] if proc.stderr else proc.stdout[-400:]
        raise HTTPException(status_code=500, detail=f"Vision pipeline error (Code {proc.returncode}): {err_msg}")

    # Register hash to prevent duplicate reprocessing
    PROCESSED_VIDEO_HASHES.add(video_hash)

    # Parse stdout for incident summary
    stdout_text = proc.stdout
    stopline_count = 0
    helmet_count = 0
    triple_count = 0
    total_events = 0

    m_stop = re.search(r"Stop-Line Breaches:\s+(\d+)", stdout_text)
    if m_stop:
        stopline_count = int(m_stop.group(1))
    m_helm = re.search(r"No-Helmet Incidents:\s+(\d+)", stdout_text)
    if m_helm:
        helmet_count = int(m_helm.group(1))
    m_trip = re.search(r"Triple-Riding Cases:\s+(\d+)", stdout_text)
    if m_trip:
        triple_count = int(m_trip.group(1))
    m_tot = re.search(r"Total Logged Events:\s+(\d+)", stdout_text)
    if m_tot:
        total_events = int(m_tot.group(1))

    return {
        "status": "success",
        "message": "Video successfully analyzed and infractions registered.",
        "video_hash": video_hash,
        "input_filename": file.filename,
        "output_filename": output_filename,
        "annotated_video_url": f"/api/video/processed/{output_filename}",
        "summary": {
            "stop_line": stopline_count,
            "no_helmet": helmet_count,
            "triple_riding": triple_count,
            "total_logged": total_events
        }
    }


# 10. Stream / Retrieve Processed Video
@app.get("/api/video/processed/{filename}")
def get_processed_video(filename: str):
    """Serve the annotated MP4 video generated by the vision pipeline."""
    safe_name = Path(filename).name
    file_path = (PROCESSED_VIDEO_DIR / safe_name).resolve()
    if file_path.parent != PROCESSED_VIDEO_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid video path")
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Processed video file not found")
    return FileResponse(file_path, media_type="video/mp4")


# 11. Serve Frontend Dashboard
FRONTEND_FILE = PROJECT_ROOT / "frontend" / "index.html"


@app.get("/dashboard")
def serve_dashboard():
    """Serve the single-page RTO Traffic Intelligence dashboard."""
    if not FRONTEND_FILE.is_file():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found")
    return FileResponse(str(FRONTEND_FILE), media_type="text/html")



