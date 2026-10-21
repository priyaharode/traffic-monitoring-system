
import os
import uuid
import io
import hashlib
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File
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


