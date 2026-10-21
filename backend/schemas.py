# schemas.py

from datetime import datetime
from typing import Optional, Literal

from pydantic import BaseModel, Field, ConfigDict


ViolationType = Literal[
    "no_helmet",
    "triple_riding",
    "red_light",
    "stop_line"
]


class ViolationCreate(BaseModel):
    vehicle_number: Optional[str] = Field(
        default=None, max_length=20
    )
    violation_type: ViolationType
    camera_id: Optional[str] = Field(
        default=None, max_length=50
    )
    confidence: Optional[float] = Field(
        default=None, ge=0, le=1
    )
    evidence_path: Optional[str] = Field(
        default=None, max_length=255
    )


class ViolationResponse(BaseModel):
    id: int
    vehicle_number: Optional[str]
    violation_type: str
    timestamp: datetime
    camera_id: Optional[str]
    confidence: Optional[float]
    evidence_path: Optional[str]
    evidence_hash: Optional[str] = None
    status: str

    model_config = ConfigDict(from_attributes=True)


class ViolationStatusUpdate(BaseModel):
    status: Literal["pending", "approved", "rejected"]
