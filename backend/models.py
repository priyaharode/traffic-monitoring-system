
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func

from database import Base


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, index=True)

    vehicle_number = Column(String(20), nullable=True, index=True)
    violation_type = Column(String(50), nullable=False, index=True)

    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    camera_id = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)

    evidence_path = Column(String(255), nullable=True)
    evidence_hash = Column(String(64), nullable=True)

    status = Column(
        String(20),
        nullable=False,
        default="pending"
    )
