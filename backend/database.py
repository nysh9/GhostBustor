"""
Database module for PostgreSQL + PostGIS integration
Handles all database operations for ghost net sightings, fishing grounds, and predictions
"""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime, Text, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from geoalchemy2 import Geometry
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

Base = declarative_base()

# Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/ghostgear"
)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ============== DATABASE MODELS ==============

class GhostNetSightingDB(Base):
    __tablename__ = "ghost_net_sightings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    location = Column(Geometry('POINT', srid=4326), nullable=False)
    sighting_date = Column(DateTime, nullable=False)
    net_type = Column(String(50))
    estimated_size = Column(String(20))
    reported_by = Column(String(200))
    verified = Column(Boolean, default=False)
    animals_affected = Column(Integer, default=0)
    photos = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String(100))  # e.g., "NOAA", "Global Fishing Watch", "user_report"


class FishingGroundDB(Base):
    __tablename__ = "fishing_grounds"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    center = Column(Geometry('POINT', srid=4326), nullable=False)
    radius_km = Column(Float, nullable=False)
    intensity = Column(Float, default=0.5)
    source = Column(String(100))  # "NOAA Fisheries", "Global Fishing Watch"
    meta_data = Column(Text)  # JSON string for additional data
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GyreZoneDB(Base):
    __tablename__ = "gyre_zones"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    center = Column(Geometry('POINT', srid=4326), nullable=False)
    radius_km = Column(Float, nullable=False)
    intensity = Column(Float, default=0.4)
    gyre_type = Column(String(50))  # "North Pacific", "South Pacific", etc.
    source = Column(String(100))
    meta_data = Column(Text)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PredictionZoneDB(Base):
    __tablename__ = "prediction_zones"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    zone_id = Column(String(50), unique=True, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    center = Column(Geometry('POINT', srid=4326), nullable=False)
    radius_km = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    predicted_net_count = Column(Integer, default=0)
    accumulation_reason = Column(Text)
    recommended_action = Column(Text)
    historical_accuracy = Column(Float)
    model_version = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)


class ModelTrainingHistoryDB(Base):
    __tablename__ = "model_training_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version = Column(String(20), nullable=False)
    training_date = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_samples = Column(Integer)
    validation_samples = Column(Integer)
    model_path = Column(String(500))
    notes = Column(Text)


# ============== DATABASE OPERATIONS ==============

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_sightings_near_location(
    db: Session,
    lat: float,
    lon: float,
    radius_km: float = 50.0,
    days_back: int = 365
) -> List[GhostNetSightingDB]:
    """Get sightings near a location using PostGIS spatial query"""
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
    
    # PostGIS ST_DWithin query (radius in meters)
    radius_meters = radius_km * 1000
    
    point = f"ST_SetSRID(ST_MakePoint({lon}, {lat}), 4326)"
    
    return db.query(GhostNetSightingDB).filter(
        func.ST_DWithin(
            GhostNetSightingDB.location,
            func.ST_SetSRID(func.ST_MakePoint(lon, lat), 4326),
            radius_meters
        ),
        GhostNetSightingDB.sighting_date >= cutoff_date
    ).all()


def get_fishing_grounds_in_region(
    db: Session,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float
) -> List[FishingGroundDB]:
    """Get fishing grounds in a bounding box"""
    return db.query(FishingGroundDB).filter(
        FishingGroundDB.lat >= min_lat,
        FishingGroundDB.lat <= max_lat,
        FishingGroundDB.lon >= min_lon,
        FishingGroundDB.lon <= max_lon
    ).all()


def get_gyre_zones_in_region(
    db: Session,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float
) -> List[GyreZoneDB]:
    """Get gyre zones in a bounding box"""
    return db.query(GyreZoneDB).filter(
        GyreZoneDB.lat >= min_lat,
        GyreZoneDB.lat <= max_lat,
        GyreZoneDB.lon >= min_lon,
        GyreZoneDB.lon <= max_lon
    ).all()


def save_prediction_zone(db: Session, zone_data: Dict[str, Any]) -> PredictionZoneDB:
    """Save a prediction zone to database"""
    zone = PredictionZoneDB(**zone_data)
    db.add(zone)
    db.commit()
    db.refresh(zone)
    return zone


def save_training_history(
    db: Session,
    model_version: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float,
    training_samples: int,
    validation_samples: int,
    model_path: str,
    notes: Optional[str] = None
) -> ModelTrainingHistoryDB:
    """Save model training history"""
    history = ModelTrainingHistoryDB(
        model_version=model_version,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        training_samples=training_samples,
        validation_samples=validation_samples,
        model_path=model_path,
        notes=notes
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    return history

