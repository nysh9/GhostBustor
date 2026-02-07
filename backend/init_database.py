"""
Database initialization script
Creates tables and seeds initial data from real sources
"""

import os
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from database import (
    init_db, SessionLocal,
    FishingGroundDB, GyreZoneDB
)
from data_fetchers import get_noaa_fishing_grounds, get_gyre_zones
from geoalchemy2 import WKTElement

# Load environment variables
load_dotenv()


def seed_fishing_grounds(db: Session):
    """Seed fishing grounds from NOAA data"""
    print("Seeding fishing grounds...")
    
    fishing_grounds = get_noaa_fishing_grounds()
    
    for ground in fishing_grounds:
        # Check if already exists
        existing = db.query(FishingGroundDB).filter_by(
            name=ground["name"]
        ).first()
        
        if existing:
            continue
        
        lat, lon = ground["center"]
        point = WKTElement(f'POINT({lon} {lat})', srid=4326)
        
        fishing_ground = FishingGroundDB(
            name=ground["name"],
            lat=lat,
            lon=lon,
            center=point,
            radius_km=ground["radius_km"],
            intensity=ground["intensity"],
            source=ground.get("source", "NOAA Fisheries"),
            metadata=str(ground.get("metadata", {}))
        )
        
        db.add(fishing_ground)
    
    db.commit()
    print(f"Seeded {len(fishing_grounds)} fishing grounds")


def seed_gyre_zones(db: Session):
    """Seed gyre zones from oceanographic research data"""
    print("Seeding gyre zones...")
    
    gyre_zones = get_gyre_zones()
    
    for zone in gyre_zones:
        # Check if already exists
        existing = db.query(GyreZoneDB).filter_by(
            name=zone["name"]
        ).first()
        
        if existing:
            continue
        
        lat, lon = zone["center"]
        point = WKTElement(f'POINT({lon} {lat})', srid=4326)
        
        gyre_zone = GyreZoneDB(
            name=zone["name"],
            lat=lat,
            lon=lon,
            center=point,
            radius_km=zone["radius_km"],
            intensity=zone["intensity"],
            gyre_type=zone.get("gyre_type", "Unknown"),
            source=zone.get("source", "Oceanographic Research"),
            metadata=str(zone.get("metadata", {}))
        )
        
        db.add(gyre_zone)
    
    db.commit()
    print(f"Seeded {len(gyre_zones)} gyre zones")


def main():
    """Initialize database and seed data"""
    print("Initializing database...")
    init_db()
    print("Database initialized")
    
    db = SessionLocal()
    try:
        seed_fishing_grounds(db)
        seed_gyre_zones(db)
        print("\nDatabase setup complete!")
    except Exception as e:
        print(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    main()

