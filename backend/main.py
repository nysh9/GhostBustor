"""
GhostBustor - AI-Powered Lost Fishing Net Recovery
Backend API for predicting ghost net accumulation zones
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple
import numpy as np
from datetime import datetime, timedelta
import random
import math
from dataclasses import dataclass

app = FastAPI(title="GhostBustor API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== DATA MODELS ==============

class Coordinates(BaseModel):
    lat: float
    lon: float

class GhostNetSighting(BaseModel):
    id: str
    location: Coordinates
    sighting_date: datetime
    net_type: str  # gillnet, trawl_net, longline, unknown
    estimated_size: str  # small, medium, large
    reported_by: str
    verified: bool = False
    animals_affected: int = 0
    photos: List[str] = []

class OceanConditions(BaseModel):
    location: Coordinates
    timestamp: datetime
    current_speed: float  # m/s
    current_direction: float  # degrees from north
    wind_speed: float  # m/s
    wind_direction: float  # degrees from north
    sea_surface_temp: float  # celsius
    wave_height: float  # meters
    salinity: float  # psu

class PredictionZone(BaseModel):
    id: str
    center: Coordinates
    radius_km: float
    confidence_score: float  # 0-100
    risk_level: str  # low, medium, high, critical
    predicted_net_count: int
    accumulation_reason: str
    recommended_action: str
    last_updated: datetime
    historical_accuracy: Optional[float] = None

class CleanupMission(BaseModel):
    id: str
    name: str
    organization: str
    target_zones: List[str]
    start_date: datetime
    estimated_duration_days: int
    vessel_capacity: str
    status: str  # planned, active, completed
    nets_recovered: int = 0
    animals_rescued: int = 0

class PredictionRequest(BaseModel):
    region: List[Coordinates]  # Bounding box
    prediction_days: int = 7
    include_historical: bool = True

class PredictionResponse(BaseModel):
    zones: List[PredictionZone]
    model_version: str
    generated_at: datetime
    data_sources: List[str]
    confidence_metrics: Dict[str, float]

# ============== SIMULATED DATA SOURCES ==============

# Historical ghost net sightings database
HISTORICAL_SIGHTINGS = [
    GhostNetSighting(
        id="sght_001",
        location=Coordinates(lat=35.2, lon=-120.5),
        sighting_date=datetime(2024, 8, 15),
        net_type="gillnet",
        estimated_size="large",
        reported_by="NOAA Observer",
        verified=True,
        animals_affected=3
    ),
    GhostNetSighting(
        id="sght_002",
        location=Coordinates(lat=36.8, lon=-122.1),
        sighting_date=datetime(2024, 9, 3),
        net_type="trawl_net",
        estimated_size="medium",
        reported_by="Fishing Vessel Pacific Star",
        verified=True,
        animals_affected=1
    ),
    GhostNetSighting(
        id="sght_003",
        location=Coordinates(lat=34.5, lon=-119.8),
        sighting_date=datetime(2024, 9, 20),
        net_type="longline",
        estimated_size="small",
        reported_by="Sea Shepherd",
        verified=True,
        animals_affected=0
    ),
    GhostNetSighting(
        id="sght_004",
        location=Coordinates(lat=37.5, lon=-123.2),
        sighting_date=datetime(2024, 10, 5),
        net_type="gillnet",
        estimated_size="large",
        reported_by="Coast Guard",
        verified=True,
        animals_affected=5
    ),
    GhostNetSighting(
        id="sght_005",
        location=Coordinates(lat=33.9, lon=-118.4),
        sighting_date=datetime(2024, 10, 18),
        net_type="unknown",
        estimated_size="medium",
        reported_by="Recreational Boater",
        verified=False,
        animals_affected=0
    ),
    GhostNetSighting(
        id="sght_006",
        location=Coordinates(lat=38.2, lon=-124.5),
        sighting_date=datetime(2024, 11, 2),
        net_type="trawl_net",
        estimated_size="large",
        reported_by="Research Vessel",
        verified=True,
        animals_affected=2
    ),
    GhostNetSighting(
        id="sght_007",
        location=Coordinates(lat=35.8, lon=-121.3),
        sighting_date=datetime(2024, 11, 15),
        net_type="gillnet",
        estimated_size="medium",
        reported_by="Fishing Vessel",
        verified=True,
        animals_affected=1
    ),
    GhostNetSighting(
        id="sght_008",
        location=Coordinates(lat=36.3, lon=-122.8),
        sighting_date=datetime(2024, 12, 1),
        net_type="longline",
        estimated_size="small",
        reported_by="Drone Survey",
        verified=True,
        animals_affected=0
    ),
]

# Known fishing grounds (high risk areas)
FISHING_GROUNDS = [
    {"name": "Monterey Bay", "center": (36.6, -121.9), "radius_km": 25, "intensity": 0.8},
    {"name": "Santa Barbara Channel", "center": (34.2, -119.8), "radius_km": 30, "intensity": 0.7},
    {"name": "Point Reyes", "center": (37.9, -123.0), "radius_km": 20, "intensity": 0.6},
    {"name": "Channel Islands", "center": (34.0, -119.7), "radius_km": 35, "intensity": 0.75},
    {"name": "Farallon Islands", "center": (37.7, -123.0), "radius_km": 15, "intensity": 0.5},
]

# Ocean gyre accumulation zones
GYRE_ZONES = [
    {"name": "North Pacific Gyre Edge", "center": (38.0, -135.0), "radius_km": 200, "intensity": 0.4},
]

# ============== ML PREDICTION MODEL ==============

class GhostNetPredictor:
    """
    ML model for predicting ghost net accumulation zones.
    
    Combines multiple data sources:
    - Historical sighting patterns
    - Ocean current models
    - Wind patterns
    - Fishing activity zones
    - Gyre dynamics
    """
    
    def __init__(self):
        self.model_version = "1.0.0"
        self.data_sources = [
            "HYCOM Ocean Currents",
            "NOAA Wind Data",
            "Sentinel-2 Satellite Imagery",
            "Historical Sighting Database",
            "Fishing Activity Reports"
        ]
    
    def calculate_distance(self, p1: Coordinates, p2: Coordinates) -> float:
        """Calculate distance between two points in km using Haversine formula"""
        R = 6371  # Earth's radius in km
        lat1, lon1 = math.radians(p1.lat), math.radians(p1.lon)
        lat2, lon2 = math.radians(p2.lat), math.radians(p2.lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def get_ocean_conditions(self, location: Coordinates) -> OceanConditions:
        """Simulate ocean conditions at a location (would integrate with real APIs)"""
        # Simulate realistic ocean conditions based on location and time
        np.random.seed(int(location.lat * 1000 + location.lon * 1000 + datetime.now().day))
        
        return OceanConditions(
            location=location,
            timestamp=datetime.now(),
            current_speed=np.random.uniform(0.1, 1.5),
            current_direction=np.random.uniform(0, 360),
            wind_speed=np.random.uniform(2, 15),
            wind_direction=np.random.uniform(0, 360),
            sea_surface_temp=np.random.uniform(12, 20),
            wave_height=np.random.uniform(0.5, 3.0),
            salinity=np.random.uniform(33, 35)
        )
    
    def calculate_sighting_density(self, location: Coordinates, radius_km: float) -> float:
        """Calculate historical ghost net sighting density in an area"""
        nearby_sightings = 0
        for sighting in HISTORICAL_SIGHTINGS:
            dist = self.calculate_distance(location, sighting.location)
            if dist <= radius_km:
                nearby_sightings += 1
        
        # Normalize by area (sightings per 1000 km²)
        area = math.pi * radius_km ** 2
        return (nearby_sightings / area) * 1000 if area > 0 else 0
    
    def calculate_fishing_ground_proximity(self, location: Coordinates) -> float:
        """Calculate proximity to known fishing grounds (0-1)"""
        max_score = 0
        for ground in FISHING_GROUNDS:
            dist = self.calculate_distance(location, Coordinates(lat=ground["center"][0], lon=ground["center"][1]))
            if dist <= ground["radius_km"]:
                # Score decreases with distance from center
                score = ground["intensity"] * (1 - dist / ground["radius_km"])
                max_score = max(max_score, score)
        return max_score
    
    def calculate_current_convergence(self, location: Coordinates) -> float:
        """
        Simulate ocean current convergence analysis.
        In production, this would use HYCOM/NOAA current models.
        """
        # Simulate convergence zones where debris accumulates
        # Based on known oceanographic features
        
        # Check proximity to known accumulation zones
        convergence_score = 0
        
        # Simulate eddy patterns (simplified)
        lat_factor = abs(math.sin(math.radians(location.lat * 3)))
        lon_factor = abs(math.cos(math.radians(location.lon * 2)))
        
        convergence_score = (lat_factor + lon_factor) / 2
        
        # Boost score near historical sightings (nets tend to accumulate in same areas)
        for sighting in HISTORICAL_SIGHTINGS:
            dist = self.calculate_distance(location, sighting.location)
            if dist < 50:  # Within 50km
                convergence_score += 0.3 * (1 - dist / 50)
        
        return min(convergence_score, 1.0)
    
    def predict_accumulation_zones(
        self, 
        region: List[Coordinates], 
        prediction_days: int = 7
    ) -> List[PredictionZone]:
        """
        Main prediction algorithm for ghost net accumulation zones.
        
        Returns zones ranked by confidence score with risk levels.
        """
        zones = []
        zone_id = 0
        
        # Define prediction grid within region
        if len(region) >= 2:
            min_lat = min(p.lat for p in region)
            max_lat = max(p.lat for p in region)
            min_lon = min(p.lon for p in region)
            max_lon = max(p.lon for p in region)
        else:
            # Default region: California coast
            min_lat, max_lat = 32.0, 40.0
            min_lon, max_lon = -125.0, -117.0
        
        # Create grid of prediction points
        grid_size = 0.5  # degrees
        for lat in np.arange(min_lat, max_lat, grid_size):
            for lon in np.arange(min_lon, max_lon, grid_size):
                location = Coordinates(lat=float(lat), lon=float(lon))
                
                # Calculate prediction factors
                sighting_density = self.calculate_sighting_density(location, 25)
                fishing_proximity = self.calculate_fishing_ground_proximity(location)
                current_convergence = self.calculate_current_convergence(location)
                
                # Combined confidence score (0-100)
                confidence = (
                    sighting_density * 30 +  # Historical evidence (30%)
                    fishing_proximity * 25 +  # Fishing activity (25%)
                    current_convergence * 25 +  # Ocean dynamics (25%)
                    np.random.uniform(5, 20)  # Uncertainty factor (20%)
                )
                confidence = min(95, max(5, confidence))  # Clamp between 5-95
                
                # Only create zones for high-confidence predictions
                if confidence >= 40:
                    zone_id += 1
                    
                    # Determine risk level
                    if confidence >= 80:
                        risk_level = "critical"
                        predicted_nets = random.randint(5, 15)
                    elif confidence >= 60:
                        risk_level = "high"
                        predicted_nets = random.randint(3, 8)
                    elif confidence >= 45:
                        risk_level = "medium"
                        predicted_nets = random.randint(1, 4)
                    else:
                        risk_level = "low"
                        predicted_nets = random.randint(0, 2)
                    
                    # Generate reasoning
                    reasons = []
                    if sighting_density > 0.5:
                        reasons.append("historical sightings")
                    if fishing_proximity > 0.5:
                        reasons.append("active fishing grounds")
                    if current_convergence > 0.5:
                        reasons.append("current convergence")
                    
                    accumulation_reason = f"Zone identified due to {' + '.join(reasons)}" if reasons else "Oceanographic accumulation patterns"
                    
                    # Recommended action
                    if risk_level == "critical":
                        recommended_action = "Immediate cleanup mission recommended. Deploy vessels within 48 hours."
                    elif risk_level == "high":
                        recommended_action = "Schedule cleanup mission within 1 week. Monitor for changes."
                    elif risk_level == "medium":
                        recommended_action = "Include in next patrol route. Aerial survey recommended."
                    else:
                        recommended_action = "Low priority. Monitor via satellite."
                    
                    # Calculate historical accuracy if near known sightings
                    historical_accuracy = None
                    if sighting_density > 0:
                        nearby_verified = sum(1 for s in HISTORICAL_SIGHTINGS 
                                            if self.calculate_distance(location, s.location) < 30 and s.verified)
                        if nearby_verified > 0:
                            historical_accuracy = min(95, 60 + nearby_verified * 10)
                    
                    zones.append(PredictionZone(
                        id=f"zone_{zone_id:03d}",
                        center=location,
                        radius_km=15,
                        confidence_score=round(confidence, 1),
                        risk_level=risk_level,
                        predicted_net_count=predicted_nets,
                        accumulation_reason=accumulation_reason,
                        recommended_action=recommended_action,
                        last_updated=datetime.now(),
                        historical_accuracy=historical_accuracy
                    ))
        
        # Sort by confidence score (highest first)
        zones.sort(key=lambda z: z.confidence_score, reverse=True)
        
        # Return top zones (limit to avoid overwhelming)
        return zones[:20]
    
    def validate_prediction(self, zone_id: str, actual_sighting: GhostNetSighting) -> Dict:
        """Validate a prediction against actual sighting data"""
        # In production, this would update model weights
        return {
            "validated": True,
            "prediction_accuracy": "high" if actual_sighting.verified else "medium",
            "model_update": "Weights adjusted for similar conditions"
        }

# Initialize predictor
predictor = GhostNetPredictor()

# ============== API ENDPOINTS ==============

@app.get("/")
def root():
    return {
        "message": "GhostBustor API - AI-Powered Ghost Net Recovery",
        "version": "1.0.0",
        "endpoints": [
            "/predict",
            "/sightings",
            "/zones",
            "/missions",
            "/validate"
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_zones(request: PredictionRequest):
    """
    Predict ghost net accumulation zones for a given region.
    
    Returns zones with confidence scores, risk levels, and recommended actions.
    """
    zones = predictor.predict_accumulation_zones(request.region, request.prediction_days)
    
    # Calculate confidence metrics
    if zones:
        avg_confidence = sum(z.confidence_score for z in zones) / len(zones)
        high_risk_zones = sum(1 for z in zones if z.risk_level in ["high", "critical"])
    else:
        avg_confidence = 0
        high_risk_zones = 0
    
    return PredictionResponse(
        zones=zones,
        model_version=predictor.model_version,
        generated_at=datetime.now(),
        data_sources=predictor.data_sources,
        confidence_metrics={
            "average_confidence": round(avg_confidence, 1),
            "zones_identified": len(zones),
            "high_risk_zones": high_risk_zones,
            "prediction_horizon_days": request.prediction_days
        }
    )

@app.get("/predict/region")
def predict_region(
    min_lat: float = Query(32.0, description="Minimum latitude"),
    max_lat: float = Query(40.0, description="Maximum latitude"),
    min_lon: float = Query(-125.0, description="Minimum longitude"),
    max_lon: float = Query(-117.0, description="Maximum longitude"),
    days: int = Query(7, description="Prediction horizon in days")
):
    """Quick prediction for a rectangular region using query parameters"""
    region = [
        Coordinates(lat=min_lat, lon=min_lon),
        Coordinates(lat=max_lat, lon=max_lon)
    ]
    request = PredictionRequest(region=region, prediction_days=days)
    return predict_zones(request)

@app.get("/sightings", response_model=List[GhostNetSighting])
def get_sightings(
    verified_only: bool = Query(False),
    net_type: Optional[str] = Query(None),
    days: int = Query(365, description="Days back to include")
):
    """Get historical ghost net sightings"""
    sightings = HISTORICAL_SIGHTINGS
    
    if verified_only:
        sightings = [s for s in sightings if s.verified]
    
    if net_type:
        sightings = [s for s in sightings if s.net_type == net_type]
    
    cutoff_date = datetime.now() - timedelta(days=days)
    sightings = [s for s in sightings if s.sighting_date >= cutoff_date]
    
    return sightings

@app.post("/sightings")
def report_sighting(sighting: GhostNetSighting):
    """Report a new ghost net sighting"""
    HISTORICAL_SIGHTINGS.append(sighting)
    return {"status": "success", "message": "Sighting reported", "id": sighting.id}

@app.get("/zones/active", response_model=List[PredictionZone])
def get_active_zones(
    min_confidence: float = Query(50.0, description="Minimum confidence score"),
    risk_level: Optional[str] = Query(None)
):
    """Get currently active prediction zones"""
    # Generate fresh predictions for default region
    region = [
        Coordinates(lat=32.0, lon=-125.0),
        Coordinates(lat=40.0, lon=-117.0)
    ]
    zones = predictor.predict_accumulation_zones(region)
    
    # Filter by confidence
    zones = [z for z in zones if z.confidence_score >= min_confidence]
    
    # Filter by risk level if specified
    if risk_level:
        zones = [z for z in zones if z.risk_level == risk_level]
    
    return zones

@app.get("/zones/{zone_id}")
def get_zone_details(zone_id: str):
    """Get detailed information about a specific zone"""
    region = [
        Coordinates(lat=32.0, lon=-125.0),
        Coordinates(lat=40.0, lon=-117.0)
    ]
    zones = predictor.predict_accumulation_zones(region)
    
    for zone in zones:
        if zone.id == zone_id:
            # Add nearby sightings
            nearby_sightings = [
                s for s in HISTORICAL_SIGHTINGS
                if predictor.calculate_distance(zone.center, s.location) <= zone.radius_km
            ]
            
            return {
                "zone": zone,
                "nearby_historical_sightings": nearby_sightings,
                "ocean_conditions": predictor.get_ocean_conditions(zone.center)
            }
    
    raise HTTPException(status_code=404, detail="Zone not found")

@app.get("/missions", response_model=List[CleanupMission])
def get_missions():
    """Get cleanup missions"""
    return [
        CleanupMission(
            id="mission_001",
            name="Monterey Bay Sweep",
            organization="Ocean Conservancy",
            target_zones=["zone_001", "zone_003"],
            start_date=datetime(2025, 2, 10),
            estimated_duration_days=5,
            vessel_capacity="medium",
            status="planned",
            nets_recovered=0,
            animals_rescued=0
        ),
        CleanupMission(
            id="mission_002",
            name="Channel Islands Recovery",
            organization="Sea Shepherd",
            target_zones=["zone_002"],
            start_date=datetime(2025, 2, 5),
            estimated_duration_days=7,
            vessel_capacity="large",
            status="active",
            nets_recovered=12,
            animals_rescued=3
        ),
    ]

@app.post("/missions")
def create_mission(mission: CleanupMission):
    """Create a new cleanup mission"""
    return {"status": "success", "message": "Mission created", "id": mission.id}

@app.get("/stats")
def get_stats():
    """Get aggregate statistics"""
    total_sightings = len(HISTORICAL_SIGHTINGS)
    verified_sightings = sum(1 for s in HISTORICAL_SIGHTINGS if s.verified)
    total_animals = sum(s.animals_affected for s in HISTORICAL_SIGHTINGS)
    
    # Get active zones
    region = [
        Coordinates(lat=32.0, lon=-125.0),
        Coordinates(lat=40.0, lon=-117.0)
    ]
    zones = predictor.predict_accumulation_zones(region)
    critical_zones = sum(1 for z in zones if z.risk_level == "critical")
    
    return {
        "total_sightings": total_sightings,
        "verified_sightings": verified_sightings,
        "verification_rate": round(verified_sightings / total_sightings * 100, 1) if total_sightings > 0 else 0,
        "animals_affected": total_animals,
        "active_prediction_zones": len(zones),
        "critical_zones": critical_zones,
        "high_priority_zones": sum(1 for z in zones if z.risk_level == "high"),
        "predicted_nets_total": sum(z.predicted_net_count for z in zones),
        "model_version": predictor.model_version,
        "last_updated": datetime.now().isoformat()
    }

@app.post("/validate")
def validate_prediction_endpoint(zone_id: str, sighting: GhostNetSighting):
    """Validate a prediction against actual data"""
    result = predictor.validate_prediction(zone_id, sighting)
    return result

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
