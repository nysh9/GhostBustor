# GhostBustor

**AI-Powered Ghost Net Recovery System**

GhostBustor is an advanced machine learning platform that predicts ghost fishing net accumulation zones in ocean waters. By combining real-time oceanographic data, historical sighting patterns, and fishing activity data, the system helps conservation organizations and cleanup crews efficiently locate and recover lost fishing gear.

## Features

- **Real-Time Ocean Conditions**: Integrated with Open-Meteo Marine Weather API for live oceanographic data
- **Machine Learning Predictions**: Trained ML model using scikit-learn for accurate zone identification
- **Real Data Sources**: 
  - NOAA Fisheries fishing grounds data
  - Global Fishing Watch activity zones
  - Oceanographic research data for gyre zones
- **Spatial Database**: PostgreSQL + PostGIS for efficient geospatial queries
- **Model Persistence**: Save and load trained models for consistent predictions
- **RESTful API**: FastAPI-based backend with comprehensive endpoints
- **Interactive Dashboard**: Real-time map visualization of prediction zones

## Architecture

### Backend Components

1. **Data Fetchers** (`data_fetchers.py`)
   - Open-Meteo Marine Weather API integration
   - NOAA Fisheries data retrieval
   - Global Fishing Watch data processing
   - Oceanographic gyre zone data

2. **Machine Learning Model** (`ml_model.py`)
   - Gradient Boosting Regressor for confidence scoring
   - Feature engineering for oceanographic patterns
   - Model training and validation pipeline
   - Model persistence (save/load)

3. **Database Layer** (`database.py`)
   - PostgreSQL + PostGIS for spatial data
   - Models for sightings, fishing grounds, gyre zones
   - Spatial queries for efficient data retrieval

4. **API Server** (`main.py`)
   - FastAPI REST endpoints
   - Prediction zone generation
   - Real-time ocean conditions
   - Model training endpoints

### Frontend

- Interactive Leaflet map
- Real-time zone visualization
- Historical sighting markers
- Risk level filtering
- Zone detail panels

## Installation

### Prerequisites

- Python 3.9+
- PostgreSQL 12+ with PostGIS extension
- Docker and Docker Compose (optional, for easier database setup)

### Quick Setup

For detailed setup instructions, see [SETUP.md](SETUP.md).

**Using Docker (Recommended):**
```bash
# Start PostgreSQL with PostGIS
docker-compose up -d db

# Set up backend
cd backend
cp .env.example .env
pip install -r requirements.txt
python init_database.py

# Start application
python main.py
```

**Manual Setup:**
1. Install PostgreSQL and PostGIS
2. Create database: `createdb ghostgear && psql ghostgear -c "CREATE EXTENSION postgis;"`
3. Configure environment: `cd backend && cp env.example .env && edit .env`
4. Install dependencies: `pip install -r requirements.txt`
5. Initialize database: `python init_database.py`
6. Start backend: `python main.py`
7. Start frontend: `cd frontend && python -m http.server 8080`

Or use the startup script: `./start.sh`

## Usage

### API Endpoints

#### Predict Zones
```bash
POST /predict
{
  "region": [
    {"lat": 32.0, "lon": -125.0},
    {"lat": 40.0, "lon": -117.0}
  ],
  "prediction_days": 7
}
```

#### Get Ocean Conditions
```bash
GET /ocean-conditions?lat=36.0&lon=-122.0
```

#### Train Model
```bash
POST /train
```

#### Get Statistics
```bash
GET /stats
```

### Training the Model

The model can be trained using historical sighting data:

1. **Automatic Training**: Model trains automatically on startup if no saved model exists
2. **Manual Training**: Call `POST /train` to retrain with current data
3. **Model Persistence**: Trained models are saved to `models/` directory

### Data Sources

- **Open-Meteo Marine Weather API**: Real-time ocean conditions (wind, waves, sea surface temperature)
- **NOAA Fisheries**: Official fishing grounds data
- **Global Fishing Watch**: Vessel tracking and activity data
- **Oceanographic Research**: Gyre zone locations and characteristics

## Model Details

### Features

The ML model uses the following features:
- Historical sighting density
- Fishing ground proximity
- Ocean current convergence
- Gyre zone proximity
- Wind speed and direction
- Current speed
- Sea surface temperature
- Wave height
- Distance to shore
- Bathymetry depth

### Model Architecture

- **Algorithm**: Gradient Boosting Regressor
- **Output**: Confidence score (0-100) for accumulation zones
- **Training**: Supervised learning on historical sightings
- **Validation**: Train/test split with cross-validation

## Development

### Project Structure

```
ghostgear/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── database.py          # Database models and operations
│   ├── data_fetchers.py     # Real data source integrations
│   ├── ml_model.py          # ML model implementation
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # Main HTML
│   └── app.js               # Frontend JavaScript
├── start.sh                 # Startup script
└── README.md                # This file
```

### Adding New Data Sources

1. Add fetcher function in `data_fetchers.py`
2. Integrate into `GhostNetPredictor` class
3. Update feature engineering in `ml_model.py`

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Specify your license here]

## Acknowledgments

- NOAA National Data Buoy Center for ocean condition data
- NOAA Fisheries for fishing grounds data
- Global Fishing Watch for vessel tracking data
- Oceanographic research community for gyre zone data

## Contact

[Your contact information]

