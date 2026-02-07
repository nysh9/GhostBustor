# Setup Guide

This guide will help you set up GhostBustor from scratch.

## Quick Start

### Option 1: Using Docker (Recommended)

1. **Start PostgreSQL with PostGIS**
   ```bash
   docker-compose up -d db
   ```

2. **Set up environment**
   ```bash
   cd backend
   cp env.example .env
   # Edit .env with your database credentials
   ```

3. **Install dependencies and initialize**
   ```bash
   pip install -r requirements.txt
   python init_database.py
   ```

4. **Start the application**
   ```bash
   python main.py
   ```

### Option 2: Manual Setup

#### 1. Install PostgreSQL and PostGIS

**macOS (using Homebrew):**
```bash
brew install postgresql postgis
brew services start postgresql
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib postgis
sudo systemctl start postgresql
```

**Windows:**
Download and install from [PostgreSQL website](https://www.postgresql.org/download/windows/)

#### 2. Create Database

```bash
# Create database
createdb ghostgear

# Connect and enable PostGIS
psql ghostgear
```

In psql:
```sql
CREATE EXTENSION postgis;
\q
```

#### 3. Set Up Python Environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 4. Configure Environment

```bash
cp env.example .env
# Edit .env with your database URL
# DATABASE_URL=postgresql://username:password@localhost:5432/ghostgear
```

#### 5. Initialize Database

```bash
python init_database.py
```

This will:
- Create all necessary tables
- Seed fishing grounds from NOAA data
- Seed gyre zones from oceanographic research

#### 6. Start Backend

```bash
python main.py
```

The API will be available at `http://localhost:8000`

#### 7. Start Frontend

In a new terminal:
```bash
cd frontend
python -m http.server 8080
```

Or use the startup script:
```bash
./start.sh
```

## Verification

1. **Check API health**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check database**
   ```bash
   psql ghostgear -c "SELECT COUNT(*) FROM fishing_grounds;"
   ```

3. **Test prediction**
   ```bash
   curl "http://localhost:8000/predict/region?min_lat=32&max_lat=40&min_lon=-125&max_lon=-117"
   ```

## Troubleshooting

### Database Connection Issues

- Verify PostgreSQL is running: `pg_isready`
- Check connection string in `.env`
- Ensure PostGIS extension is installed: `psql ghostgear -c "\dx"`

### Model Training Issues

- Ensure you have historical sightings data
- Check that data fetchers can access NOAA APIs
- Verify model directory exists: `mkdir -p models`

### API Errors

- Check logs for detailed error messages
- Verify all dependencies are installed: `pip list`
- Ensure port 8000 is not in use

## Next Steps

- Add historical sighting data to improve predictions
- Train the model: `POST /train`
- Configure additional data sources (Global Fishing Watch API key, etc.)

