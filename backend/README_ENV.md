# Environment Configuration

## Setting Up Your .env File

1. **Copy the example file:**
   ```bash
   cp env.example .env
   ```

2. **Edit the .env file with your settings:**
   ```bash
   # Required: Database connection
   DATABASE_URL=postgresql://username:password@localhost:5432/ghostgear
   
   # Optional: Model path (defaults to models/ghostnet_model_latest.pkl)
   MODEL_PATH=models/ghostnet_model_latest.pkl
   
   # Optional: API Keys (if you have them)
   # GFW_API_KEY=your_global_fishing_watch_api_key
   # NOAA_API_KEY=your_noaa_api_key
   ```

## Database URL Format

The `DATABASE_URL` follows this format:
```
postgresql://[user]:[password]@[host]:[port]/[database]
```

Examples:
- Local PostgreSQL: `postgresql://postgres:postgres@localhost:5432/ghostgear`
- Docker: `postgresql://postgres:postgres@db:5432/ghostgear`
- Remote: `postgresql://user:pass@example.com:5432/ghostgear`

## Security Note

⚠️ **Never commit your `.env` file to version control!**

The `.env` file is already in `.gitignore` to prevent accidental commits.


