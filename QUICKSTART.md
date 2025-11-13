# Quick Start Guide

## Prerequisites

1. **MongoDB**: Make sure MongoDB is installed and running on `localhost:27017`
   - Download from: https://www.mongodb.com/try/download/community
   - Or use Docker: `docker run -d -p 27017:27017 mongo`

2. **Python 3.8+**: Required for backend

3. **Node.js 16+**: Required for frontend

## Setup Steps

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (copy from .env.example)
# Set MONGODB_URL, DATABASE_NAME, and SECRET_KEY

# Start backend server
uvicorn app.main:app --reload --port 8000
```

Backend will be available at: `http://localhost:8000`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at: `http://localhost:5173`

## First Run

1. **Start MongoDB** (if not already running)

2. **Start Backend**: Run the backend server (see above)

3. **Start Frontend**: Run the frontend server (see above)

4. **Create Account**: 
   - Go to `http://localhost:5173`
   - Click "Sign Up"
   - Create an account

5. **Run Your First Benchmark**:
   - Go to "Benchmark" page
   - Select "Prebuilt Configuration"
   - Configure your pipeline (chunk size, embedding model, etc.)
   - Click "Next"
   - Select benchmark types (Single, Multilingual, Multi-Hop)
   - Click "Start Benchmarking"
   - Wait for results (first run will download models, so it may take time)

## Notes

- **First Run**: Models will be downloaded from Hugging Face on first use. This may take several minutes depending on your internet connection.
- **Model Caching**: Models are cached in `models_cache/` directory for faster subsequent runs.
- **Database**: MongoDB collections are created automatically on first use.
- **Vector DB**: ChromaDB data is stored in `chroma_db/` directory.

## Troubleshooting

### MongoDB Connection Error
- Make sure MongoDB is running: `mongod` or check Docker container
- Verify connection string in `.env` file

### Model Download Issues
- Check internet connection
- Models are large (several GB), ensure sufficient disk space
- First download may take 10-30 minutes

### Port Already in Use
- Backend: Change port in `uvicorn` command: `--port 8001`
- Frontend: Change port in `vite.config.js`

### Import Errors
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

## Windows Users

You can use the provided batch files:
- `start_backend.bat` - Starts backend server
- `start_frontend.bat` - Starts frontend server

Make sure to activate virtual environment first in backend directory.

