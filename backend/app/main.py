from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import auth, benchmark, leaderboard
from app.database import connect_db, close_db

app = FastAPI(title="RAG Benchmark API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(benchmark.router, prefix="/api/benchmark", tags=["benchmark"])
app.include_router(leaderboard.router, prefix="/api/leaderboard", tags=["leaderboard"])

@app.on_event("startup")
async def startup_event():
    await connect_db()

@app.on_event("shutdown")
async def shutdown_event():
    await close_db()

@app.get("/")
async def root():
    return {"message": "RAG Benchmark API"}




