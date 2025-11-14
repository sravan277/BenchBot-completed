from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import json
import os
import uuid
import asyncio
from app.database import get_db
from app.routers.auth import get_current_user
from app.services.rag_pipeline import RAGPipeline
from app.services.evaluator import BenchmarkEvaluator

router = APIRouter()

# Get absolute path to corpus directory (go up from backend/app/routers to project root)
CORPUS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "corpse_data"

# In-memory progress store (in production, use Redis or similar)
progress_store: Dict[str, Dict[str, Any]] = {}

class PrebuiltConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_db: str = "chroma"
    reranking_strategy: str = "none"
    top_k: int = 5

class BenchmarkRequest(BaseModel):
    config: Optional[PrebuiltConfig] = None
    benchmark_types: List[str]  # ["single", "multilingual", "multi_hop"]
    pipeline_json: Optional[Dict[str, Any]] = None
    benchmark_name: Optional[str] = None  # Optional name for the benchmark

def update_progress(job_id: str, step: str, progress: float, details: str = ""):
    """Update progress for a benchmark job"""
    if job_id not in progress_store:
        progress_store[job_id] = {
            "status": "running",
            "current_step": "",
            "progress": 0.0,
            "details": "",
            "error": None
        }
    progress_store[job_id].update({
        "current_step": step,
        "progress": min(100.0, max(0.0, progress)),
        "details": details
    })

async def run_benchmark_task(
    job_id: str,
    request: BenchmarkRequest,
    current_user: dict,
    pipeline_config: Dict[str, Any]
):
    """Run benchmark in background task"""
    try:
        # Update progress immediately
        update_progress(job_id, "Initializing", 1.0, "Setting up pipeline and evaluator...")
        await asyncio.sleep(0.1)  # Small delay to ensure progress is visible
        
        # Create pipeline
        pipeline = RAGPipeline(pipeline_config)
        
        # Create evaluator with progress callback
        evaluator = BenchmarkEvaluator(CORPUS_DIR)
        evaluator.set_progress_callback(lambda step, prog, details: update_progress(job_id, step, prog, details))
        
        # Determine languages for multilingual
        languages = ["en", "hi", "te"] if "multilingual" in request.benchmark_types else ["en"]
        
        # Run evaluations
        all_results = {}
        total_types = len(request.benchmark_types)
        
        for idx, benchmark_type in enumerate(request.benchmark_types):
            base_progress = 10 + (idx * 70 / total_types)
            update_progress(job_id, f"Running {benchmark_type}", base_progress, f"Evaluating {benchmark_type} benchmark...")
            
            if benchmark_type == "multilingual":
                result = evaluator.evaluate_pipeline(pipeline, "multilingual", languages, job_id=job_id, base_progress=base_progress, progress_range=70/total_types)
            else:
                result = evaluator.evaluate_pipeline(pipeline, benchmark_type, ["en"], job_id=job_id, base_progress=base_progress, progress_range=70/total_types)
            
            all_results[benchmark_type] = result
        
        update_progress(job_id, "Saving results", 85.0, "Saving benchmark results to database...")
        
        # Save to database
        db = get_db()
        benchmark_name = request.benchmark_name or f"Benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        benchmark_doc = {
            "user_id": str(current_user["_id"]),
            "username": current_user["username"],
            "benchmark_name": benchmark_name,
            "pipeline_config": pipeline_config,
            "benchmark_types": request.benchmark_types,
            "results": all_results,
            "created_at": datetime.utcnow(),
            "overall_score": calculate_overall_score(all_results)
        }
        
        result_id = await db.leaderboard.insert_one(benchmark_doc)
        
        progress_store[job_id].update({
            "status": "completed",
            "current_step": "Completed",
            "progress": 100.0,
            "details": "Benchmark completed successfully!",
            "benchmark_id": str(result_id.inserted_id),
            "benchmark_name": benchmark_name,
            "results": all_results,
            "pipeline_config": pipeline_config
        })
    
    except Exception as e:
        progress_store[job_id].update({
            "status": "error",
            "error": str(e),
            "details": f"Error: {str(e)}"
        })

@router.post("/run")
async def run_benchmark(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Run benchmark evaluation"""
    try:
        # Validate and create pipeline config
        if request.pipeline_json:
            pipeline_config = request.pipeline_json
            # Validate pipeline_json
            if "embedding_model" not in pipeline_config:
                raise HTTPException(status_code=400, detail="embedding_model is required in pipeline_json")
            if pipeline_config.get("chunk_size", 500) < 50 or pipeline_config.get("chunk_size", 500) > 5000:
                raise HTTPException(status_code=400, detail="chunk_size must be between 50 and 5000")
        elif request.config:
            # Validate prebuilt config
            if request.config.chunk_size < 50 or request.config.chunk_size > 5000:
                raise HTTPException(status_code=400, detail="chunk_size must be between 50 and 5000")
            if request.config.chunk_overlap < 0 or request.config.chunk_overlap >= request.config.chunk_size:
                raise HTTPException(status_code=400, detail="chunk_overlap must be between 0 and less than chunk_size")
            if request.config.top_k < 1 or request.config.top_k > 50:
                raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
            pipeline_config = request.config.dict()
        else:
            raise HTTPException(status_code=400, detail="Either config or pipeline_json must be provided")
        
        # Create job ID
        job_id = str(uuid.uuid4())
        progress_store[job_id] = {
            "status": "running",
            "current_step": "Initializing",
            "progress": 0.0,
            "details": "Starting benchmark...",
            "error": None
        }
        
        
        # Run in background
        background_tasks.add_task(
            run_benchmark_task,
            job_id,
            request,
            current_user,
            pipeline_config
        )
        
        return {
            "job_id": job_id,
            "message": "Benchmark started"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting benchmark: {str(e)}")

@router.get("/progress/{job_id}")
async def get_progress(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get progress for a benchmark job"""
    if job_id not in progress_store:
        raise HTTPException(status_code=404, detail="Job not found")
    
    progress_data = progress_store[job_id].copy()
    status = progress_data.get("status", "running")
    
    # If completed, return results immediately (no cleanup needed here, frontend should stop polling)
    if status == "completed":
        return {
            "status": "completed",
            "progress": 100.0,
            "current_step": "Completed",
            "details": "Benchmark completed successfully!",
            "benchmark_id": progress_data.get("benchmark_id"),
            "benchmark_name": progress_data.get("benchmark_name"),
            "results": progress_data.get("results")
        }
    
    # If error, return error immediately
    if status == "error":
        return {
            "status": "error",
            "error": progress_data.get("error"),
            "details": progress_data.get("details", "An error occurred")
        }
    
    # Return current progress
    return progress_data

@router.post("/upload-pipeline")
async def upload_pipeline(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload RAG pipeline JSON configuration"""
    try:
        content = await file.read()
        pipeline_json = json.loads(content.decode("utf-8"))
        
        # Validate pipeline JSON structure
        required_fields = ["embedding_model"]
        for field in required_fields:
            if field not in pipeline_json:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        return {"pipeline_config": pipeline_json, "message": "Pipeline uploaded successfully"}
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/results/{benchmark_id}")
async def get_benchmark_results(
    benchmark_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get benchmark results by ID"""
    from bson import ObjectId
    db = get_db()
    try:
        benchmark = await db.leaderboard.find_one({"_id": ObjectId(benchmark_id)})
    except:
        raise HTTPException(status_code=400, detail="Invalid benchmark ID")
    
    if not benchmark:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    # Check if user owns this benchmark or it's public
    if str(benchmark["user_id"]) != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Access denied")
    
    benchmark["_id"] = str(benchmark["_id"])
    return benchmark

def calculate_overall_score(results: Dict[str, Any]) -> float:
    """Calculate overall score from all benchmark results"""
    scores = []
    for benchmark_type, result in results.items():
        metrics = result.get("overall_metrics", {})
        # Weighted average: F1 (40%), similarity (30%), precision (20%), recall (10%)
        score = (
            metrics.get("f1_score", 0) * 0.4 +
            metrics.get("similarity_score", 0) * 0.3 +
            metrics.get("precision", 0) * 0.2 +
            metrics.get("recall", 0) * 0.1
        )
        scores.append(score)
    
    return float(sum(scores) / len(scores)) if scores else 0.0

