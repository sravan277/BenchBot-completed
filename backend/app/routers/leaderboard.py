from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.database import get_db
from bson import ObjectId

router = APIRouter()

@router.get("/")
async def get_leaderboard(
    sort_by: str = Query("overall_score", description="Sort by: overall_score, f1_score, precision, recall, similarity_score, latency"),
    order: str = Query("desc", description="Order: asc or desc"),
    limit: int = Query(100, description="Number of results to return"),
    benchmark_type: Optional[str] = Query(None, description="Filter by benchmark type")
):
    """Get leaderboard of all benchmarks"""
    db = get_db()
    
    # Build query
    query = {}
    if benchmark_type:
        query["benchmark_types"] = benchmark_type
    
    # Build sort
    sort_order = -1 if order == "desc" else 1
    
    # Map sort_by to actual field
    if sort_by == "overall_score":
        sort_field = "overall_score"
    elif sort_by in ["f1_score", "precision", "recall", "similarity_score", "latency"]:
        # Need to extract from results
        sort_field = f"results.{benchmark_type or 'single'}.overall_metrics.{sort_by}"
    else:
        sort_field = "overall_score"
    
    # Get benchmarks
    cursor = db.leaderboard.find(query).sort(sort_field, sort_order).limit(limit)
    benchmarks = await cursor.to_list(length=limit)
    
    # Format results
    leaderboard = []
    for benchmark in benchmarks:
        # Extract metrics for sorting
        metrics = {}
        for bt in benchmark.get("benchmark_types", []):
            if bt in benchmark.get("results", {}):
                overall = benchmark["results"][bt].get("overall_metrics", {})
                metrics[bt] = {
                    "f1_score": overall.get("f1_score", 0),
                    "precision": overall.get("precision", 0),
                    "recall": overall.get("recall", 0),
                    "similarity_score": overall.get("similarity_score", 0),
                    "latency": overall.get("latency", 0)
                }
        
        leaderboard.append({
            "id": str(benchmark["_id"]),
            "username": benchmark.get("username", "Unknown"),
            "benchmark_name": benchmark.get("benchmark_name", f"Benchmark {str(benchmark['_id'])[:8]}"),
            "pipeline_config": benchmark.get("pipeline_config", {}),
            "benchmark_types": benchmark.get("benchmark_types", []),
            "overall_score": benchmark.get("overall_score", 0),
            "metrics": metrics,
            "created_at": benchmark.get("created_at").isoformat() if benchmark.get("created_at") else None
        })
    
    return {"leaderboard": leaderboard, "total": len(leaderboard)}

@router.get("/{benchmark_id}")
async def get_benchmark_detail(benchmark_id: str):
    """Get detailed benchmark results"""
    from fastapi import HTTPException
    db = get_db()
    
    try:
        benchmark = await db.leaderboard.find_one({"_id": ObjectId(benchmark_id)})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid benchmark ID: {str(e)}")
    
    if not benchmark:
        raise HTTPException(status_code=404, detail="Benchmark not found")
    
    # Format response
    benchmark["_id"] = str(benchmark["_id"])
    benchmark["user_id"] = str(benchmark["user_id"])
    if benchmark.get("created_at"):
        benchmark["created_at"] = benchmark["created_at"].isoformat()
    
    return benchmark

