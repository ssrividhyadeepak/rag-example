"""
Management Router

Handles system management and statistics endpoints.
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..models import StatsResponse, HealthCheckResponse, ErrorResponse
from ..dependencies import get_rag_pipeline, get_vector_store
from src.rag_engine import ATMRagPipeline
from src.vector_store import HybridVectorStore

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stats", response_model=StatsResponse)
async def get_system_statistics(
    rag_pipeline: ATMRagPipeline = Depends(get_rag_pipeline),
    vector_store: HybridVectorStore = Depends(get_vector_store)
):
    """
    Get comprehensive system statistics.

    Returns information about:
    - Query processing metrics
    - Vector store statistics
    - System performance metrics
    """
    try:
        logger.info("Collecting system statistics")

        # Get pipeline stats
        pipeline_stats = rag_pipeline.get_statistics()

        # Get vector store stats
        try:
            vector_stats = vector_store.get_statistics()
        except Exception as e:
            logger.warning(f"Could not get vector store stats: {e}")
            vector_stats = {
                "total_documents": 0,
                "total_vectors": 0,
                "index_size_mb": 0,
                "last_updated": None
            }

        # System metrics
        current_time = datetime.utcnow()

        return {
            "timestamp": current_time,
            "query_stats": {
                "total_queries": pipeline_stats.get("total_queries", 0),
                "successful_queries": pipeline_stats.get("successful_queries", 0),
                "failed_queries": pipeline_stats.get("failed_queries", 0),
                "average_response_time": pipeline_stats.get("average_response_time", 0.0),
                "queries_per_hour": pipeline_stats.get("queries_per_hour", 0.0)
            },
            "vector_store_stats": {
                "total_documents": vector_stats.get("total_documents", 0),
                "total_vectors": vector_stats.get("total_vectors", 0),
                "index_size_mb": vector_stats.get("index_size_mb", 0.0),
                "last_updated": vector_stats.get("last_updated")
            },
            "system_stats": {
                "uptime_seconds": time.time() - getattr(rag_pipeline, '_start_time', time.time()),
                "memory_usage_mb": 0,  # Could implement actual memory monitoring
                "cpu_usage_percent": 0,  # Could implement actual CPU monitoring
                "disk_usage_mb": 0  # Could implement actual disk monitoring
            },
            "component_health": {
                "rag_engine": "healthy" if rag_pipeline else "unhealthy",
                "vector_store": "healthy" if vector_store else "unhealthy",
                "embeddings": "healthy",  # Assume healthy if we got here
                "database": "unknown"  # Would need MongoDB connection check
            }
        }

    except Exception as e:
        logger.error(f"Failed to collect system statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.get("/health/detailed", response_model=HealthCheckResponse)
async def detailed_health_check(
    rag_pipeline: ATMRagPipeline = Depends(get_rag_pipeline),
    vector_store: HybridVectorStore = Depends(get_vector_store)
):
    """
    Comprehensive health check of all system components.
    """
    try:
        logger.info("Performing detailed health check")

        components = {}
        overall_status = "healthy"

        # Check RAG Pipeline
        try:
            pipeline_health = rag_pipeline.health_check()
            components["rag_engine"] = {
                "status": "healthy" if pipeline_health.get("status") == "healthy" else "unhealthy",
                "details": pipeline_health.get("details", "RAG engine operational")
            }
        except Exception as e:
            components["rag_engine"] = {
                "status": "unhealthy",
                "details": f"RAG engine error: {str(e)}"
            }
            overall_status = "unhealthy"

        # Check Vector Store
        try:
            vector_health = vector_store.health_check()
            components["vector_store"] = {
                "status": "healthy" if vector_health.get("status") == "healthy" else "unhealthy",
                "details": vector_health.get("details", "Vector store operational")
            }
        except Exception as e:
            components["vector_store"] = {
                "status": "unhealthy",
                "details": f"Vector store error: {str(e)}"
            }
            overall_status = "unhealthy"

        # Check Embeddings (basic check)
        try:
            # Try to generate a test embedding
            test_embedding = rag_pipeline.embedding_generator.generate_embeddings(["test"])
            if test_embedding is not None and len(test_embedding) > 0:
                components["embeddings"] = {
                    "status": "healthy",
                    "details": "Embedding generation operational"
                }
            else:
                components["embeddings"] = {
                    "status": "unhealthy",
                    "details": "Embedding generation failed"
                }
                overall_status = "unhealthy"
        except Exception as e:
            components["embeddings"] = {
                "status": "unhealthy",
                "details": f"Embeddings error: {str(e)}"
            }
            overall_status = "unhealthy"

        # Database check (placeholder)
        components["database"] = {
            "status": "unknown",
            "details": "Database health check not implemented"
        }

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow(),
            "components": components,
            "uptime_seconds": time.time() - getattr(rag_pipeline, '_start_time', time.time()),
            "version": "1.0.0"
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow(),
            "components": {
                "system": {
                    "status": "unhealthy",
                    "details": f"Health check failed: {str(e)}"
                }
            },
            "uptime_seconds": 0,
            "version": "1.0.0"
        }


@router.post("/system/restart")
async def restart_system():
    """
    Restart system components (placeholder).
    """
    try:
        logger.info("System restart requested")

        # In a real implementation, this would restart components
        return JSONResponse({
            "message": "System restart initiated",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        })

    except Exception as e:
        logger.error(f"System restart failed: {e}")
        raise HTTPException(status_code=500, detail=f"Restart failed: {str(e)}")


@router.get("/system/version")
async def get_system_version():
    """
    Get system version information.
    """
    return {
        "version": "1.0.0",
        "build": "development",
        "timestamp": datetime.utcnow(),
        "components": {
            "api": "1.0.0",
            "rag_engine": "1.0.0",
            "vector_store": "1.0.0",
            "embeddings": "sentence-transformers"
        }
    }