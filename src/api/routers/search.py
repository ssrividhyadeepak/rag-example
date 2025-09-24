"""
Search Router

Handles log search and retrieval endpoints.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query

from ..models import SearchRequest, SearchResponse, LogEntry
from ..dependencies import get_rag_pipeline, get_vector_store
from src.rag_engine import ATMRagPipeline
from src.vector_store import HybridVectorStore

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/search", response_model=SearchResponse)
async def search_logs(
    request: SearchRequest,
    rag_pipeline: ATMRagPipeline = Depends(get_rag_pipeline)
):
    """
    Search ATM logs using semantic similarity.

    Performs semantic search across ATM logs to find relevant entries
    based on the search query. Supports various filters for precise results.
    """
    start_time = time.time()

    try:
        logger.info(f"Searching logs for: {request.query}")

        # Build filters from request
        filters = request.filters or {}

        # Add specific filters if provided
        if request.operation_type:
            filters["operation"] = request.operation_type
        if request.status:
            filters["status"] = request.status
        if request.atm_id:
            filters["atm_id"] = request.atm_id
        if request.error_code:
            filters["error_code"] = request.error_code

        # Add time range filter
        if request.start_date or request.end_date:
            time_filter = {}
            if request.start_date:
                time_filter["$gte"] = request.start_date
            if request.end_date:
                time_filter["$lte"] = request.end_date
            filters["timestamp"] = time_filter

        # Perform search
        search_results = rag_pipeline.search_logs(
            query=request.query,
            top_k=request.top_k,
            filters=filters
        )

        # Convert results to response format
        log_entries = []
        for result in search_results.get("results", []):
            log_entry = LogEntry(
                log_id=result["log_id"],
                timestamp=result["timestamp"],
                operation=result["operation"],
                status=result["status"],
                message=result["message"],
                atm_id=result.get("atm_id"),
                error_code=result.get("error_code"),
                amount=result.get("amount"),
                similarity_score=result.get("similarity_score")
            )
            log_entries.append(log_entry)

        processing_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            results=log_entries,
            total_found=search_results.get("total_found", len(log_entries)),
            query=request.query,
            filters_applied=filters,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.error(f"Error searching logs: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/search/recent")
async def search_recent_logs(
    hours: int = Query(24, ge=1, le=168, description="Hours back to search"),
    operation: Optional[str] = Query(None, description="Filter by operation type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    atm_id: Optional[str] = Query(None, description="Filter by ATM ID"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
    rag_pipeline: ATMRagPipeline = Depends(get_rag_pipeline)
):
    """
    Get recent ATM logs within specified time range.

    Retrieves the most recent ATM logs, optionally filtered by various criteria.
    Useful for monitoring recent activity and identifying patterns.
    """
    try:
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # Build filters
        filters = {
            "timestamp": {
                "$gte": start_time,
                "$lte": end_time
            }
        }

        if operation:
            filters["operation"] = operation
        if status:
            filters["status"] = status
        if atm_id:
            filters["atm_id"] = atm_id

        # Get recent issues using RAG pipeline
        recent_results = rag_pipeline.get_recent_issues(
            hours_back=hours,
            filters=filters,
            limit=limit
        )

        # Convert to log entries
        log_entries = []
        results = recent_results.get("results", [])

        for result in results:
            log_entry = LogEntry(
                log_id=result["log_id"],
                timestamp=result["timestamp"],
                operation=result["operation"],
                status=result["status"],
                message=result["message"],
                atm_id=result.get("atm_id"),
                error_code=result.get("error_code"),
                amount=result.get("amount")
            )
            log_entries.append(log_entry)

        return {
            "results": log_entries,
            "total_found": len(log_entries),
            "time_range": {
                "start": start_time,
                "end": end_time,
                "hours": hours
            },
            "filters_applied": filters,
            "summary": recent_results.get("response", "Recent logs retrieved")
        }

    except Exception as e:
        logger.error(f"Error getting recent logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent logs: {str(e)}")


@router.get("/search/errors")
async def search_error_logs(
    error_code: Optional[str] = Query(None, description="Specific error code"),
    atm_id: Optional[str] = Query(None, description="Filter by ATM ID"),
    hours: int = Query(24, ge=1, le=168, description="Hours back to search"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    rag_pipeline: ATMRagPipeline = Depends(get_rag_pipeline)
):
    """
    Search for error logs specifically.

    Finds logs with error conditions, optionally filtered by error code,
    ATM ID, and time range. Useful for troubleshooting and error analysis.
    """
    try:
        # Build error-specific query
        if error_code:
            search_query = f"error {error_code}"
        else:
            search_query = "error failed denied timeout"

        # Build filters
        filters = {
            "status": {"$in": ["denied", "failed", "error", "timeout"]},
            "timestamp": {
                "$gte": datetime.utcnow() - timedelta(hours=hours)
            }
        }

        if error_code:
            filters["error_code"] = error_code
        if atm_id:
            filters["atm_id"] = atm_id

        # Search for errors
        search_results = rag_pipeline.search_logs(
            query=search_query,
            top_k=limit,
            filters=filters
        )

        # Convert results
        log_entries = []
        for result in search_results.get("results", []):
            log_entry = LogEntry(
                log_id=result["log_id"],
                timestamp=result["timestamp"],
                operation=result["operation"],
                status=result["status"],
                message=result["message"],
                atm_id=result.get("atm_id"),
                error_code=result.get("error_code"),
                amount=result.get("amount"),
                similarity_score=result.get("similarity_score")
            )
            log_entries.append(log_entry)

        # Group by error code for summary
        error_summary = {}
        for entry in log_entries:
            if entry.error_code:
                error_summary[entry.error_code] = error_summary.get(entry.error_code, 0) + 1

        return {
            "results": log_entries,
            "total_found": len(log_entries),
            "error_summary": error_summary,
            "filters_applied": filters,
            "time_range_hours": hours
        }

    except Exception as e:
        logger.error(f"Error searching error logs: {e}")
        raise HTTPException(status_code=500, detail=f"Error search failed: {str(e)}")


@router.get("/search/operations/{operation}")
async def search_by_operation(
    operation: str,
    status: Optional[str] = Query(None, description="Filter by status"),
    atm_id: Optional[str] = Query(None, description="Filter by ATM ID"),
    hours: int = Query(24, ge=1, le=168, description="Hours back to search"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    rag_pipeline: ATMRagPipeline = Depends(get_rag_pipeline)
):
    """
    Search logs by operation type.

    Find all logs for a specific operation (withdrawal, deposit, balance_inquiry, etc.)
    with optional additional filters.
    """
    try:
        # Validate operation type
        valid_operations = ["withdrawal", "deposit", "balance_inquiry", "transfer", "card_operation"]
        if operation not in valid_operations:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid operation. Must be one of: {', '.join(valid_operations)}"
            )

        # Build filters
        filters = {
            "operation": operation,
            "timestamp": {
                "$gte": datetime.utcnow() - timedelta(hours=hours)
            }
        }

        if status:
            filters["status"] = status
        if atm_id:
            filters["atm_id"] = atm_id

        # Search for operations
        search_results = rag_pipeline.search_logs(
            query=f"{operation} transaction",
            top_k=limit,
            filters=filters
        )

        # Convert results
        log_entries = []
        status_summary = {}

        for result in search_results.get("results", []):
            log_entry = LogEntry(
                log_id=result["log_id"],
                timestamp=result["timestamp"],
                operation=result["operation"],
                status=result["status"],
                message=result["message"],
                atm_id=result.get("atm_id"),
                error_code=result.get("error_code"),
                amount=result.get("amount"),
                similarity_score=result.get("similarity_score")
            )
            log_entries.append(log_entry)

            # Count statuses
            status_summary[result["status"]] = status_summary.get(result["status"], 0) + 1

        return {
            "results": log_entries,
            "total_found": len(log_entries),
            "operation": operation,
            "status_summary": status_summary,
            "filters_applied": filters,
            "time_range_hours": hours
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching by operation: {e}")
        raise HTTPException(status_code=500, detail=f"Operation search failed: {str(e)}")


@router.get("/search/atm/{atm_id}")
async def search_by_atm(
    atm_id: str,
    operation: Optional[str] = Query(None, description="Filter by operation"),
    status: Optional[str] = Query(None, description="Filter by status"),
    hours: int = Query(24, ge=1, le=168, description="Hours back to search"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    rag_pipeline: ATMRagPipeline = Depends(get_rag_pipeline)
):
    """
    Search logs for a specific ATM.

    Get all activity for a particular ATM with optional filters.
    Useful for analyzing individual ATM performance and issues.
    """
    try:
        # Build filters
        filters = {
            "atm_id": atm_id,
            "timestamp": {
                "$gte": datetime.utcnow() - timedelta(hours=hours)
            }
        }

        if operation:
            filters["operation"] = operation
        if status:
            filters["status"] = status

        # Search for ATM logs
        search_results = rag_pipeline.search_logs(
            query=f"ATM {atm_id}",
            top_k=limit,
            filters=filters
        )

        # Convert results and analyze
        log_entries = []
        operation_summary = {}
        status_summary = {}

        for result in search_results.get("results", []):
            log_entry = LogEntry(
                log_id=result["log_id"],
                timestamp=result["timestamp"],
                operation=result["operation"],
                status=result["status"],
                message=result["message"],
                atm_id=result.get("atm_id"),
                error_code=result.get("error_code"),
                amount=result.get("amount"),
                similarity_score=result.get("similarity_score")
            )
            log_entries.append(log_entry)

            # Count operations and statuses
            operation_summary[result["operation"]] = operation_summary.get(result["operation"], 0) + 1
            status_summary[result["status"]] = status_summary.get(result["status"], 0) + 1

        # Calculate success rate
        total_operations = len(log_entries)
        successful_operations = status_summary.get("success", 0)
        success_rate = (successful_operations / total_operations * 100) if total_operations > 0 else 0

        return {
            "results": log_entries,
            "total_found": len(log_entries),
            "atm_id": atm_id,
            "operation_summary": operation_summary,
            "status_summary": status_summary,
            "success_rate_percentage": round(success_rate, 2),
            "filters_applied": filters,
            "time_range_hours": hours
        }

    except Exception as e:
        logger.error(f"Error searching by ATM: {e}")
        raise HTTPException(status_code=500, detail=f"ATM search failed: {str(e)}")