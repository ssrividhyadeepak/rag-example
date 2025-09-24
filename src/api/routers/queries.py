"""
Query Processing Router

Handles natural language query processing endpoints.
"""

import time
import logging
from typing import List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from ..models import (
    QueryRequest, QueryResponse, BatchQueryRequest, BatchQueryResponse,
    ErrorResponse
)
from ..dependencies import get_rag_pipeline, get_query_processor
from src.rag_engine import ATMRagPipeline
from src.query_processor import QueryProcessor

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    rag_pipeline: ATMRagPipeline = Depends(get_rag_pipeline),
    query_processor: QueryProcessor = Depends(get_query_processor)
):
    """
    Process a natural language query about ATM operations.

    This endpoint accepts natural language queries and returns intelligent responses
    based on the ATM log data using RAG (Retrieval Augmented Generation).

    Examples:
    - "Why is ATM001 showing DDL_EXCEEDED errors?"
    - "What does NETWORK_ERROR mean?"
    - "Analyze withdrawal failures in the last 24 hours"
    """
    start_time = time.time()

    try:
        logger.info(f"Processing query: {request.query}")

        # Validate query
        validation_result = query_processor.validate_query(request.query)
        if not validation_result["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid query: {validation_result.get('message', 'Query validation failed')}"
            )

        if not validation_result["is_atm_related"]:
            logger.warning(f"Non-ATM related query: {request.query}")
            return QueryResponse(
                response="I can only help with ATM-related questions. Please ask about ATM operations, errors, troubleshooting, or analysis.",
                response_type="info",
                query=request.query,
                confidence=0.0,
                sources_count=0,
                metadata={"warning": "Query not ATM-related"},
                generated_at=datetime.utcnow(),
                processing_time_ms=(time.time() - start_time) * 1000
            )

        # Process query to extract intent and entities
        processed_query = query_processor.process_query(request.query)

        # Merge request filters with optimized filters
        filters = request.filters or {}
        filters.update(processed_query.optimized_filters)

        # Use processed parameters or request defaults
        top_k = request.top_k if request.top_k != 5 else processed_query.suggested_top_k
        response_type = request.response_type.value if request.response_type.value != "auto" else processed_query.response_type

        # Generate response using RAG pipeline
        rag_response = await rag_pipeline.process_query(
            query=request.query,
            filters=filters,
            top_k=top_k,
            response_type=response_type
        )

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Build response
        response = QueryResponse(
            response=rag_response["response"],
            response_type=rag_response["response_type"],
            query=request.query,
            confidence=rag_response["confidence"],
            sources_count=rag_response["sources_count"],
            metadata=rag_response.get("metadata", {}) if request.include_metadata else None,
            generated_at=rag_response["generated_at"],
            processing_time_ms=processing_time_ms
        )

        logger.info(f"Query processed successfully in {processing_time_ms:.2f}ms")
        return response

    except HTTPException:
        raise
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        logger.error(f"Error processing query: {e}")

        return QueryResponse(
            response="I apologize, but I encountered an error while processing your query. Please try again or contact support if the issue persists.",
            response_type="error",
            query=request.query,
            confidence=0.0,
            sources_count=0,
            metadata={"error": str(e), "processing_time_ms": processing_time_ms},
            generated_at=datetime.utcnow(),
            processing_time_ms=processing_time_ms
        )


@router.post("/query/batch", response_model=BatchQueryResponse)
async def process_batch_queries(
    request: BatchQueryRequest,
    rag_pipeline: ATMRagPipeline = Depends(get_rag_pipeline),
    query_processor: QueryProcessor = Depends(get_query_processor)
):
    """
    Process multiple queries in batch.

    Efficient processing of multiple queries at once. Useful for analysis
    or when you have several related questions.
    """
    start_time = time.time()

    try:
        logger.info(f"Processing batch of {len(request.queries)} queries")

        # Process queries in batch
        batch_results = rag_pipeline.batch_process_queries(
            queries=request.queries,
            response_type=request.response_type.value,
            top_k=request.top_k
        )

        # Convert to response format
        responses = []
        successful_count = 0
        failed_count = 0

        for i, (query, result) in enumerate(zip(request.queries, batch_results)):
            try:
                if "error" not in result:
                    response = QueryResponse(
                        response=result["response"],
                        response_type=result["response_type"],
                        query=query,
                        confidence=result["confidence"],
                        sources_count=result["sources_count"],
                        metadata=result.get("metadata"),
                        generated_at=result.get("generated_at", datetime.utcnow()),
                        processing_time_ms=result.get("processing_time_ms", 0)
                    )
                    successful_count += 1
                else:
                    response = QueryResponse(
                        response=f"Error processing query: {result['error']}",
                        response_type="error",
                        query=query,
                        confidence=0.0,
                        sources_count=0,
                        metadata={"error": result["error"]},
                        generated_at=datetime.utcnow(),
                        processing_time_ms=0
                    )
                    failed_count += 1

                responses.append(response)

            except Exception as e:
                logger.error(f"Error processing batch query {i}: {e}")
                failed_count += 1
                responses.append(QueryResponse(
                    response="Error processing this query",
                    response_type="error",
                    query=query,
                    confidence=0.0,
                    sources_count=0,
                    metadata={"error": str(e)},
                    generated_at=datetime.utcnow(),
                    processing_time_ms=0
                ))

        total_processing_time_ms = (time.time() - start_time) * 1000

        return BatchQueryResponse(
            responses=responses,
            total_queries=len(request.queries),
            successful_queries=successful_count,
            failed_queries=failed_count,
            total_processing_time_ms=total_processing_time_ms
        )

    except Exception as e:
        logger.error(f"Error processing batch queries: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


@router.post("/query/validate")
async def validate_query(
    query: str,
    query_processor: QueryProcessor = Depends(get_query_processor)
):
    """
    Validate a query without processing it.

    Useful for checking if a query is valid and ATM-related before sending
    it to the main processing endpoint.
    """
    try:
        validation_result = query_processor.validate_query(query)
        processed_query = query_processor.process_query(query)

        return {
            "is_valid": validation_result["is_valid"],
            "is_atm_related": validation_result["is_atm_related"],
            "detected_intent": processed_query.intent_result.intent.value,
            "confidence": processed_query.intent_result.confidence,
            "extracted_entities": [
                {
                    "type": entity.entity_type,
                    "value": entity.value,
                    "normalized_value": entity.normalized_value
                }
                for entity in processed_query.entity_result.entities
            ],
            "suggested_response_type": processed_query.response_type,
            "suggested_top_k": processed_query.suggested_top_k,
            "suggestions": validation_result.get("suggestions", [])
        }

    except Exception as e:
        logger.error(f"Error validating query: {e}")
        raise HTTPException(status_code=500, detail=f"Query validation failed: {str(e)}")


@router.get("/query/examples")
async def get_query_examples():
    """
    Get example queries for different use cases.

    Returns a collection of example queries that demonstrate the system's capabilities.
    """
    examples = {
        "troubleshooting": [
            "Why is ATM001 showing DDL_EXCEEDED errors?",
            "Fix withdrawal problems on ATM002",
            "ATM003 keeps timing out during transactions",
            "Customer complaints about ATM004 not dispensing cash"
        ],
        "error_explanation": [
            "What does DDL_EXCEEDED mean?",
            "Explain NETWORK_ERROR code",
            "What causes CARD_ERROR?",
            "TIMEOUT error explanation"
        ],
        "analysis": [
            "Analyze withdrawal failures in the last 24 hours",
            "Show ATM performance trends",
            "Count network errors by ATM",
            "Which ATMs have the most issues?"
        ],
        "search": [
            "Show me logs for ATM001 yesterday",
            "Find all deposit failures",
            "Recent balance inquiry errors",
            "Successful transactions in the morning"
        ],
        "general": [
            "System status overview",
            "Recent issues summary",
            "ATM availability report",
            "Help with transaction monitoring"
        ]
    }

    return {
        "examples": examples,
        "total_examples": sum(len(queries) for queries in examples.values()),
        "categories": list(examples.keys()),
        "usage_tips": [
            "Be specific about ATM IDs when known (e.g., ATM001, ATM002)",
            "Include time ranges for better analysis (e.g., 'last 24 hours', 'yesterday')",
            "Mention specific error codes if known (e.g., DDL_EXCEEDED, NETWORK_ERROR)",
            "Ask about specific operations (withdrawal, deposit, balance_inquiry)",
            "Use natural language - the system understands conversational queries"
        ]
    }