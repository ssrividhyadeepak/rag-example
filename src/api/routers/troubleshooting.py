"""
Troubleshooting Router

Handles ATM troubleshooting and error analysis endpoints.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query

from ..models import (
    TroubleshootRequest, TroubleshootResponse, AnalyzeRequest, AnalyzeResponse,
    LogEntry, AnalysisData
)
from ..dependencies import get_rag_pipeline, get_query_processor
from src.rag_engine import ATMRagPipeline
from src.query_processor import QueryProcessor

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/troubleshoot", response_model=TroubleshootResponse)
async def troubleshoot_issue(
    request: TroubleshootRequest,
    rag_pipeline: ATMRagPipeline = Depends(get_rag_pipeline)
):
    """
    Get troubleshooting guidance for ATM issues.

    Analyzes ATM problems and provides specific troubleshooting steps
    based on historical data and similar cases.
    """
    try:
        logger.info(f"Troubleshooting request: error_code={request.error_code}, atm_id={request.atm_id}")

        # Use specific troubleshooting method if error code is provided
        if request.error_code:
            result = rag_pipeline.troubleshoot_error(
                error_code=request.error_code,
                atm_id=request.atm_id,
                operation=request.operation,
                recent_only=request.recent_only
            )
        else:
            # General troubleshooting based on description
            description = request.description or "general ATM issue"
            result = rag_pipeline.process_query_sync(
                query=f"troubleshoot {description}",
                response_type="troubleshooting"
            )

        # Extract troubleshooting steps from response
        response_text = result.get("response", "")
        troubleshooting_steps = []

        # Parse response for step-by-step guidance
        if "steps:" in response_text.lower() or "step " in response_text.lower():
            lines = response_text.split('\n')
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ["step", "check", "verify", "ensure", "test"]):
                    if line and not line.startswith("Step"):
                        line = f"Step: {line}"
                    troubleshooting_steps.append(line)
        else:
            # Generic steps if no specific steps found
            troubleshooting_steps = [
                "Check ATM physical status and error displays",
                "Verify network connectivity and communication",
                "Review recent transaction logs for patterns",
                "Test basic ATM functions if accessible",
                "Contact technical support if issue persists"
            ]

        # Get similar cases
        similar_cases = []
        sources = result.get("sources", [])
        for source in sources[:5]:  # Limit to 5 similar cases
            log_entry = LogEntry(
                log_id=source.get("log_id", ""),
                timestamp=source.get("timestamp", datetime.utcnow()),
                operation=source.get("operation", ""),
                status=source.get("status", ""),
                message=source.get("message", ""),
                atm_id=source.get("atm_id"),
                error_code=source.get("error_code"),
                amount=source.get("amount"),
                similarity_score=source.get("similarity_score")
            )
            similar_cases.append(log_entry)

        # Generate root cause analysis
        root_cause = _analyze_root_cause(request, result, similar_cases)

        # Estimate resolution time
        resolution_time = _estimate_resolution_time(request.error_code, len(similar_cases))

        return TroubleshootResponse(
            troubleshooting_steps=troubleshooting_steps,
            similar_cases=similar_cases,
            root_cause_analysis=root_cause,
            confidence=result.get("confidence", 0.5),
            estimated_resolution_time=resolution_time
        )

    except Exception as e:
        logger.error(f"Error in troubleshooting: {e}")
        raise HTTPException(status_code=500, detail=f"Troubleshooting failed: {str(e)}")


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_performance(
    request: AnalyzeRequest,
    rag_pipeline: ATMRagPipeline = Depends(get_rag_pipeline)
):
    """
    Perform analysis on ATM performance and operations.

    Analyzes ATM data to identify trends, patterns, and performance metrics
    over specified time ranges.
    """
    try:
        logger.info(f"Analysis request: type={request.analysis_type}, hours={request.time_range_hours}")

        # Build analysis query based on type
        if request.analysis_type == "performance":
            query = f"analyze ATM performance trends in last {request.time_range_hours} hours"
        elif request.analysis_type == "errors":
            query = f"analyze error patterns in last {request.time_range_hours} hours"
        elif request.analysis_type == "operations":
            query = f"analyze operation statistics in last {request.time_range_hours} hours"
        else:
            query = f"analyze ATM {request.analysis_type} in last {request.time_range_hours} hours"

        # Perform analysis
        result = rag_pipeline.process_query_sync(
            query=query,
            filters=request.filters,
            response_type="analysis"
        )

        # Extract metrics from the analysis
        key_metrics = _extract_key_metrics(result, request.time_range_hours)

        # Generate trends
        trends = _generate_trends_analysis(result, request.include_trends)

        # Generate recommendations
        recommendations = _generate_recommendations(result, key_metrics)

        # Format time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=request.time_range_hours)
        time_range = f"{start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')} UTC"

        return AnalyzeResponse(
            analysis_summary=result.get("response", "Analysis completed"),
            key_metrics=key_metrics,
            trends=trends,
            recommendations=recommendations,
            time_range=time_range,
            analysis_confidence=result.get("confidence", 0.7)
        )

    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/troubleshoot/error/{error_code}")
async def troubleshoot_error_code(
    error_code: str,
    atm_id: Optional[str] = Query(None, description="Specific ATM ID"),
    recent_only: bool = Query(True, description="Focus on recent occurrences"),
    rag_pipeline: ATMRagPipeline = Depends(get_rag_pipeline)
):
    """
    Get specific troubleshooting guidance for an error code.

    Provides detailed information about a specific error code including
    common causes, solutions, and related cases.
    """
    try:
        # Get troubleshooting guidance
        result = rag_pipeline.troubleshoot_error(
            error_code=error_code,
            atm_id=atm_id,
            recent_only=recent_only
        )

        # Get error explanation
        explanation_result = rag_pipeline.process_query_sync(
            query=f"what does {error_code} mean",
            response_type="error"
        )

        return {
            "error_code": error_code,
            "explanation": explanation_result.get("response", "Error code explanation not available"),
            "troubleshooting_guidance": result.get("response", "No specific guidance available"),
            "confidence": result.get("confidence", 0.5),
            "similar_cases_count": result.get("sources_count", 0),
            "atm_filter": atm_id,
            "recent_only": recent_only
        }

    except Exception as e:
        logger.error(f"Error troubleshooting error code {error_code}: {e}")
        raise HTTPException(status_code=500, detail=f"Error code troubleshooting failed: {str(e)}")


@router.get("/analyze/summary")
async def get_analysis_summary(
    hours: int = Query(24, ge=1, le=168, description="Hours back to analyze"),
    rag_pipeline: ATMRagPipeline = Depends(get_rag_pipeline)
):
    """
    Get a quick analysis summary of ATM operations.

    Provides a high-level overview of ATM performance and issues
    over the specified time period.
    """
    try:
        # Get recent issues analysis
        recent_analysis = rag_pipeline.get_recent_issues(hours_back=hours)

        # Get general statistics from the pipeline
        pipeline_stats = rag_pipeline.get_pipeline_statistics()

        # Generate summary
        summary = {
            "time_period_hours": hours,
            "analysis_timestamp": datetime.utcnow(),
            "overview": recent_analysis.get("response", "No recent issues found"),
            "issues_found": recent_analysis.get("sources_count", 0),
            "system_health": pipeline_stats.get("pipeline_stats", {}),
            "quick_stats": {
                "total_logs_in_system": pipeline_stats.get("vector_store_stats", {}).get("mongodb_docs", 0),
                "analysis_confidence": recent_analysis.get("confidence", 0.0)
            }
        }

        return summary

    except Exception as e:
        logger.error(f"Error generating analysis summary: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis summary failed: {str(e)}")


# Helper functions
def _analyze_root_cause(request: TroubleshootRequest, result: Dict[str, Any], similar_cases: List[LogEntry]) -> str:
    """Analyze potential root cause based on available data."""
    if request.error_code:
        error_patterns = {
            "DDL_EXCEEDED": "Daily withdrawal limit exceeded - customer or system limit configuration issue",
            "NETWORK_ERROR": "Network connectivity issue - check network infrastructure and connections",
            "CARD_ERROR": "Card reading or processing issue - check card reader hardware",
            "TIMEOUT": "Transaction timeout - network latency or system performance issue",
            "INSUFFICIENT_FUNDS": "Account balance issue - customer account or bank system problem"
        }

        base_cause = error_patterns.get(request.error_code, f"Error code {request.error_code} requires investigation")

        if len(similar_cases) > 3:
            return f"{base_cause}. Pattern detected: {len(similar_cases)} similar cases found, suggesting systemic issue."
        else:
            return f"{base_cause}. Isolated incident with {len(similar_cases)} similar cases."

    return "Root cause requires additional investigation based on specific error patterns and symptoms."


def _estimate_resolution_time(error_code: Optional[str], similar_cases_count: int) -> str:
    """Estimate resolution time based on error type and complexity."""
    time_estimates = {
        "DDL_EXCEEDED": "5-15 minutes (configuration change)",
        "NETWORK_ERROR": "15-60 minutes (network troubleshooting)",
        "CARD_ERROR": "30-120 minutes (hardware inspection/replacement)",
        "TIMEOUT": "10-30 minutes (performance optimization)",
        "INSUFFICIENT_FUNDS": "5-10 minutes (account verification)"
    }

    if error_code and error_code in time_estimates:
        base_time = time_estimates[error_code]
        if similar_cases_count > 5:
            return f"{base_time} (may take longer due to systemic nature)"
        return base_time

    return "15-60 minutes (depends on issue complexity)"


def _extract_key_metrics(result: Dict[str, Any], time_range_hours: int) -> List[AnalysisData]:
    """Extract key metrics from analysis result."""
    metrics = []

    # Default metrics based on sources count
    sources_count = result.get("sources_count", 0)

    metrics.append(AnalysisData(
        metric_name="Total Events",
        current_value=sources_count,
        trend="stable",
        change_percentage=0.0
    ))

    metrics.append(AnalysisData(
        metric_name="Analysis Period",
        current_value=f"{time_range_hours} hours",
        trend=None,
        change_percentage=None
    ))

    metrics.append(AnalysisData(
        metric_name="Data Confidence",
        current_value=f"{result.get('confidence', 0.5):.1%}",
        trend="stable",
        change_percentage=None
    ))

    return metrics


def _generate_trends_analysis(result: Dict[str, Any], include_trends: bool) -> Dict[str, Any]:
    """Generate trends analysis."""
    if not include_trends:
        return {"trends_included": False}

    return {
        "trends_included": True,
        "time_series": "Insufficient data for detailed trends",
        "patterns": "Analysis based on available log data",
        "forecasts": "Trend forecasting requires longer data history"
    }


def _generate_recommendations(result: Dict[str, Any], metrics: List[AnalysisData]) -> List[str]:
    """Generate actionable recommendations."""
    recommendations = []

    sources_count = result.get("sources_count", 0)
    confidence = result.get("confidence", 0.5)

    if sources_count == 0:
        recommendations.append("No issues found in the analyzed period - system appears stable")
        recommendations.append("Continue monitoring for any emerging patterns")
    elif sources_count > 10:
        recommendations.append("High activity detected - investigate for potential systemic issues")
        recommendations.append("Consider implementing additional monitoring for affected ATMs")
    else:
        recommendations.append("Normal activity levels detected")
        recommendations.append("Monitor identified issues for resolution")

    if confidence < 0.5:
        recommendations.append("Low confidence in analysis - consider expanding time range or checking data quality")

    recommendations.append("Regular performance monitoring recommended")

    return recommendations