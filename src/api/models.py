"""
Pydantic Models for ATM RAG API

Defines request and response models for all API endpoints.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class ResponseType(str, Enum):
    """Response type enumeration."""
    AUTO = "auto"
    TROUBLESHOOTING = "troubleshooting"
    ERROR = "error"
    ANALYSIS = "analysis"
    INFO = "info"


class QueryIntent(str, Enum):
    """Query intent enumeration."""
    TROUBLESHOOTING = "troubleshooting"
    ERROR_EXPLANATION = "error"
    ANALYSIS = "analysis"
    SEARCH = "search"
    INFO = "info"
    UNKNOWN = "unknown"


# Request Models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., min_length=1, max_length=1000, description="User's query about ATM operations")
    response_type: ResponseType = Field(ResponseType.AUTO, description="Desired response type")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of similar logs to retrieve")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters for log search")
    include_metadata: bool = Field(True, description="Include metadata in response")

    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty or only whitespace."""
        if not v.strip():
            raise ValueError('Query cannot be empty or only whitespace')
        return v.strip()


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    operation_type: Optional[str] = Field(None, description="Filter by operation type")
    status: Optional[str] = Field(None, description="Filter by status")
    atm_id: Optional[str] = Field(None, description="Filter by ATM ID")
    error_code: Optional[str] = Field(None, description="Filter by error code")
    start_date: Optional[datetime] = Field(None, description="Start date for time range filter")
    end_date: Optional[datetime] = Field(None, description="End date for time range filter")


class TroubleshootRequest(BaseModel):
    """Request model for troubleshooting endpoint."""
    error_code: Optional[str] = Field(None, description="Specific error code to troubleshoot")
    atm_id: Optional[str] = Field(None, description="Specific ATM ID")
    operation: Optional[str] = Field(None, description="Operation type")
    description: Optional[str] = Field(None, description="Problem description")
    recent_only: bool = Field(True, description="Focus on recent issues")


class AnalyzeRequest(BaseModel):
    """Request model for analysis endpoint."""
    analysis_type: str = Field("performance", description="Type of analysis to perform")
    time_range_hours: int = Field(24, ge=1, le=168, description="Time range in hours")
    filters: Optional[Dict[str, Any]] = Field(None, description="Analysis filters")
    include_trends: bool = Field(True, description="Include trend analysis")


class LogUploadRequest(BaseModel):
    """Request model for log upload."""
    logs: List[Dict[str, Any]] = Field(..., description="ATM log entries to upload")
    validate_logs: bool = Field(True, description="Validate logs before processing")
    process_immediately: bool = Field(True, description="Process logs immediately")


# Response Models
class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    response: str = Field(..., description="Generated response text")
    response_type: str = Field(..., description="Type of response generated")
    query: str = Field(..., description="Original query")
    confidence: float = Field(..., ge=0, le=1, description="Response confidence score")
    sources_count: int = Field(..., ge=0, description="Number of sources used")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional response metadata")
    generated_at: datetime = Field(..., description="Response generation timestamp")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class LogEntry(BaseModel):
    """Model for a single log entry."""
    log_id: str = Field(..., description="Unique log identifier")
    timestamp: datetime = Field(..., description="Log timestamp")
    operation: str = Field(..., description="ATM operation")
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Log message")
    atm_id: Optional[str] = Field(None, description="ATM identifier")
    error_code: Optional[str] = Field(None, description="Error code if applicable")
    amount: Optional[float] = Field(None, description="Transaction amount")
    similarity_score: Optional[float] = Field(None, description="Similarity score for search results")


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    results: List[LogEntry] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of matching logs")
    query: str = Field(..., description="Original search query")
    filters_applied: Dict[str, Any] = Field(..., description="Filters that were applied")
    processing_time_ms: float = Field(..., description="Search processing time")


class TroubleshootResponse(BaseModel):
    """Response model for troubleshooting endpoint."""
    troubleshooting_steps: List[str] = Field(..., description="Recommended troubleshooting steps")
    similar_cases: List[LogEntry] = Field(..., description="Similar cases found")
    root_cause_analysis: str = Field(..., description="Potential root cause analysis")
    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence")
    estimated_resolution_time: Optional[str] = Field(None, description="Estimated time to resolve")


class AnalysisData(BaseModel):
    """Analysis data structure."""
    metric_name: str = Field(..., description="Name of the metric")
    current_value: Union[int, float, str] = Field(..., description="Current metric value")
    trend: Optional[str] = Field(None, description="Trend direction")
    change_percentage: Optional[float] = Field(None, description="Percentage change")


class AnalyzeResponse(BaseModel):
    """Response model for analysis endpoint."""
    analysis_summary: str = Field(..., description="Summary of analysis results")
    key_metrics: List[AnalysisData] = Field(..., description="Key performance metrics")
    trends: Dict[str, Any] = Field(..., description="Trend analysis data")
    recommendations: List[str] = Field(..., description="Recommendations based on analysis")
    time_range: str = Field(..., description="Time range analyzed")
    analysis_confidence: float = Field(..., ge=0, le=1, description="Analysis confidence score")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component health status")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="API version")


class SystemStats(BaseModel):
    """System statistics model."""
    total_logs: int = Field(..., description="Total number of logs")
    total_embeddings: int = Field(..., description="Total number of embeddings")
    recent_logs_24h: int = Field(..., description="Logs added in last 24 hours")
    query_count_24h: int = Field(..., description="Queries processed in last 24 hours")
    average_response_time_ms: float = Field(..., description="Average response time")
    storage_size_mb: float = Field(..., description="Storage size in MB")


class StatsResponse(BaseModel):
    """Response model for system statistics."""
    statistics: SystemStats = Field(..., description="System statistics")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    resource_usage: Dict[str, Any] = Field(..., description="Resource usage information")
    timestamp: datetime = Field(..., description="Statistics generation timestamp")


class LogUploadResponse(BaseModel):
    """Response model for log upload."""
    uploaded_count: int = Field(..., description="Number of logs uploaded")
    processed_count: int = Field(..., description="Number of logs processed")
    failed_count: int = Field(..., description="Number of logs that failed")
    processing_time_ms: float = Field(..., description="Total processing time")
    errors: List[str] = Field(default_factory=list, description="Error messages if any")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


# Validation Models
class QueryValidation(BaseModel):
    """Query validation result."""
    is_valid: bool = Field(..., description="Whether query is valid")
    is_atm_related: bool = Field(..., description="Whether query is ATM-related")
    detected_intent: QueryIntent = Field(..., description="Detected query intent")
    confidence: float = Field(..., ge=0, le=1, description="Validation confidence")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class BatchQueryRequest(BaseModel):
    """Request model for batch query processing."""
    queries: List[str] = Field(..., max_items=10, description="List of queries to process")
    response_type: ResponseType = Field(ResponseType.AUTO, description="Response type for all queries")
    top_k: int = Field(5, ge=1, le=20, description="Number of results per query")


class BatchQueryResponse(BaseModel):
    """Response model for batch query processing."""
    responses: List[QueryResponse] = Field(..., description="Responses for each query")
    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    total_processing_time_ms: float = Field(..., description="Total processing time")