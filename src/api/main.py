"""
ATM RAG API Main Application

FastAPI application providing REST endpoints for the ATM RAG system.
"""

import os
import time
import logging
from typing import Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .models import (
    QueryRequest, QueryResponse, SearchRequest, SearchResponse,
    TroubleshootRequest, TroubleshootResponse, AnalyzeRequest, AnalyzeResponse,
    HealthCheckResponse, StatsResponse, LogUploadRequest, LogUploadResponse,
    ErrorResponse, BatchQueryRequest, BatchQueryResponse
)
from .dependencies import get_rag_pipeline, get_query_processor, get_vector_store
from .routers import queries, search, management, troubleshooting

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    logger.info("Starting ATM RAG API server...")
    app.state.start_time = time.time()
    app.state.query_count = 0
    app.state.total_response_time = 0.0

    try:
        # Initialize core components
        logger.info("Initializing core components...")
        # Components will be initialized on first request via dependencies
        logger.info("ATM RAG API server started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down ATM RAG API server...")


# Create FastAPI application
app = FastAPI(
    title="ATM RAG API",
    description="Intelligent ATM operations assistance system with RAG-powered query processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(queries.router, prefix="/api/v1", tags=["queries"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])
app.include_router(troubleshooting.router, prefix="/api/v1", tags=["troubleshooting"])
app.include_router(management.router, prefix="/api/v1", tags=["management"])

# Mount static files for web interface
try:
    app.mount("/static", StaticFiles(directory="src/api/static"), name="static")
except Exception:
    logger.warning("Static files directory not found - web interface disabled")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic information."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ATM RAG API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            .endpoint { background: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; font-family: monospace; }
            .status { background: #2ecc71; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .feature { margin: 10px 0; padding: 10px; background: #e8f6f3; border-left: 4px solid #1abc9c; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏧 ATM RAG API</h1>
            <p><span class="status">ONLINE</span> - Intelligent ATM Operations Assistance System</p>

            <h2>🎯 Features</h2>
            <div class="feature">💬 <strong>Natural Language Queries:</strong> Ask questions about ATM operations in plain English</div>
            <div class="feature">🔍 <strong>Log Search:</strong> Find relevant ATM logs with semantic search</div>
            <div class="feature">🛠️ <strong>Troubleshooting:</strong> Get specific guidance for ATM error codes and issues</div>
            <div class="feature">📊 <strong>Analytics:</strong> Performance analysis and trend identification</div>

            <h2>📋 API Endpoints</h2>
            <div class="endpoint">POST /api/v1/query - Process natural language queries</div>
            <div class="endpoint">POST /api/v1/search - Search ATM logs</div>
            <div class="endpoint">POST /api/v1/troubleshoot - Get troubleshooting guidance</div>
            <div class="endpoint">POST /api/v1/analyze - Perform analysis</div>
            <div class="endpoint">GET /api/v1/health - System health check</div>
            <div class="endpoint">GET /api/v1/stats - System statistics</div>

            <h2>📚 Documentation</h2>
            <p>
                <a href="/docs">🔗 Interactive API Documentation (Swagger UI)</a><br>
                <a href="/redoc">🔗 Alternative Documentation (ReDoc)</a>
            </p>

            <h2>💡 Example Queries</h2>
            <div class="endpoint">"Why is ATM001 showing DDL_EXCEEDED errors?"</div>
            <div class="endpoint">"What does NETWORK_ERROR mean?"</div>
            <div class="endpoint">"Analyze withdrawal failures in the last 24 hours"</div>
            <div class="endpoint">"Show me recent timeout issues"</div>

            <p style="margin-top: 30px; text-align: center; color: #7f8c8d;">
                ATM RAG System v1.0.0 - Built with FastAPI & sentence-transformers
            </p>
        </div>
    </body>
    </html>
    """
    return html_content


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Basic health check endpoint."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "components": {
                "api": {"status": "healthy", "details": "API server running"},
                "database": {"status": "unknown", "details": "Not checked in basic health"},
                "embeddings": {"status": "unknown", "details": "Not checked in basic health"}
            },
            "uptime_seconds": time.time() - app.state.start_time,
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "NotFound",
            "message": "The requested resource was not found",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An internal server error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app


if __name__ == "__main__":
    # Development server
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )