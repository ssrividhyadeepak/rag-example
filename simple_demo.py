#!/usr/bin/env python3
"""
Simple ATM RAG Demo - Working Version

A minimal demonstration of the ATM RAG system that works without MongoDB
and provides instant responses to test the React UI integration.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime
from typing import Optional, List, Dict, Any
import time
import json

app = FastAPI(
    title="ATM RAG Demo API",
    description="Simple working version for testing the React UI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    filters: Optional[Dict] = None

class QueryResponse(BaseModel):
    response: str
    response_type: str
    query: str
    confidence: float
    sources_count: int
    metadata: Dict[str, Any]
    generated_at: str
    processing_time_ms: float

# Mock ATM data for demo responses
MOCK_ATM_RESPONSES = {
    "ddl_exceeded": {
        "response": """🏧 **ATM ERROR ANALYSIS**

**Error Code:** DDL_EXCEEDED
**Meaning:** Daily Dollar Limit Exceeded

**What happened:**
ATM 1123 failed because the customer exceeded their daily withdrawal limit. This is a common protective measure.

**Similar Recent Issues:**
• ATM 1123: 3 DDL_EXCEEDED errors today at 10:00-11:00 AM
• ATM 1124: 2 similar errors yesterday
• ATM 1125: 1 error this morning

**Resolution:**
✅ Customer should wait until next business day
✅ Or contact bank to increase daily limit
✅ ATM is functioning normally

**Prevention:** Customers should monitor daily withdrawal amounts.""",
        "confidence": 0.92,
        "sources": 3
    },
    "timeout": {
        "response": """🏧 **ATM TIMEOUT ANALYSIS**

**Issue:** Network timeout detected on ATM 1123 at 10:00 AM

**Root Cause Analysis:**
• Network connectivity interrupted
• Response time exceeded 30 seconds
• Transaction automatically cancelled for security

**Current Status:** ✅ Resolved
**Impact:** 1 customer affected

**Recommended Actions:**
1. Monitor network stability
2. Check router/switch connectivity
3. Verify ISP connection quality""",
        "confidence": 0.88,
        "sources": 2
    },
    "network_error": {
        "response": """🏧 **NETWORK ERROR EXPLANATION**

**Error Code:** NETWORK_ERROR
**Common Causes:**
• Internet connection lost
• Router/switch failure
• ISP service interruption
• Firewall blocking ATM traffic

**Troubleshooting Steps:**
1. Check physical network cables
2. Restart network equipment
3. Test internet connectivity
4. Contact ISP if needed

**Recent Pattern:** 3 network errors across ATMs 1120-1125 today""",
        "confidence": 0.85,
        "sources": 4
    }
}

def get_smart_response(query: str) -> Dict[str, Any]:
    """Generate intelligent responses based on query content."""
    query_lower = query.lower()

    # Detect query type and generate appropriate response
    if any(term in query_lower for term in ["ddl", "daily", "limit", "1123", "10 am", "10:00"]):
        return MOCK_ATM_RESPONSES["ddl_exceeded"]
    elif any(term in query_lower for term in ["timeout", "network", "connection"]):
        return MOCK_ATM_RESPONSES["timeout"]
    elif "network_error" in query_lower:
        return MOCK_ATM_RESPONSES["network_error"]
    elif any(term in query_lower for term in ["error", "code", "mean", "analysis"]):
        return {
            "response": f"""🏧 **ATM ERROR ASSISTANCE**

I can help explain ATM errors and issues!

**For your query:** "{query}"

**Common ATM Error Codes:**
• **DDL_EXCEEDED** - Daily dollar limit exceeded
• **NETWORK_ERROR** - Connectivity issues
• **TIMEOUT_ERROR** - Transaction timeout
• **CASH_JAM** - Mechanical dispenser issue
• **CARD_RETAINED** - Security card capture

**Need specific help?** Try asking:
• "What does DDL_EXCEEDED mean?"
• "Why did ATM 1123 fail today?"
• "Show me network errors"

I can analyze specific ATM IDs, error codes, and time ranges!""",
            "confidence": 0.75,
            "sources": 1
        }
    else:
        return {
            "response": f"""🏧 **ATM SUPPORT ASSISTANT**

I can help with ATM operations, troubleshooting, and analysis!

**Your query:** "{query}"

**I can help with:**
• 🔍 **Error Analysis** - Explain error codes and causes
• 📊 **Performance Reports** - ATM status and trends
• 🛠️ **Troubleshooting** - Step-by-step problem solving
• 📈 **Pattern Detection** - Identify recurring issues

**Try asking:**
• "Why did ATM 1123 fail today at 10 AM?"
• "What does DDL_EXCEEDED error mean?"
• "Show me recent timeout issues"
• "Analyze withdrawal patterns"

How can I help you today?""",
            "confidence": 0.65,
            "sources": 0
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": {"status": "healthy", "details": "Demo API running"},
            "database": {"status": "demo", "details": "Using mock data"},
            "embeddings": {"status": "demo", "details": "Mock responses"},
            "rag_engine": {"status": "demo", "details": "Simplified RAG"}
        },
        "version": "demo-1.0.0"
    }

@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process queries with intelligent responses."""
    start_time = time.time()

    try:
        # Get intelligent response
        smart_response = get_smart_response(request.query)

        # Simulate processing time
        await asyncio.sleep(0.1)

        processing_time = (time.time() - start_time) * 1000

        return QueryResponse(
            response=smart_response["response"],
            response_type="troubleshooting" if smart_response["confidence"] > 0.8 else "info",
            query=request.query,
            confidence=smart_response["confidence"],
            sources_count=smart_response["sources"],
            metadata={
                "response_type": "intelligent_demo",
                "sources_used": smart_response["sources"],
                "demo_mode": True,
                "atm_system": "functional"
            },
            generated_at=datetime.utcnow().isoformat(),
            processing_time_ms=processing_time
        )

    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        return QueryResponse(
            response=f"I encountered an error while processing your query: {str(e)}. Please try again or contact support.",
            response_type="error",
            query=request.query,
            confidence=0.0,
            sources_count=0,
            metadata={"error": str(e)},
            generated_at=datetime.utcnow().isoformat(),
            processing_time_ms=processing_time
        )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "ATM RAG Demo API",
        "status": "running",
        "endpoints": ["/api/v1/query", "/health"],
        "demo": True
    }

if __name__ == "__main__":
    import asyncio
    print("🚀 Starting ATM RAG Demo API on http://localhost:8001")
    print("📝 This is a working demo for testing the React UI")
    print("🔗 React UI should connect to: http://localhost:8001")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")