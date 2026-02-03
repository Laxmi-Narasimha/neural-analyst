"""FastAPI server - Main application with REST endpoints."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid
import asyncio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from .orchestrator import OrchestratorAgent
from .config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Data Adequacy Agent",
    description="Comprehensive data validation system for AI assistants",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state management
active_sessions: Dict[str, OrchestratorAgent] = {}
upload_directory = "uploads"
os.makedirs(upload_directory, exist_ok=True)


# Pydantic models
class ValidationRequest(BaseModel):
    goal: str
    domain: str = "general"
    files: Optional[List[str]] = None
    namespace: Optional[str] = None


class ContinueValidationRequest(BaseModel):
    session_id: str
    answers: Dict[str, str]


class QueryRequest(BaseModel):
    namespace: str
    query: str
    top_k: int = 5


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test connections
        orchestrator = OrchestratorAgent()
        connection_status = await orchestrator.test_connections()
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": "2024-08-22T16:00:00Z",
            "services": connection_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")


@app.post("/api/validate")
async def start_validation(
    goal: str = Form(...),
    domain: str = Form("general"),
    files: List[UploadFile] = File(None)
):
    """
    Start a new validation session.
    
    Args:
        goal: User's AI assistant goal description
        domain: Domain context (automotive, manufacturing, real_estate, general)
        files: Optional uploaded files
        
    Returns:
        Session information and clarifying questions
    """
    try:
        # Save uploaded files
        uploaded_file_paths = []
        if files:
            for file in files:
                if file.filename:
                    file_path = os.path.join(upload_directory, f"{uuid.uuid4().hex}_{file.filename}")
                    with open(file_path, "wb") as f:
                        content = await file.read()
                        f.write(content)
                    uploaded_file_paths.append(file_path)
                    logger.info(f"Saved uploaded file: {file.filename} -> {file_path}")
        
        # Initialize orchestrator
        orchestrator = OrchestratorAgent()
        
        # Start validation
        result = await orchestrator.run_validation(
            user_goal=goal,
            domain=domain,
            files=uploaded_file_paths
        )
        
        if result["success"]:
            # Store orchestrator in active sessions
            session_id = result["session_id"]
            active_sessions[session_id] = orchestrator
            
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Validation failed"))
            
    except Exception as e:
        logger.error(f"Validation start failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/validate/continue")
async def continue_validation(request: ContinueValidationRequest):
    """
    Continue validation with user answers to clarifying questions.
    
    Args:
        request: Session ID and user answers
        
    Returns:
        Complete validation results
    """
    try:
        session_id = request.session_id
        
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        orchestrator = active_sessions[session_id]
        
        # Continue validation
        result = await orchestrator.continue_validation(request.answers)
        
        if result["success"]:
            # Clean up session after completion
            if result.get("readiness_level"):
                del active_sessions[session_id]
            
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Validation continuation failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation continuation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get current session status."""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        orchestrator = active_sessions[session_id]
        status = orchestrator.get_session_status()
        
        return JSONResponse(content=status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query_knowledge_base(request: QueryRequest):
    """
    Query a knowledge base namespace.
    
    Args:
        request: Query parameters
        
    Returns:
        Query results with relevant chunks
    """
    try:
        orchestrator = OrchestratorAgent()
        vector_manager = orchestrator.ingestion_agent.vector_manager
        
        result = await vector_manager.query_knowledge_base(
            request.query,
            request.namespace,
            request.top_k
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports/{report_id}")
async def download_report(report_id: str, format: str = "markdown"):
    """
    Download a generated report.
    
    Args:
        report_id: Report identifier
        format: Report format (markdown or json)
        
    Returns:
        Report file
    """
    try:
        reports_dir = "reports"
        
        if format == "markdown":
            filename = f"report_{report_id}.md"
            media_type = "text/markdown"
        elif format == "json":
            filename = f"report_{report_id}.json"
            media_type = "application/json"
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'markdown' or 'json'")
        
        file_path = os.path.join(reports_dir, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Report not found")
        
        return FileResponse(
            file_path,
            media_type=media_type,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config")
async def get_configuration():
    """Get system configuration (non-sensitive parts)."""
    return {
        "supported_domains": list(config.DOMAIN_CONFIGS.keys()),
        "supported_file_types": [".pdf", ".docx", ".txt", ".csv", ".xlsx"],
        "max_file_size_mb": config.MAX_FILE_SIZE_MB,
        "quality_thresholds": config.QUALITY_THRESHOLDS,
        "scoring_weights": config.SCORING_WEIGHTS
    }


@app.delete("/api/session/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up a session and its resources."""
    try:
        if session_id in active_sessions:
            # Clean up Pinecone namespace if needed
            orchestrator = active_sessions[session_id]
            namespace = orchestrator.session_state.get("namespace")
            
            if namespace and namespace.startswith("session_"):
                vector_manager = orchestrator.ingestion_agent.vector_manager
                cleanup_success = vector_manager.cleanup_namespace(namespace)
                logger.info(f"Namespace cleanup for {namespace}: {cleanup_success}")
            
            del active_sessions[session_id]
        
        return {"success": True, "message": "Session cleaned up"}
        
    except Exception as e:
        logger.error(f"Session cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions")
async def list_active_sessions():
    """List all active sessions."""
    sessions = []
    
    for session_id, orchestrator in active_sessions.items():
        status = orchestrator.get_session_status()
        sessions.append(status)
    
    return {"active_sessions": sessions, "total_count": len(sessions)}


# Background task for periodic cleanup
async def cleanup_old_sessions():
    """Background task to clean up old sessions."""
    while True:
        try:
            # This would implement logic to clean up sessions older than X hours
            # For now, just a placeholder
            await asyncio.sleep(3600)  # Check every hour
        except Exception as e:
            logger.error(f"Session cleanup task failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting AI Data Adequacy Agent server...")
    
    # Validate configuration
    try:
        config.validate_config()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise
    
    # Test connections
    try:
        orchestrator = OrchestratorAgent()
        connection_status = await orchestrator.test_connections()
        
        openai_status = connection_status.get("openai", {}).get("status")
        pinecone_status = connection_status.get("pinecone", {}).get("status")
        
        if openai_status != "success":
            logger.warning(f"OpenAI connection issue: {connection_status.get('openai')}")
        
        if pinecone_status != "success":
            logger.warning(f"Pinecone connection issue: {connection_status.get('pinecone')}")
        
        logger.info("External service connections tested")
        
    except Exception as e:
        logger.error(f"Service connection test failed: {str(e)}")
    
    # Start background tasks
    asyncio.create_task(cleanup_old_sessions())
    
    logger.info("Server startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on application shutdown."""
    logger.info("Shutting down AI Data Adequacy Agent server...")
    
    # Clean up active sessions
    for session_id in list(active_sessions.keys()):
        try:
            await cleanup_session(session_id)
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {str(e)}")
    
    logger.info("Server shutdown completed")


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )
