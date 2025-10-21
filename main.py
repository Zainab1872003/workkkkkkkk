# main.py
"""
Multimodel Agent API - Main FastAPI Application
Version: 1.0.0

Built with:
- FastAPI for API framework
- AGNO for AI agents with MCP tool integration
- Groq for LLM (free open-source models)
- MongoDB for document metadata
- Milvus (Zilliz Cloud) for vector storage
"""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.database import init_indexes, close_mongo_connection

# Import routers
from routers.health import router as health_router
from routers.chat import router as chat_router
from routers.storage import router as storage_router
from routers.operations import router as operations_router
from routers.usecases import router as usecases_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Application Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown events
    """
    # === STARTUP ===
    logger.info("="*70)
    logger.info(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info("="*70)
    
    # 1. Check Groq API key
    if settings.GROQ_API_KEY:
        logger.info("✅ Groq API key configured")
        logger.info(f"📊 Default model: {settings.GROQ_MODEL}")
        logger.info(f"⚡ Fast model: {settings.GROQ_FAST_MODEL}")
    else:
        logger.warning("⚠️  Groq API key not configured!")
        logger.warning("   Get free key from: https://console.groq.com/keys")
    
    # 2. Check Milvus/Zilliz Cloud connection
    if settings.MILVUS_URI and settings.MILVUS_PASSWORD:
        logger.info("✅ Milvus/Zilliz Cloud credentials configured")
        logger.info(f"🗄️  Collection: {settings.MILVUS_COLLECTION_NAME}")
    else:
        logger.warning("⚠️  Milvus credentials not configured!")
        logger.warning("   Get free Zilliz Cloud account from: https://cloud.zilliz.com/")
    
    # 3. Initialize MongoDB
    try:
        await init_indexes()
        logger.info("✅ MongoDB connected and indexed")
        logger.info(f"📁 Database: {settings.DATABASE_NAME}")
    except Exception as e:
        logger.error(f"❌ MongoDB initialization failed: {e}")
    
    # 4. Create upload directory
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    logger.info(f"✅ Upload directory ready: {settings.UPLOAD_DIR}")
    
    # 5. Check API keys
    api_keys_count = len([k for k in settings.API_KEYS.split(",") if k.strip()])
    logger.info(f"🔑 API keys configured: {api_keys_count}")
    
    logger.info("="*70)
    logger.info("🎉 Application startup complete!")
    logger.info("")
    logger.info("📚 Available Endpoints:")
    logger.info("   • Health Check: GET /")
    logger.info("   • Health Check: GET /health-check")
    logger.info("   • Readiness: GET /readiness")
    logger.info("   • Chat: POST /api/stream/chat")
    logger.info("")
    logger.info("📝 API Documentation:")
    logger.info("   • Swagger UI: http://localhost:8000/docs")
    logger.info("   • ReDoc: http://localhost:8000/redoc")
    logger.info("")
    logger.info("🔧 MCP Servers:")
    logger.info("   • Document MCP: mcp_servers/document_mcp.py")
    logger.info("   • Ijarah MCP: mcp_servers/ijarah_mcp.py")
    logger.info("   • Usecase MCP: mcp_servers/usecase_mcp.py")
    logger.info("="*70)
    
    yield
    
    # === SHUTDOWN ===
    logger.info("")
    logger.info("="*70)
    logger.info("🛑 Shutting down application...")
    logger.info("="*70)
    
    # Close MongoDB connection
    await close_mongo_connection()
    
    logger.info("✅ Cleanup complete")
    logger.info("👋 Goodbye!")
    logger.info("="*70)

# ============================================================================
# FastAPI Application Instance
# ============================================================================

app = FastAPI(
    title="Multimodel Agent API",
    description=(
        "A comprehensive AI agent platform providing chat capabilities, "
        "document management, usecase orchestration, feedback processing, "
        "and prompt management services.\n\n"
        "## Features\n"
        "- 🤖 AGNO AI agents with Groq LLM (free open-source models)\n"
        "- 📄 Document processing and RAG (Retrieval Augmented Generation)\n"
        "- 🗄️ Milvus/Zilliz Cloud vector database for semantic search\n"
        "- 🔧 MCP (Model Context Protocol) servers for tool integration\n"
        "- 💰 Islamic banking (Ijarah) calculations\n"
        "- 📊 Usecase management and orchestration\n\n"
        "## Authentication Required\n"
        "**All API endpoints require authentication via API key.**\n\n"
        "Add the following header to all requests:\n"
        "```http\n"
        "x-api-key: YOUR_API_KEY_HERE\n"
        "```\n\n"
        "Contact your system administrator to obtain the API key.\n\n"
        "## Available Tools (via MCP servers)\n"
        "- `document_retriever`: Search and retrieve information from uploaded documents\n"
        "- `calculate_bike_ijarah`: Calculate Islamic bike financing installments\n"
        "- `image_generation`: Generate images from text prompts (coming soon)\n\n"
        "## Architecture\n"
        "- **Backend**: FastAPI with async/await\n"
        "- **AI Framework**: AGNO with MCP tool integration\n"
        "- **LLM**: Groq (llama-3.3-70b-versatile)\n"
        "- **Vector DB**: Milvus/Zilliz Cloud\n"
        "- **Document DB**: MongoDB\n"
        "- **Embeddings**: sentence-transformers (local)\n"
    ),
    version="1.0.0",
    contact={
        "name": "AI Team"
    },
    license_info={
        "name": "Proprietary"
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        }
    ]
)

# ============================================================================
# CORS Middleware
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Include Routers
# ============================================================================

# Health check endpoints (no authentication required)
app.include_router(
    health_router,
    tags=["Health"]
)

# Chat endpoint (requires authentication)
app.include_router(
    chat_router,
    prefix="/api",
    tags=["Chat"]
)

# Future routers (to be added):
app.include_router(storage_router, prefix="/api", tags=["Storage"])
app.include_router(operations_router, prefix="/api", tags=["Operations"])
app.include_router(
    usecases_router,
    prefix="/api",
    tags=["Usecases"]
)

# ============================================================================
# Root Endpoint Override (for custom branding)
# ============================================================================

@app.get("/", include_in_schema=False)
async def custom_root():
    """Custom root endpoint with branding"""
    return {
        "message": "Multimodel Agent API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health-check"
    }
# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("="*70)
    logger.info("Starting Uvicorn development server...")
    logger.info("="*70)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
    
