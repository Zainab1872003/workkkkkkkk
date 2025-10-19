# routers/storage.py
"""
Storage routes matching OpenAPI spec
"""
import os
import json
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from core.config import settings
from agno_tools.document_tools import upload_document as upload_tool
from core.database import get_collection
from middleware.api_key_auth import get_api_key

router = APIRouter(prefix="/api/storage", tags=["Storage"])
logger = logging.getLogger(__name__)

# Global flag for upload processing
is_processing_upload = False


@router.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    ocr_type: str = Form("image"),
    files: Optional[str] = Form(None),
    webhook_url: Optional[str] = Form(None),
    api_key: str = Depends(get_api_key)
):
    """
    Upload files for processing and embedding creation.
    
    Args:
        user_id: User ID
        ocr_type: OCR processing type (image/document)
        files: Files to upload in JSON format
        webhook_url: Webhook URL for status updates
    
    Returns:
        Upload status message
    """
    global is_processing_upload
    
    if is_processing_upload:
        raise HTTPException(
            status_code=503,
            detail="Service is busy processing an upload"
        )
    
    try:
        is_processing_upload = True
        logger.info(f"📤 Upload started - User: {user_id}, OCR: {ocr_type}")
        
        # Parse files JSON
        if files:
            files_list = json.loads(files)
        else:
            files_list = []
        
        # Process files in background
        def process_files():
            global is_processing_upload
            try:
                for file_info in files_list:
                    # Your upload processing logic here
                    pass
                
                # Send webhook notification if provided
                if webhook_url:
                    # TODO: Send webhook notification
                    pass
                    
            finally:
                is_processing_upload = False
        
        background_tasks.add_task(process_files)
        
        return JSONResponse({
            "status": "Upload processing started",
            "message": "The file has been submitted to AI"
        })
        
    except Exception as e:
        is_processing_upload = False
        logger.error(f"❌ Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete")
async def delete_files(
    files_list: Optional[str] = Form(None),
    api_key: str = Depends(get_api_key)
):
    """
    Delete specific files from storage.
    
    Args:
        files_list: JSON array of filenames to delete
    
    Returns:
        Deletion status
    """
    try:
        logger.info(f"🗑️ Delete files request")
        
        if not files_list:
            raise HTTPException(status_code=400, detail="No files specified")
        
        files = json.loads(files_list)
        
        # Delete from MongoDB
        collection = get_collection()
        result = await collection.delete_many({
            "filename": {"$in": files}
        })
        
        logger.info(f"✅ Deleted {result.deleted_count} files")
        
        return JSONResponse({
            "response": "Files deleted successfully",
            "deleted_count": result.deleted_count
        })
        
    except Exception as e:
        logger.error(f"❌ Delete error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/list_documents")
async def list_documents(
    user_id: str = Form(...),
    api_key: str = Depends(get_api_key)
):
    """
    List all documents for a user.
    
    Args:
        user_id: User ID
    
    Returns:
        List of document filenames
    """
    try:
        logger.info(f"📋 List documents - User: {user_id}")
        
        collection = get_collection()
        cursor = collection.find({"user_id": user_id})
        documents = await cursor.to_list(length=1000)
        
        filenames = [doc.get("filename") for doc in documents]
        
        return JSONResponse({
            "response": filenames
        })
        
    except Exception as e:
        logger.error(f"❌ List error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
