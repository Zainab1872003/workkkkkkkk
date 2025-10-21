# routers/storage.py
"""
Storage routes matching OpenAPI specification
Uses DocumentStore and document loader directly (no orchestrator)
"""
import os
import logging
import hashlib
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pymongo.errors import DuplicateKeyError

from core.config import settings
from core.database import get_collection
from core.document_loader1 import load_and_process_document
from core.vectorstore import DocumentStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/storage", tags=["Storage"])

# Global flag to track upload processing status
is_processing_upload = False

# Toggle webhook functionality
ENABLE_WEBHOOKS = False


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_file_hash(content: bytes) -> str:
    """Calculate SHA-256 hash of file content for duplicate detection"""
    return hashlib.sha256(content).hexdigest()


# ============================================================================
# /api/storage/upload - Upload and Process Documents
# ============================================================================
@router.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    files: List[UploadFile] = File(...),
    ocr_type: str = Form(default="image"),
    webhook_url: Optional[str] = Form(default=None)
):
    """
    Upload and process files using DocumentStore directly.
    
    - **user_id**: User ID for document ownership
    - **files**: List of files to upload
    - **ocr_type**: OCR processing type (image/document)
    - **webhook_url**: Optional webhook for status updates (currently disabled)
    """
    global is_processing_upload
    
    # Check if service is busy
    if is_processing_upload:
        raise HTTPException(
            status_code=503,
            detail="Service is busy processing an upload. Please try again later."
        )
    
    try:
        is_processing_upload = True
        
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Create upload directory
        upload_dir = Path(settings.UPLOAD_DIR) / user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📤 Starting upload for user {user_id}: {len(files)} file(s)")
        
        # Read file contents before passing to background task
        file_data_list = []
        for file in files:
            # Validate file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in settings.ALLOWED_EXTENSIONS:
                logger.warning(f"⚠️ Skipping unsupported file: {file.filename}")
                continue
            
            # Read and validate file size
            file_content = await file.read()
            if len(file_content) > settings.MAX_UPLOAD_SIZE:
                logger.warning(f"⚠️ File too large: {file.filename}")
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} too large. Max: {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB"
                )
            
            file_data_list.append({
                "filename": file.filename,
                "content": file_content,
                "content_type": file.content_type,
                "file_hash": calculate_file_hash(file_content)
            })
            await file.close()
        
        if not file_data_list:
            is_processing_upload = False
            raise HTTPException(status_code=400, detail="No valid files to upload")
        
        # Process files in background
        background_tasks.add_task(
            process_upload_task,
            file_data_list=file_data_list,
            user_id=user_id,
            upload_dir=upload_dir,
            ocr_type=ocr_type
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "Upload processing started",
                "message": "The file has been submitted to AI",
                "user_id": user_id,
                "file_count": len(file_data_list)
            }
        )
        
    except HTTPException:
        is_processing_upload = False
        raise
    except Exception as e:
        is_processing_upload = False
        logger.error(f"❌ Upload error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def process_upload_task(
    file_data_list: List[dict],
    user_id: str,
    upload_dir: Path,
    ocr_type: str
):
    """Background task to process uploaded files using DocumentStore"""
    global is_processing_upload
    
    try:
        uploaded_files = []
        col = get_collection()
        
        # Create user-specific DocumentStore
        collection_name = f"user_{user_id}"
        store = DocumentStore(collection_name="rag_langchain")
        
        for file_data in file_data_list:
            try:
                filename = file_data["filename"]
                content = file_data["content"]
                file_hash = file_data["file_hash"]
                file_ext = Path(filename).suffix.lower()
                
                logger.info(f"📁 Processing: {filename}")
                logger.info(f"📝 File hash: {file_hash[:16]}...")
                
                # Check for existing document in MongoDB
                existing_doc = await col.find_one({"filename": filename, "user_id": user_id})
                
                is_duplicate = False
                is_updated = False
                
                if existing_doc:
                    existing_file_hash = existing_doc.get("file_hash")
                    
                    # Check if file is identical
                    if existing_file_hash == file_hash:
                        logger.info(f"✅ File '{filename}' is identical - skipping processing")
                        uploaded_files.append({
                            "filename": filename,
                            "status": "duplicate",
                            "message": f"File '{filename}' is identical. No changes detected."
                        })
                        continue
                    
                    # File is modified - delete existing version
                    logger.info(f"🔄 File '{filename}' is modified - updating...")
                    await delete_document_internal(user_id, filename, store)
                    is_updated = True
                
                # Save file to disk
                file_path = upload_dir / filename
                with open(file_path, "wb") as f:
                    f.write(content)
                
                file_size = file_path.stat().st_size
                logger.info(f"💾 Saved {filename} ({file_size} bytes)")
                
                # Load and chunk document
                logger.info(f"📄 Loading and chunking: {filename}")
                chunks = load_and_process_document(
                    file_path=str(file_path),
                    chunk_size=settings.CHUNK_SIZE,
                    chunk_overlap=settings.CHUNK_OVERLAP
                )
                
                if not chunks:
                    raise Exception(f"No content could be extracted from {filename}")
                
                logger.info(f"✂️ Created {len(chunks)} chunks from {filename}")
                
                # Store in vector database (Milvus)
                logger.info(f"🗄️ Storing in vector database: {filename}")
                vectorstore, document_id = store.store_documents(
                    docs=chunks,
                    filename=filename,
                    file_size=file_size
                )
                
                if not document_id:
                    raise Exception("Failed to generate document_id")
                
                # Save metadata to MongoDB
                doc_record = {
                    "user_id": user_id,
                    "filename": filename,
                    "file_type": file_ext.lstrip('.'),
                    "upload_date": datetime.utcnow().isoformat(),
                    "file_size": file_size,
                    "file_hash": file_hash,
                    "chunk_count": len(chunks),
                    "document_id": document_id,
                    "collection_name": collection_name,
                    "ocr_type": ocr_type
                }
                
                # Insert or update in MongoDB
                try:
                    await col.insert_one(doc_record)
                    logger.info(f"✅ Saved metadata to MongoDB for {filename}")
                except DuplicateKeyError:
                    await col.replace_one(
                        {"filename": filename, "user_id": user_id},
                        doc_record
                    )
                    logger.info(f"✅ Updated metadata in MongoDB for {filename}")
                
                # Clean up temporary file
                if file_path.exists():
                    file_path.unlink()
                
                uploaded_files.append({
                    "filename": filename,
                    "status": "updated" if is_updated else "success",
                    "message": f"Document '{filename}' {'updated' if is_updated else 'uploaded'} successfully",
                    "chunk_count": len(chunks),
                    "document_id": document_id
                })
                
                logger.info(f"✅ Successfully processed {filename}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {file_data['filename']}: {e}", exc_info=True)
                uploaded_files.append({
                    "filename": file_data['filename'],
                    "status": "error",
                    "message": str(e)
                })
                
                # Clean up on error
                file_path = upload_dir / file_data['filename']
                if file_path.exists():
                    file_path.unlink()
                continue
        
        logger.info(f"✅ Upload task complete: {len(uploaded_files)}/{len(file_data_list)} files processed")
        
    except Exception as e:
        logger.error(f"❌ Upload task error: {e}", exc_info=True)
    finally:
        is_processing_upload = False


# ============================================================================
# /api/storage/delete - Delete Files
# ============================================================================
@router.post("/delete")
async def delete_files(
    files_list: List[str] = Form(...),
    user_id: Optional[str] = Form(default=None)
):
    """
    Delete specific files from storage using DocumentStore.
    
    - **files_list**: List of filenames to delete
    - **user_id**: User ID (required)
    """
    try:
        if not files_list:
            raise HTTPException(status_code=400, detail="No files specified for deletion")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        collection_name = f"user_{user_id}"
        store = DocumentStore(collection_name="rag_langchain")
        
        deleted_results = []
        deleted_count = 0
        
        for filename in files_list:
            try:
                result = await delete_document_internal(user_id, filename, store)
                
                if result["success"]:
                    deleted_count += 1
                    deleted_results.append({
                        "filename": filename,
                        "status": "deleted",
                        "message": result["message"]
                    })
                    logger.info(f"🗑️ Deleted: {filename}")
                else:
                    deleted_results.append({
                        "filename": filename,
                        "status": "not_found",
                        "message": result["message"]
                    })
                    logger.warning(f"⚠️ Delete issue: {result['message']}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to delete {filename}: {e}")
                deleted_results.append({
                    "filename": filename,
                    "status": "error",
                    "message": str(e)
                })
                continue
        
        return JSONResponse(
            status_code=200,
            content={
                "response": "Files deleted successfully",
                "deleted_count": deleted_count,
                "requested_count": len(files_list),
                "details": deleted_results
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Delete error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


async def delete_document_internal(user_id: str, filename: str, store: DocumentStore) -> dict:
    """Internal function to delete document using DocumentStore"""
    try:
        col = get_collection()
        
        # Check if document exists in MongoDB
        existing_doc = await col.find_one({"filename": filename, "user_id": user_id})
        if not existing_doc:
            return {
                "message": f"Document '{filename}' not found for user {user_id}",
                "success": False
            }
        
        # Delete from MongoDB
        logger.info(f"🗑️ Deleting '{filename}' from MongoDB...")
        mongo_result = await col.delete_one({"filename": filename, "user_id": user_id})
        
        # Delete from vector store
        vector_success = store.delete_document_by_filename(filename)
        
        if mongo_result.deleted_count > 0:
            return {
                "message": f"Document '{filename}' deleted successfully",
                "success": True
            }
        else:
            return {
                "message": f"Failed to delete '{filename}' from MongoDB",
                "success": False
            }
            
    except Exception as e:
        logger.error(f"❌ Error deleting document {filename}: {e}")
        return {
            "message": f"Error deleting document: {str(e)}",
            "success": False
        }


# ============================================================================
# /api/storage/list_documents - List User Documents
# ============================================================================
@router.post("/list_documents")
async def list_user_documents(
    user_id: str = Form(...)
):
    """
    List all documents for a specific user.
    
    - **user_id**: User ID to list documents for
    """
    try:
        col = get_collection()
        
        # Query user documents from MongoDB
        cursor = col.find({"user_id": user_id}).sort("upload_date", -1)
        documents = await cursor.to_list(length=1000)
        
        if not documents:
            return JSONResponse(
                status_code=200,
                content={
                    "response": [],
                    "message": f"No documents found for user {user_id}",
                    "count": 0,
                    "user_id": user_id
                }
            )
        
        # Format document list
        doc_list = []
        for doc in documents:
            doc_info = {
                "filename": doc.get("filename"),
                "document_id": doc.get("document_id"),
                "file_size": doc.get("file_size"),
                "file_type": doc.get("file_type"),
                "chunk_count": doc.get("chunk_count"),
                "upload_date": doc.get("upload_date"),
                "file_hash": doc.get("file_hash", "")[:16] + "..." if doc.get("file_hash") else "",
                "collection_name": doc.get("collection_name")
            }
            doc_list.append(doc_info)
        
        logger.info(f"📋 Listed {len(doc_list)} documents for user {user_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "response": doc_list,
                "count": len(doc_list),
                "user_id": user_id
            }
        )
        
    except Exception as e:
        logger.error(f"❌ List error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"List failed: {str(e)}")


# ============================================================================
# Additional Endpoints
# ============================================================================
@router.get("/status")
async def get_upload_status():
    """Get current upload processing status"""
    try:
        col = get_collection()
        
        # Get document counts
        total_docs = await col.count_documents({})
        
        return JSONResponse(
            status_code=200,
            content={
                "is_processing": is_processing_upload,
                "total_documents": total_docs,
                "service_status": "✅ Active"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to get status: {str(e)}",
                "is_processing": is_processing_upload
            }
        )


# ============================================================================
# Readiness Check
# ============================================================================
async def check_readiness():
    """Check if service is ready to accept uploads"""
    global is_processing_upload
    
    if is_processing_upload:
        return JSONResponse(
            status_code=503,
            content={"detail": "Service is busy processing an upload"}
        )
    
    return JSONResponse(
        status_code=200,
        content={"status": "Service is ready"}
    )
