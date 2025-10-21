# routers/usecases.py
"""
Usecase management routes
Implements /api/operations/usecases endpoints from OpenAPI spec
"""
import logging
import json
from datetime import datetime

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse

from core.database import get_or_create_collection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/operations/usecases", tags=["Usecases"])


# ============================================================================
# /api/operations/usecases/store-usecase - Store Usecase
# ============================================================================
@router.post("/store-usecase")
async def store_usecase(
    usecase: str = Form(..., description="Usecase details in JSON format")
):
    """
    Store a new usecase configuration.
    
    - **usecase**: JSON string containing usecase details
    
    Required fields in JSON:
    - id: Unique usecase identifier
    - name: Usecase name
    - description: Usecase description
    - systemPrompt: System prompt for the usecase
    
    Example:
    ```
    {
        "id": "usecase1",
        "name": "Banking Assistant",
        "description": "AI banking assistant",
        "systemPrompt": "You are a helpful banking assistant"
    }
    ```
    """
    try:
        logger.info("="*70)
        logger.info("📝 STORE USECASE REQUEST")
        logger.info("="*70)
        
        # Parse JSON
        try:
            usecase_data = json.loads(usecase)
            logger.info(f"✅ Parsed JSON successfully")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON format: {str(e)}"
            )
        
        # Validate required fields
        required_fields = ["id", "name", "description", "systemPrompt"]
        missing_fields = [field for field in required_fields if field not in usecase_data]
        
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {', '.join(missing_fields)}"
            )
        
        usecase_id = usecase_data["id"]
        usecase_name = usecase_data["name"]
        usecase_description = usecase_data["description"]
        system_prompt = usecase_data["systemPrompt"]
        
        logger.info(f"   ID:   {usecase_id}")
        logger.info(f"   Name: {usecase_name}")
        
        # Get or create 'usecases' collection
        usecases_indexes = [
            {"keys": [("id", 1)], "unique": True, "name": "id_unique"}
        ]
        
        col = await get_or_create_collection("usecases", usecases_indexes)
        
        # Check for duplicates
        existing = await col.find_one({"id": usecase_id})
        
        if existing:
            logger.warning(f"⚠️ Usecase '{usecase_id}' already exists")
            raise HTTPException(
                status_code=409,
                detail=f"Usecase with id '{usecase_id}' already exists"
            )
        
        # Prepare document (only store what's in the spec)
        usecase_document = {
            "id": usecase_id,
            "name": usecase_name,
            "description": usecase_description,
            "systemPrompt": system_prompt,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Insert into MongoDB
        result = await col.insert_one(usecase_document)
        logger.info(f"✅ Stored usecase with _id: {result.inserted_id}")
        logger.info("="*70)
        
        return JSONResponse(
            status_code=200,
            content={
                "response": "Usecase stored successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Store usecase error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Additional endpoints for usecase management
# ============================================================================

@router.get("/get-usecase/{usecase_id}")
async def get_usecase(usecase_id: str):
    """Get a usecase by ID"""
    try:
        col = await get_or_create_collection("usecases")
        usecase = await col.find_one({"id": usecase_id})
        
        if not usecase:
            raise HTTPException(
                status_code=404,
                detail=f"Usecase '{usecase_id}' not found"
            )
        
        # Remove MongoDB _id
        usecase.pop("_id", None)
        
        return JSONResponse(
            status_code=200,
            content={
                "response": usecase
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get usecase error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list-usecases")
async def list_usecases(limit: int = 100):
    """List all usecases"""
    try:
        col = await get_or_create_collection("usecases")
        
        cursor = col.find({}).sort("created_at", -1).limit(limit)
        usecases = await cursor.to_list(length=limit)
        
        # Remove MongoDB _id from each usecase
        for usecase in usecases:
            usecase.pop("_id", None)
        
        return JSONResponse(
            status_code=200,
            content={
                "response": usecases,
                "total_count": len(usecases)
            }
        )
        
    except Exception as e:
        logger.error(f"❌ List usecases error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete-usecase/{usecase_id}")
async def delete_usecase(usecase_id: str):
    """Delete a usecase by ID"""
    try:
        col = await get_or_create_collection("usecases")
        result = await col.delete_one({"id": usecase_id})
        
        if result.deleted_count == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Usecase '{usecase_id}' not found"
            )
        
        logger.info(f"✅ Deleted usecase: {usecase_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "response": f"Usecase '{usecase_id}' deleted successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Delete usecase error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
