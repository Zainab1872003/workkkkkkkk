# routers/operations.py
"""
Operations routes for index management and querying
Implements /api/operations endpoints from OpenAPI spec
"""
import logging
from typing import Optional , List

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse

from core.database import get_collection
from core.rag_chain import get_rag_chain
from datetime import datetime
from langchain_core.documents import Document
from core.vectorstore import DocumentStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/operations/index", tags=["Index Operations"])


# ============================================================================
# /api/operations/index/update_index - Update AI Index
# ============================================================================
@router.post("/update_index")
async def update_index(
    index_id: str = Form(...),
    new_text: List[str] = Form(...),
    user_id: Optional[str] = Form(default=None),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200)
):
    """
    Update AI index by adding new text content to an existing document.
    
    - **index_id**: Document identifier (can be filename or document_id)
    - **new_text**: Array of new text strings to add to the index
    - **user_id**: Optional user ID (for filtering)
    - **chunk_size**: Size of text chunks (default: 1000)
    - **chunk_overlap**: Overlap between chunks (default: 200)
    """
    try:
        logger.info("="*70)
        logger.info("📝 UPDATE INDEX REQUEST:")
        logger.info(f"   index_id:   '{index_id}'")
        logger.info(f"   user_id:    '{user_id}'")
        logger.info(f"   new_text:   {len(new_text)} items")
        logger.info(f"   chunk_size: {chunk_size}")
        logger.info("="*70)
        
        # Validate inputs
        if not new_text or len(new_text) == 0:
            raise HTTPException(status_code=400, detail="new_text cannot be empty")
        
        # Step 1: Find document in MongoDB
        col = get_collection()
        
        # Build query - search by filename or document_id
        query = {"$or": [{"filename": index_id}, {"document_id": index_id}]}
        if user_id:
            query["user_id"] = user_id
        
        doc_metadata = await col.find_one(query)
        
        if not doc_metadata:
            # Show available documents
            if user_id:
                available_docs = await col.find({"user_id": user_id}).to_list(length=100)
            else:
                available_docs = await col.find({}).to_list(length=100)
            
            available_list = [{
                "filename": d.get("filename"),
                "document_id": d.get("document_id"),
                "user_id": d.get("user_id")
            } for d in available_docs]
            
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Document with index_id '{index_id}' not found",
                    "searched_for": index_id,
                    "user_id_filter": user_id,
                    "available_documents": available_list,
                    "hint": "Use 'filename' or 'document_id' as index_id"
                }
            )
        
        logger.info(f"✅ Document found in MongoDB")
        logger.info(f"   filename:        {doc_metadata.get('filename')}")
        logger.info(f"   document_id:     {doc_metadata.get('document_id')}")
        logger.info(f"   current_chunks:  {doc_metadata.get('chunk_count')}")
        
        # Get document details
        filename = doc_metadata.get("filename")
        document_id = doc_metadata.get("document_id")
        collection_name = doc_metadata.get("collection_name", "rag_langchain")
        current_chunk_count = doc_metadata.get("chunk_count", 0)
        file_size = doc_metadata.get("file_size", 0)
        
        # Step 2: Process new text into chunks
        logger.info(f"📄 Processing {len(new_text)} new text items...")
        
        # Combine all new text items
        combined_text = "\n\n".join(new_text)
        logger.info(f"   Combined text length: {len(combined_text)} characters")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        text_chunks = text_splitter.split_text(combined_text)
        logger.info(f"✂️  Split into {len(text_chunks)} raw chunks")
        
        # Create Document objects with proper metadata
        documents = []
        for i, chunk_text in enumerate(text_chunks):
            # Calculate chunk index (continue from existing chunks)
            chunk_index = current_chunk_count + i
            
            # Create metadata matching your MongoDB structure
            metadata = {
                "filename": filename,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "source": "update_index_api",
                "update_date": datetime.now().isoformat(),
                "is_appended": True,
                "original_chunk_count": current_chunk_count
            }
            
            doc = Document(
                page_content=chunk_text,
                metadata=metadata
            )
            documents.append(doc)
        
        logger.info(f"📦 Created {len(documents)} Document objects")
        
        # Step 3: Add chunks to Milvus vector store
        logger.info(f"🗄️  Adding chunks to Milvus collection: {collection_name}")
        
        try:
            store = DocumentStore(collection_name="rag_langchain")
            
            # Store documents in vector database
            _, returned_doc_id = store.store_documents(
                docs=documents,
                filename=filename,
                file_size=file_size
            )
            
            logger.info(f"✅ Successfully added {len(documents)} chunks to Milvus")
            logger.info(f"   Returned document_id: {returned_doc_id}")
            
        except Exception as e:
            logger.error(f"❌ Milvus insert failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to add chunks to vector store: {str(e)}"
            )
        
        # Step 4: Update MongoDB metadata
        new_chunk_count = current_chunk_count + len(documents)
        
        update_record = {
            "date": datetime.now().isoformat(),
            "chunks_added": len(documents),
            "source": "update_index_api",
            "text_items": len(new_text)
        }
        
        await col.update_one(
            {"document_id": document_id},
            {
                "$set": {
                    "chunk_count": new_chunk_count,
                    "last_updated": datetime.now().isoformat()
                },
                "$push": {
                    "update_history": update_record
                }
            }
        )
        
        logger.info(f"✅ Updated MongoDB metadata")
        logger.info(f"   Old chunk count: {current_chunk_count}")
        logger.info(f"   New chunk count: {new_chunk_count}")
        logger.info("="*70)
        
        return JSONResponse(
            status_code=200,
            content={
                "response": "Files updated successfully",
                "success": True,
                "document": {
                    "filename": filename,
                    "document_id": document_id,
                    "user_id": doc_metadata.get("user_id"),
                    "collection_name": "rag_langchain",
                },
                "update_summary": {
                    "new_text_items_provided": len(new_text),
                    "new_chunks_created": len(documents),
                    "previous_chunk_count": current_chunk_count,
                    "updated_chunk_count": new_chunk_count,
                    "total_new_characters": len(combined_text)
                },
                "message": f"Successfully added {len(documents)} new chunks to '{filename}'"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Update index error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


# ============================================================================
# /api/operations/index/query_index - Query AI Index
# ============================================================================
@router.post("/query_index")
async def query_index(
    filename: str = Form(...),
    user_id: str = Form(...),
    query: Optional[str] = Form(default=None),
    top_k: int = Form(default=20)
):
    """
    Query the AI index for specific document content.
    
    This endpoint allows querying a specific document or getting all chunks from a document.
    
    - **filename**: Filename to query (e.g., "document.pdf")
    - **user_id**: User ID who owns the document
    - **query**: Optional search query. If not provided, returns all chunks metadata
    - **top_k**: Number of top results to return (default: 3)
    """
    try:
        logger.info(f"🔍 Query index request - User: {user_id}, File: {filename}, Query: {query}")
        
        # Step 1: Verify document exists in MongoDB
        col = get_collection()
        doc_metadata = await col.find_one({"filename": filename, "user_id": user_id})
        print("m yaha aa gya")
        print(doc_metadata)
        
        if not doc_metadata:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{filename}' not found for user '{user_id}'"
            )
        
        # Get collection name
        collection_name = doc_metadata.get("collection_name", f"rag_langchain")
        
        # Step 2: Query vector store
        if query:
            # Semantic search with RAG chain
            result = await query_with_rag(
                query=query,
                filename=filename,
                collection_name="rag_langchain",
                top_k=top_k
            )
        else:
            # Just retrieve document chunks metadata
            result = await get_document_chunks(
                filename=filename,
                collection_name="rag_langchain",
                user_id=user_id
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "response": result,
                "filename": filename,
                "user_id": user_id,
                "query": query if query else "metadata_only"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Query index error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


async def query_with_rag(
    query: str,
    filename: str,
    collection_name: str,
    top_k: int = 3
) -> dict:
    """Query document using RAG chain with filename filtering"""
    try:
        logger.info(f"🤖 Running RAG query: '{query}' on {filename}")
        
        # Get RAG chain for user collection
        rag_chain = get_rag_chain(collection_name="rag_langchain")
        
        # Query the RAG chain
        # Note: The RAG chain searches across all documents in the collection
        # We'll filter by filename in the response
        result = rag_chain.invoke({"query": query})
        
        # Extract answer and sources
        answer = result.get("result", "No answer found.")
        source_docs = result.get("source_documents", [])
        
        # Filter sources by filename
        filtered_sources = [
            doc for doc in source_docs
            if doc.metadata.get("filename") == filename
        ]
        
        # Format sources
        sources = []
        for i, doc in enumerate(filtered_sources[:top_k], 1):
            meta = doc.metadata
            sources.append({
                "chunk_index": meta.get("chunk_index", i-1),
                "text": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": {
                    "filename": meta.get("filename"),
                    "document_id": meta.get("document_id"),
                    "page_number": meta.get("page_number", "N/A"),
                    "document_type": meta.get("document_type", "unknown"),
                    "chunk_size": len(doc.page_content)
                }
            })
        
        logger.info(f"✅ RAG query complete - {len(filtered_sources)} relevant chunks found")
        
        return {
            "query_type": "semantic_search",
            "answer": answer,
            "sources_found": len(filtered_sources),
            "sources_returned": len(sources),
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"❌ RAG query error: {e}")
        raise


async def get_document_chunks(
    filename: str,
    collection_name: str,
    user_id: str
) -> dict:
    """Retrieve all chunks metadata for a document"""
    try:
        logger.info(f"📄 Retrieving chunks metadata for {filename}")
        
        # Initialize DocumentStore
        store = DocumentStore(collection_name="rag_langchain")
        
        # Query Milvus for all chunks of this document
        expr = f'filename == "{filename}"'
        
        results = store.collection.query(
            expr=expr,
            output_fields=["id", "filename", "document_id", "chunk_index", "text", "metadata"],
            limit=1000  # Adjust as needed
        )
        
        if not results:
            return {
                "query_type": "metadata_retrieval",
                "message": f"No chunks found for {filename}",
                "chunks_found": 0,
                "chunks": []
            }
        
        # Format chunks
        chunks = []
        for result in results:
            chunk_info = {
                "chunk_id": result.get("id"),
                "chunk_index": result.get("chunk_index"),
                "text_preview": result.get("text", "")[:200] + "..." if len(result.get("text", "")) > 200 else result.get("text", ""),
                "text_length": len(result.get("text", "")),
                "metadata": result.get("metadata", {})
            }
            chunks.append(chunk_info)
        
        # Sort by chunk_index
        chunks.sort(key=lambda x: x.get("chunk_index", 0))
        
        logger.info(f"✅ Retrieved {len(chunks)} chunks for {filename}")
        
        return {
            "query_type": "metadata_retrieval",
            "filename": filename,
            "chunks_found": len(chunks),
            "chunks": chunks
        }
        
    except Exception as e:
        logger.error(f"❌ Chunks retrieval error: {e}")
        raise


# # ============================================================================
# # Additional Index Operations (Bonus)
# # ============================================================================
# @router.post("/search_across_documents")
# async def search_across_documents(
#     user_id: str = Form(...),
#     query: str = Form(...),
#     top_k: int = Form(default=5)
# ):
#     """
#     Search across ALL documents for a user (not just one file).
    
#     - **user_id**: User ID
#     - **query**: Search query
#     - **top_k**: Number of results (default: 5)
#     """
#     try:
#         logger.info(f"🔍 Cross-document search - User: {user_id}, Query: {query}")
        
#         collection_name = f"user_{user_id}"
        
#         # Get RAG chain
#         rag_chain = get_rag_chain(collection_name=collection_name)
        
#         # Query
#         result = rag_chain.invoke({"query": query})
        
#         # Extract answer and sources
#         answer = result.get("result", "No answer found.")
#         source_docs = result.get("source_documents", [])
        
#         # Format sources
#         sources = []
#         for i, doc in enumerate(source_docs[:top_k], 1):
#             meta = doc.metadata
#             sources.append({
#                 "rank": i,
#                 "filename": meta.get("filename"),
#                 "chunk_index": meta.get("chunk_index", 0),
#                 "text_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
#                 "document_id": meta.get("document_id"),
#                 "relevance_score": "high"  # Milvus returns by distance, higher is better
#             })
        
#         logger.info(f"✅ Cross-document search complete - {len(sources)} results")
        
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "query": query,
#                 "user_id": user_id,
#                 "answer": answer,
#                 "total_sources": len(source_docs),
#                 "sources_returned": len(sources),
#                 "sources": sources
#             }
#         )
        
#     except Exception as e:
#         logger.error(f"❌ Cross-document search error: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/collection_stats")
async def get_collection_stats(user_id: str):
    """
    Get statistics about user's vector collection.
    
    - **user_id**: User ID
    """
    try:
        collection_name = f"user_{user_id}"
        store = DocumentStore(collection_name=collection_name)
        
        # Get collection stats
        stats = store.collection.num_entities
        
        # Get unique documents count
        col = get_collection()
        doc_count = await col.count_documents({"user_id": user_id})
        
        return JSONResponse(
            status_code=200,
            content={
                "user_id": user_id,
                "collection_name": collection_name,
                "total_chunks": stats,
                "total_documents": doc_count,
                "average_chunks_per_document": round(stats / doc_count, 2) if doc_count > 0 else 0
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
