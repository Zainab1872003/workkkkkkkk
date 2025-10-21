# agno_tools/document_tools.py
"""
Document management tools for AGNO agent
All tools work with Milvus/Zilliz vector database
"""
import os
import logging
from typing import Optional

from core.config import settings
from core.rag_chain import get_rag_chain
from core.document_loader1 import load_and_process_document
from core.vectorstore import DocumentStore
from core.database import get_collection

logger = logging.getLogger(__name__)


# ============================================================================
# TOOL 1: Query Documents (RAG)
# ============================================================================

def query_documents(query: str, user_id: str = "default") -> str:
    """
    Search through uploaded documents using RAG (Retrieval Augmented Generation).
    
    Use this tool whenever a user asks about information in their uploaded documents.
    
    Args:
        query: The search query or question
        user_id: User ID for accessing user-specific documents
    
    Returns:
        Answer with source citations from the documents
    
    Example:
        query_documents("What are the faculty website headers?", "test_user")
    """
    try:
        logger.info(f"🔍 Tool: query_documents - User: {user_id}, Query: {query}")
        
        # Use user-specific collection or default
        if user_id == "default":
            collection_name = "rag_langchain"
        else:
            collection_name = f"user_{user_id}"
        
        # Get RAG chain and query
        rag_chain = get_rag_chain(collection_name=collection_name)
        result = rag_chain.invoke({"query": query})
        
        # Extract answer and sources
        answer = result.get("result", "No answer found in the documents.")
        source_docs = result.get("source_documents", [])
        
        # Format response with sources
        response = f"{answer}\n\n**Sources:**\n"
        for i, doc in enumerate(source_docs[:3], 1):
            meta = doc.metadata
            filename = meta.get('filename', 'Unknown')
            page = meta.get('page_number', 'N/A')
            doc_type = meta.get('document_type', '')
            response += f"{i}. {filename} ({doc_type}) - Page {page}\n"
        
        logger.info(f"✅ Tool success - Found {len(source_docs)} sources")
        return response
        
    except Exception as e:
        logger.error(f"❌ Tool error: {e}", exc_info=True)
        return f"Error searching documents: {str(e)}"


# ============================================================================
# TOOL 2: Upload Document
# ============================================================================

def upload_document(
    file_path: str,
    user_id: str,
    chunk_size: int = 5000,
    chunk_overlap: int = 200
) -> str:
    """
    Upload and process a document into the vector database.
    
    Args:
        file_path: Path to the document file
        user_id: User ID for user-specific storage
        chunk_size: Size of text chunks (default: 5000)
        chunk_overlap: Overlap between chunks (default: 200)
    
    Returns:
        Status message with document details
    
    Example:
        upload_document("/path/to/document.pdf", "user123")
    """
    try:
        logger.info(f"📤 Tool: upload_document - File: {file_path}, User: {user_id}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            return f"❌ Error: File not found at {file_path}"
        
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        # Load and chunk document
        chunks = load_and_process_document(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if not chunks:
            return f"❌ Error: No content could be extracted from {filename}"
        
        # Store in vector database
        collection_name = f"rag_langchain"
        store = DocumentStore(collection_name=collection_name)
        vectorstore, document_id = store.store_documents(
            docs=chunks,
            filename=filename,
            file_size=file_size
        )
        
        if document_id:
            logger.info(f"✅ Uploaded {filename} - {len(chunks)} chunks")
            return f"✅ Success: Uploaded **{filename}** with {len(chunks)} chunks (Document ID: {document_id})"
        else:
            return f"❌ Error: Failed to store {filename} in database"
            
    except Exception as e:
        logger.error(f"❌ Upload error: {e}", exc_info=True)
        return f"❌ Error uploading document: {str(e)}"


# ============================================================================
# TOOL 3: List Documents
# ============================================================================

async def list_documents(user_id: str) -> str:
    """
    List all documents uploaded by a specific user.
    
    Args:
        user_id: User ID to list documents for
    
    Returns:
        List of document filenames with metadata
    
    Example:
        await list_documents("user123")
    """
    try:
        logger.info(f"📋 Tool: list_documents - User: {user_id}")
        
        # Get documents from MongoDB
        collection = get_collection()
        cursor = collection.find({"user_id": user_id})
        documents = await cursor.to_list(length=1000)
        
        if not documents:
            return f"📄 No documents found for user: {user_id}"
        
        # Format document list
        doc_list = []
        for doc in documents:
            filename = doc.get("filename", "Unknown")
            upload_date = doc.get("upload_date", "Unknown")
            file_size = doc.get("file_size", 0)
            size_kb = round(file_size / 1024, 2)
            doc_list.append(f"• **{filename}** ({size_kb} KB) - Uploaded: {upload_date}")
        
        response = f"📄 **Documents for {user_id}** ({len(doc_list)} total):\n\n"
        response += "\n".join(doc_list)
        
        logger.info(f"✅ Listed {len(doc_list)} documents")
        return response
        
    except Exception as e:
        logger.error(f"❌ List error: {e}", exc_info=True)
        return f"❌ Error listing documents: {str(e)}"


# ============================================================================
# TOOL 4: Delete Document
# ============================================================================

async def delete_document(user_id: str, filename: str) -> str:
    """
    Delete a specific document for a user.
    
    Args:
        user_id: User ID
        filename: Name of the file to delete
    
    Returns:
        Deletion status message
    
    Example:
        await delete_document("user123", "document.pdf")
    """
    try:
        logger.info(f"🗑️  Tool: delete_document - User: {user_id}, File: {filename}")
        
        # Delete from MongoDB
        collection = get_collection()
        result = await collection.delete_one({
            "user_id": user_id,
            "filename": filename
        })
        
        # Delete from vector store
        collection_name = f"rag_langchain"
        store = DocumentStore(collection_name=collection_name)
        store.delete_by_filename(filename)
        
        if result.deleted_count > 0:
            logger.info(f"✅ Deleted {filename}")
            return f"✅ Successfully deleted **{filename}** for user {user_id}"
        else:
            return f"⚠️  Document **{filename}** not found for user {user_id}"
        
    except Exception as e:
        logger.error(f"❌ Delete error: {e}", exc_info=True)
        return f"❌ Error deleting document: {str(e)}"


# ============================================================================
# TOOL 5: Calculate Bike Ijarah (Example finance tool)
# ============================================================================

def calculate_bike_ijarah(
    bike_price: float,
    down_payment: float,
    tenure_months: int,
    profit_rate: float = 15.0
) -> str:
    """
    Calculate Islamic bike financing (Ijarah) monthly installments.
    
    Args:
        bike_price: Total bike price in PKR
        down_payment: Initial down payment in PKR
        tenure_months: Loan tenure in months
        profit_rate: Annual profit rate percentage (default: 15%)
    
    Returns:
        Detailed calculation breakdown with monthly installment
    
    Example:
        calculate_bike_ijarah(500000, 100000, 36, 15.0)
    """
    try:
        logger.info(f"💰 Tool: calculate_bike_ijarah")
        
        # Calculate financing
        financed_amount = bike_price - down_payment
        total_profit = (financed_amount * profit_rate * tenure_months) / (100 * 12)
        total_amount = financed_amount + total_profit
        monthly_installment = total_amount / tenure_months
        
        # Format result
        result = f"""
**🏍️ Bike Ijarah Calculation**

**Bike Details:**
• Bike Price: PKR {bike_price:,.2f}
• Down Payment: PKR {down_payment:,.2f}
• Financed Amount: PKR {financed_amount:,.2f}

**Financing Terms:**
• Profit Rate: {profit_rate}% per annum
• Tenure: {tenure_months} months

**Payment Breakdown:**
• Total Profit: PKR {total_profit:,.2f}
• Total Amount Payable: PKR {total_amount:,.2f}
• **Monthly Installment: PKR {monthly_installment:,.2f}** 💳
        """.strip()
        
        logger.info("✅ Calculation complete")
        return result
        
    except Exception as e:
        logger.error(f"❌ Calculation error: {e}")
        return f"❌ Error calculating Ijarah: {str(e)}"


# ============================================================================
# Tool Registry - Easy tool management
# ============================================================================

AVAILABLE_TOOLS = {
    "document_retriever": query_documents,
    "document_uploader": upload_document,
    "document_lister": list_documents,
    "document_deleter": delete_document,
    "calculate_bike_ijarah": calculate_bike_ijarah,
}


def get_tools(tool_names: list, user_id: str = "default") -> list:
    """
    Get tool functions by name.
    
    Args:
        tool_names: List of tool names to retrieve
        user_id: User ID to inject into tools (for document tools)
    
    Returns:
        List of tool functions
    
    Example:
        tools = get_tools(["document_retriever", "calculate_bike_ijarah"], "user123")
    """
    tools = []
    
    for tool_name in tool_names:
        if tool_name in AVAILABLE_TOOLS:
            tool_func = AVAILABLE_TOOLS[tool_name]
            
            # For document tools, create a wrapper that injects user_id
            if tool_name in ["document_retriever", "document_uploader"]:
                if tool_name == "document_retriever":
                    def wrapped_query(query: str) -> str:
                        return query_documents(query, user_id)
                    tools.append(wrapped_query)
                elif tool_name == "document_uploader":
                    def wrapped_upload(file_path: str) -> str:
                        return upload_document(file_path, user_id)
                    tools.append(wrapped_upload)
            else:
                tools.append(tool_func)
            
            logger.info(f"✓ Loaded tool: {tool_name}")
    
    return tools
