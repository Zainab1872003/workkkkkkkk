# # agno_tools/document_tools.py
# """
# Tools for Document Agent - RAG and document operations
# """
# import logging
# from typing import Optional
# from core.rag_chain import get_rag_chain
# from core.vectorstore import DocumentStore

# logger = logging.getLogger(__name__)


# def query_documents(query: str, user_id: str = "default") -> str:
#     """
#     Search through uploaded documents using RAG.
    
#     Args:
#         query: The search query
#         user_id: User identifier
    
#     Returns:
#         Answer with citations
#     """
#     try:
#         logger.info(f"📄 query_documents: '{query[:50]}...'")
        
#         rag_chain = get_rag_chain(collection_name="rag_langchain")
#         result = rag_chain.invoke({"query": query})
        
#         answer = result.get("result", "No answer found.")
#         sources = result.get("source_documents", [])
        
#         response = f"{answer}\n\n**Sources:**\n"
#         for i, doc in enumerate(sources[:3], 1):
#             meta = doc.metadata
#             response += f"{i}. {meta.get('filename', 'Unknown')} - Page {meta.get('page_number', 'N/A')}\n"
        
#         return response
#     except Exception as e:
#         logger.error(f"❌ Error: {e}")
#         return f"Error: {str(e)}"


# def list_documents(user_id: str = "default") -> str:
#     """
#     List all uploaded documents for a user.
    
#     Args:
#         user_id: User identifier
    
#     Returns:
#         List of document names
#     """
#     try:
#         logger.info(f"📚 list_documents for user: {user_id}")
#         # TODO: Implement actual document listing from vector store
#         return "Document listing feature - coming soon!"
#     except Exception as e:
#         return f"Error: {str(e)}"


# def summarize_document(filename: str, user_id: str = "default") -> str:
#     """
#     Generate a summary of a specific document.
    
#     Args:
#         filename: Name of the document to summarize
#         user_id: User identifier
    
#     Returns:
#         Document summary
#     """
#     try:
#         logger.info(f"📝 summarize_document: {filename}")
#         # TODO: Implement document summarization
#         return f"Summary feature for {filename} - coming soon!"
#     except Exception as e:
#         return f"Error: {str(e)}"


# # Tool registry for Document Agent
# DOCUMENT_TOOLS = {
#     "query_documents": query_documents,
#     "list_documents": list_documents,
#     "summarize_document": summarize_document,
# }


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
            collection_name = f"rag_langchain"
        
        # Get RAG chain and query
        rag_chain = get_rag_chain(collection_name="rag_langchain")
        result = rag_chain.invoke({"query": query})
        
        # Extract answer and sources
        answer = result.get("result", "No answer found in the documents.")
        print("Answer:", answer)
        source_docs = result.get("source_documents", [])
        print("Source Documents:", source_docs)

        # Format response with sources
        response = f"{answer}\n\n**Sources:**\n"
        print(response)
        
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

# agno_tools/document_tools.py

# Update the calculate_bike_ijarah function with strict typing

# agno_tools/document_tools.py

from pydantic import BaseModel, Field

# ============================================================================
# TOOL 5: Calculate Bike Ijarah with Pydantic Schema
# ============================================================================

class BikeIjarahInput(BaseModel):
    """Input schema for bike Ijarah calculation"""
    bike_price: float = Field(
        ..., 
        description="Total bike price in PKR (e.g., 500000)",
        gt=0
    )
    down_payment: float = Field(
        ..., 
        description="Initial down payment in PKR (e.g., 100000)",
        ge=0
    )
    tenure_months: int = Field(
        ..., 
        description="Loan tenure in months (e.g., 36)",
        gt=0,
        le=72
    )
    profit_rate: float = Field(
        default=15.0,
        description="Annual profit rate percentage (default: 15%)",
        ge=0,
        le=100
    )


def calculate_bike_ijarah(
    bike_price: float,
    down_payment: float,
    tenure_months: int,
    profit_rate: float = 15.0
) -> str:
    """
    Calculate Islamic bike financing (Ijarah) monthly installments.
    
    This tool calculates the monthly payment for Islamic bike financing.
    All numerical inputs must be provided as numbers, not strings.
    
    Args:
        bike_price (float): Total bike price in PKR. Example: 500000
        down_payment (float): Initial down payment in PKR. Example: 100000
        tenure_months (int): Loan tenure in months. Example: 36
        profit_rate (float): Annual profit rate percentage. Default: 15.0
    
    Returns:
        str: Detailed calculation breakdown with monthly installment
    
    Examples:
        >>> calculate_bike_ijarah(500000, 100000, 36, 15.0)
        >>> calculate_bike_ijarah(750000, 150000, 48, 12.5)
    """
    try:
        logger.info(f"💰 Tool: calculate_bike_ijarah")
        logger.info(f"   Inputs: price={bike_price}, down={down_payment}, months={tenure_months}, rate={profit_rate}")
        
        # Type conversion with validation
        bike_price = float(bike_price)
        down_payment = float(down_payment)
        tenure_months = int(tenure_months)
        profit_rate = float(profit_rate)
        
        # Business logic validation
        if bike_price <= 0:
            return "❌ Error: Bike price must be greater than zero"
        
        if down_payment < 0:
            return "❌ Error: Down payment cannot be negative"
        
        if down_payment >= bike_price:
            return "❌ Error: Down payment must be less than bike price"
        
        if tenure_months <= 0:
            return "❌ Error: Tenure must be at least 1 month"
        
        if tenure_months > 72:
            return "❌ Error: Maximum tenure is 72 months"
        
        if profit_rate < 0 or profit_rate > 100:
            return "❌ Error: Profit rate must be between 0% and 100%"
        
        # Calculate financing
        financed_amount = bike_price - down_payment
        total_profit = (financed_amount * profit_rate * tenure_months) / (100 * 12)
        total_amount = financed_amount + total_profit
        monthly_installment = total_amount / tenure_months
        
        # Format result
        result = f"""
**🏍️ Bike Ijarah Calculation Results**

**Bike Details:**
• Bike Price: PKR {bike_price:,.0f}
• Down Payment: PKR {down_payment:,.0f}
• Financed Amount: PKR {financed_amount:,.0f}

**Financing Terms:**
• Profit Rate: {profit_rate:.1f}% per annum
• Tenure: {tenure_months} months ({tenure_months/12:.1f} years)

**Payment Breakdown:**
• Total Profit: PKR {total_profit:,.2f}
• Total Amount Payable: PKR {total_amount:,.2f}
• **Monthly Installment: PKR {monthly_installment:,.2f}** 💳

**Summary:** You will pay PKR {monthly_installment:,.0f} per month for {tenure_months} months.
        """.strip()
        
        logger.info(f"✅ Calculation complete - Monthly: PKR {monthly_installment:,.2f}")
        return result
        
    except (ValueError, TypeError) as e:
        logger.error(f"❌ Type conversion error: {e}")
        return f"❌ Error: Invalid input types. All numbers must be provided as numeric values, not strings. Details: {str(e)}"
    except Exception as e:
        logger.error(f"❌ Calculation error: {e}", exc_info=True)
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
