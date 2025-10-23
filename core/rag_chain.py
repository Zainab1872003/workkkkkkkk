
"""
RAG Chain module for document querying using LangChain with Zilliz Cloud (Milvus) vector store.
Handles retrieval and question-answering chains with Groq LLM.
Updated for Milvus/Zilliz: Uses token auth, embedding_function, and reuses DocumentStore for consistency.
"""

import os
from typing import Optional ,List
from langchain_core.callbacks import CallbackManagerForRetrieverRun

import logging
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever 

from langchain.chains import RetrievalQA
from core.config import settings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # For Groq/OpenRouter compatibility

from core.config import (
    GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL,
    MILVUS_URI, MILVUS_PASSWORD  # For Milvus token auth (user empty)
)
from core.embeddings import get_embeddings_model
from core.vectorstore import init_milvus, DocumentStore  # Reuse for correct Milvus wrapper




# ============================================================================
# RETRIEVER WITH BGE RERANKING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# BGE RERANKED RETRIEVER (Proper LangChain Implementation)
# ============================================================================

class BGERerankedRetriever(BaseRetriever):
    """
    LangChain-compatible retriever with BGE reranking.
    Inherits from BaseRetriever for proper integration.
    """
    
    base_retriever: BaseRetriever
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 5
    batch_size: int = 32
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve documents and rerank them using BGE.
        This method is called by LangChain's retrieval system.
        """
        from core.reranker import rerank_documents, batch_rerank_documents
        
        # Step 1: Get initial documents via vector search
        docs = self.base_retriever.invoke(query)
        logger.info(f"📄 Retrieved {len(docs)} initial documents")
        
        if not docs:
            return []
        
        # Step 2: Rerank with BGE
        try:
            if len(docs) > self.batch_size:
                reranked_docs = batch_rerank_documents(
                    query=query,
                    documents=docs,
                    top_k=self.top_k,
                    batch_size=self.batch_size,
                    model_name=self.model_name
                )
            else:
                reranked_docs = rerank_documents(
                    query=query,
                    documents=docs,
                    top_k=self.top_k,
                    model_name=self.model_name
                )
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"❌ Reranking failed: {e}")
            logger.warning(f"⚠️ Falling back to top {self.top_k} from vector search")
            return docs[:self.top_k]


# ============================================================================
# RETRIEVER WITH RERANKING
# ============================================================================

def get_retriever(collection_name: str = "rag_langchain", k: int = 5):
    """
    Initialize Milvus retriever with optional BGE reranking.
    
    Returns:
        BaseRetriever instance (with or without reranking)
    """
    # Determine retrieval strategy
    if settings.RERANKER_ENABLED:
        initial_k = settings.RERANKER_INITIAL_K
        final_k = settings.RERANKER_TOP_K
        logger.info(f"🔍 Retriever: Fetch {initial_k} docs → Rerank to top {final_k} (BGE v2-m3)")
    else:
        initial_k = k
        final_k = k
        logger.info(f"🔍 Retriever: Fetch {initial_k} docs (no reranking)")
    
    # Initialize Milvus vector store
    embeddings = get_embeddings_model()
    init_milvus(collection_name=collection_name)
    
    # Reuse DocumentStore for correct Milvus wrapper
    store = DocumentStore(collection_name=collection_name)
    vectorstore = store._create_langchain_wrapper()
    
    if vectorstore is None:
        raise ValueError(f"Failed to create Milvus vectorstore")
    
    # Create base retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": initial_k})
    
    # Wrap with BGE reranker if enabled
    if settings.RERANKER_ENABLED:
        try:
            logger.info(f"✅ BGE Reranker enabled: {settings.RERANKER_MODEL}")
            
            return BGERerankedRetriever(
                base_retriever=base_retriever,
                model_name=settings.RERANKER_MODEL,
                top_k=final_k,
                batch_size=settings.RERANKER_BATCH_SIZE
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to enable reranking: {e}")
            logger.warning("⚠️ Falling back to base retriever")
            return base_retriever
    
    return base_retriever



# ---- Retriever ----
# def get_retriever(collection_name: str = "rag_langchain", k: int = 5):
#     """
#     Initialize Milvus vector store and return a LangChain retriever.
    
#     Args:
#         collection_name: Milvus collection name (underscore for validity).
#         k: Number of top documents to retrieve.
    
#     Returns:
#         LangChain retriever instance.
#     """
#     embeddings = get_embeddings_model()
#     init_milvus(collection_name=collection_name)  # Ensures connection and collection setup
    
#     # Reuse DocumentStore for correct Milvus wrapper (avoids direct init arg errors)
#     store = DocumentStore(collection_name=collection_name)
#     vectorstore = store._create_langchain_wrapper()  # Handles embedding_function, token auth, schema
    
#     if vectorstore is None:
#         raise ValueError(f"Failed to create Milvus vectorstore for collection '{collection_name}'")
    
#     return vectorstore.as_retriever(search_kwargs={"k": k})

# def get_llm(model: Optional[str] = None):
#     """
#     Configure Groq free LLM (e.g., Llama 3.1) via LangChain ChatOpenAI.
#     - Requires GROQ_API_KEY in your environment/config.
    
#     Args:
#         model: Optional model name override.
    
#     Returns:
#         LangChain LLM instance.
#     """
#     key = GROQ_API_KEY
#     if not key:
#         raise ValueError("GROQ_API_KEY not set in config or .env")
#     model_name = model or GROQ_MODEL

#     return ChatOpenAI(
#         model=model_name,
#         api_key=key,
#         base_url="https://api.groq.com/openai/v1",  # Groq API endpoint
#         temperature=0.1,
#         max_tokens=2048,
#     )

# Alternative LLM (commented; uncomment for OpenRouter fallback)
# def get_llm(model: Optional[str] = None):
#     """
#     Configure DeepSeek-R1 (or any OpenRouter model) via LangChain ChatOpenAI.
#     - Requires OPENROUTER_API_KEY in your environment.
#     """
#     key = os.getenv("OPENROUTER_API_KEY")
#     if not key:
#         raise ValueError("OPENROUTER_API_KEY not set in .env")
#     model_name = model or "deepseek/deepseek-r1:free"

#     return ChatOpenAI(
#         model=model_name,
#         api_key=key,
#         base_url="https://openrouter.ai/api/v1",  # OpenRouter API
#         temperature=0.1,
#         max_tokens=2048,
#     )

def get_llm(model: Optional[str] = None):
    """Configure OpenAI LLM for RAG chain."""
    from core.config import settings
    
    key = settings.OPENAI_API_KEY
    if not key:
        raise ValueError("OPENAI_API_KEY not set in config or .env")
    
    model_name = model or settings.OPENAI_MODEL
    return ChatOpenAI(
        model=model_name,
        api_key=key,
        # No base_url needed for OpenAI (uses default)
        temperature=0.1,
        max_tokens=2048,
    )

# ---- Prompt for RAG ----
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a careful RAG assistant.\n"
        "Use ONLY the provided context to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

# ---- RAG Chain ----
def get_rag_chain(collection_name: str = "rag_langchain"):
    """
    Returns a RetrievalQA chain that:
      - Retrieves top-k documents from Milvus (Zilliz Cloud)
      - Stuffs them into the prompt
      - Calls a Groq LLM
      - Returns the answer + source documents
    
    Args:
        collection_name: Milvus collection name.
    
    Returns:
        RetrievalQA chain instance.
    """
    retriever = get_retriever(collection_name=collection_name)
    llm = get_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
        verbose=False,
    )
    return qa_chain
