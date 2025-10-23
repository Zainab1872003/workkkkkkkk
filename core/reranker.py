# core/reranker.py
"""
Production-grade local reranker using BAAI/bge-reranker-v2-m3
- FREE (no API costs)
- Multilingual support (100+ languages)
- State-of-the-art performance
- Runs locally on your server
"""
import logging
import os
from typing import List, Optional
import torch
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Global cache for reranker model
_reranker_model = None
_model_name = "BAAI/bge-reranker-v2-m3"


def get_bge_reranker(model_name: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None):
    """
    Get cached BGE reranker model instance.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ('cuda', 'cpu', or None for auto)
    
    Returns:
        FlagReranker instance
    """
    global _reranker_model, _model_name
    
    if _reranker_model is None or _model_name != model_name:
        try:
            from FlagEmbedding import FlagReranker
            
            # Auto-detect device if not specified
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            logger.info(f"🔄 Loading BGE Reranker: {model_name} on {device.upper()}...")
            
            hf_token = os.getenv('HF_TOKEN')

            _reranker_model = FlagReranker(
                model_name,
                use_fp16=True if device == 'cuda' else False,
                use_auth_token=hf_token  # Add this line
            )
            _model_name = model_name
            
            logger.info(f"✅ BGE Reranker loaded successfully on {device.upper()}")
            
        except ImportError:
            raise ImportError("Install FlagEmbedding: pip install FlagEmbedding")
        except Exception as e:
            logger.error(f"❌ Failed to load BGE Reranker: {e}")
            raise
    
    return _reranker_model


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 4,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    normalize: bool = True
) -> List[Document]:
    """
    Rerank documents using BGE reranker v2-m3.
    
    Args:
        query: Search query
        documents: List of retrieved documents
        top_k: Number of top documents to return after reranking
        model_name: BGE reranker model name
        normalize: Whether to normalize scores to [0, 1]
    
    Returns:
        Reranked list of top-k documents with scores
    """
    try:
        if not documents:
            logger.warning("⚠️ No documents to rerank")
            return []
        
        logger.info(f"🔄 Reranking {len(documents)} documents with BGE v2-m3...")
        
        # Get BGE reranker model
        reranker = get_bge_reranker(model_name)
        
        # Prepare query-document pairs for reranking
        # Format: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Compute relevance scores
        # scores are in range [-inf, +inf], higher = more relevant
        scores = reranker.compute_score(pairs, normalize=normalize)
        
        # Handle single document case (returns float instead of list)
        if isinstance(scores, float):
            scores = [scores]
        
        # Create list of (document, score) tuples
        doc_score_pairs = list(zip(documents, scores))
        
        # Sort by score in descending order
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k documents and add scores to metadata
        reranked_docs = []
        for doc, score in doc_score_pairs[:top_k]:
            doc.metadata['rerank_score'] = float(score)
            doc.metadata['rerank_model'] = model_name
            reranked_docs.append(doc)
        
        # Log top scores
        top_scores = [f"{score:.3f}" for _, score in doc_score_pairs[:3]]
        logger.info(f"✅ Reranked: {len(documents)} → {len(reranked_docs)} docs (top scores: {top_scores})")
        
        return reranked_docs
        
    except Exception as e:
        logger.error(f"❌ BGE reranking failed: {e}", exc_info=True)
        logger.warning(f"⚠️ Falling back to original retrieval (no reranking)")
        return documents[:top_k]


def batch_rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 5,
    batch_size: int = 32,
    model_name: str = "BAAI/bge-reranker-v2-m3"
) -> List[Document]:
    """
    Rerank large document sets in batches for memory efficiency.
    
    Args:
        query: Search query
        documents: List of retrieved documents
        top_k: Number of top documents to return
        batch_size: Number of documents to process per batch
        model_name: BGE reranker model name
    
    Returns:
        Reranked list of top-k documents
    """
    try:
        if len(documents) <= batch_size:
            return rerank_documents(query, documents, top_k, model_name)
        
        logger.info(f"🔄 Batch reranking {len(documents)} documents (batch_size={batch_size})...")
        
        all_scores = []
        reranker = get_bge_reranker(model_name)
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            pairs = [[query, doc.page_content] for doc in batch_docs]
            batch_scores = reranker.compute_score(pairs, normalize=True)
            
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            
            all_scores.extend(batch_scores)
            logger.debug(f"   Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        # Sort and get top-k
        doc_score_pairs = list(zip(documents, all_scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        reranked_docs = []
        for doc, score in doc_score_pairs[:top_k]:
            doc.metadata['rerank_score'] = float(score)
            doc.metadata['rerank_model'] = model_name
            reranked_docs.append(doc)
        
        logger.info(f"✅ Batch reranked: {len(documents)} → {len(reranked_docs)} docs")
        return reranked_docs
        
    except Exception as e:
        logger.error(f"❌ Batch reranking failed: {e}")
        return documents[:top_k]
