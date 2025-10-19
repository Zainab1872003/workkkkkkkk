
"""
Professional document loader with OCR support and standardized metadata.
Supports PDF, Word, PowerPoint, Excel, and text files with accurate text extraction.
"""

import os
import logging
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredPowerPointLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.excel import load_and_chunk_excel_with_ocr

logger = logging.getLogger(__name__)

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""
    pass

class DocumentTypeHandler:
    """Handles different document type processing with standardized metadata."""
    
    SUPPORTED_EXTENSIONS = {
        ".pdf": "pdf",
        ".doc": "word", 
        ".docx": "word",
        ".ppt": "powerpoint",
        ".pptx": "powerpoint", 
        ".xls": "excel",
        ".xlsx": "excel",
        ".txt": "text"
    }
    
    # Fields to exclude from metadata for Pinecone compatibility
    EXCLUDED_METADATA_FIELDS = {
        "coordinates", "layout_height", "layout_width", 
        "points", "coordinate_system", "chunk_reference_md"
    }
    
    # Indicators that text was extracted from images
    IMAGE_INDICATORS = (
        "image", "figure", "picture", "photo", "diagram", 
        "graphic", "drawing", "table"
    )
    
    @staticmethod
    def office_to_pdf(input_path: str, output_dir: str = "uploads") -> str:
        """Convert Office documents to PDF using LibreOffice."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            base_name = Path(input_path).stem
            output_path = Path(output_dir) / f"{base_name}.pdf"
            
            subprocess.run([
                "soffice", "--headless", "--convert-to", "pdf",
                "--outdir", str(output_dir), str(input_path)
            ], check=True, timeout=300)
            
            return str(output_path)
            
        except subprocess.TimeoutExpired:
            raise DocumentProcessingError(f"Timeout converting {input_path} to PDF")
        except subprocess.CalledProcessError as e:
            raise DocumentProcessingError(f"Failed to convert {input_path} to PDF: {e}")
    
    @staticmethod
    def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata for vector store compatibility."""
        if not metadata:
            return {}
            
        cleaned = {}
        for key, value in metadata.items():
            # Skip problematic fields
            if key in DocumentTypeHandler.EXCLUDED_METADATA_FIELDS or value is None:
                continue
                
            # Keep only compatible types
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
                
        return cleaned
    
    @staticmethod
    def is_text_from_image(metadata: Dict[str, Any]) -> bool:
        """Determine if text was extracted from an image."""
        if not isinstance(metadata, dict):
            return False
            
        # Check standard metadata fields
        fields_to_check = [
            metadata.get("category", ""),
            metadata.get("type", ""),
            metadata.get("element_class", ""),
            metadata.get("parent_id", "")
        ]
        
        for field in fields_to_check:
            field_str = str(field).lower()
            if any(indicator in field_str for indicator in DocumentTypeHandler.IMAGE_INDICATORS):
                return True
        
        # Check for explicit image metadata keys
        image_keys = (
            "image", "image_id", "image_metadata", "image_path", 
            "image_bbox", "image_bytes", "text_as_html"
        )
        return any(key in metadata for key in image_keys)

class DocumentChunker:
    """Handles document chunking with accurate reference tracking."""
    
    def __init__(self, chunk_size: int = 5000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", "],
            length_function=len,
            is_separator_regex=False
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents with standardized metadata."""
        all_chunks = []
        chunk_counters = defaultdict(int)
        
        for doc in documents:
            doc_chunks = self._chunk_single_document(doc, chunk_counters)
            all_chunks.extend(doc_chunks)
            
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _chunk_single_document(self, doc: Document, chunk_counters: Dict[str, int]) -> List[Document]:
        """Chunk a single document with metadata preservation."""
        # Extract document metadata
        doc_type = doc.metadata.get("document_type", "unknown")
        reference_type = doc.metadata.get("reference_type", "page")
        source_type = doc.metadata.get("source_type", "text")
        extraction_method = doc.metadata.get("extraction_method", "unknown")
        
        # Get reference number
        reference_num = self._get_reference_number(doc.metadata, reference_type)
        
        # Split document into chunks
        page_chunks = self.splitter.split_documents([doc])
        
        # Process each chunk
        chunks = []
        for chunk in page_chunks:
            processed_chunk = self._process_chunk(
                chunk, doc_type, reference_type, source_type, 
                extraction_method, reference_num, chunk_counters
            )
            chunks.append(processed_chunk)
            
        return chunks
    
    def _get_reference_number(self, metadata: Dict[str, Any], reference_type: str) -> int:
        """Extract reference number from metadata."""
        if reference_type == "slide":
            return metadata.get("slide_number", 1)
        elif reference_type == "sheet":
            return metadata.get("sheet_number", 1)
        else:
            return metadata.get("page_number", 1)
    
    def _process_chunk(self, chunk: Document, doc_type: str, reference_type: str, 
                      source_type: str, extraction_method: str, reference_num: int,
                      chunk_counters: Dict[str, int]) -> Document:
        """Process individual chunk with standardized metadata."""
        # Generate chunk counter key and increment
        counter_key = f"{reference_type}_{reference_num}_{source_type}"
        chunk_counters[counter_key] += 1
        chunk_number = chunk_counters[counter_key]
        
        # Build standardized metadata (excluding content and reference_md)
        chunk.metadata.update({
            "page_number": reference_num,
            "reference_type": reference_type,
            "source_type": source_type,
            "extraction_method": extraction_method,
            "chunk_number": chunk_number,
            "document_type": doc_type,
            "text": chunk.page_content,  # Only text content in metadata
            "chunk_reference": self._create_reference(reference_type, reference_num, chunk_number, source_type)
        })
        
        # Add specific reference numbers for different document types
        if reference_type == "slide":
            chunk.metadata["slide_number"] = reference_num
        elif reference_type == "sheet":
            chunk.metadata["sheet_number"] = reference_num
            
        return chunk
    
    def _create_reference(self, ref_type: str, ref_num: int, chunk_num: int, source_type: str) -> str:
        """Create human-readable reference string."""
        base_ref = f"{ref_type.title()} {ref_num}, Chunk {chunk_num}"
        
        if source_type == "text_from_image":
            return f"{base_ref} (text from image)"
        return base_ref

class DocumentLoader:
    """Main document loader with support for multiple file types."""
    
    def __init__(self):
        self.handler = DocumentTypeHandler()
        
    def load_documents(self, file_path: str) -> List[Document]:
        """Load documents based on file extension."""
        try:
            file_path = Path(file_path)
            extension = file_path.suffix.lower()
            
            if extension not in DocumentTypeHandler.SUPPORTED_EXTENSIONS:
                raise DocumentProcessingError(f"Unsupported file type: {extension}")
                
            doc_type = DocumentTypeHandler.SUPPORTED_EXTENSIONS[extension]
            logger.info(f"Loading {doc_type} document: {file_path.name}")
            
            # Route to appropriate loader
            if extension == ".pdf":
                return self._load_pdf(str(file_path))
            elif extension in [".doc", ".docx"]:
                return self._load_word(str(file_path))
            elif extension in [".ppt", ".pptx"]:
                return self._load_powerpoint(str(file_path))
            elif extension in [".xls", ".xlsx"]:
                return self._load_excel(str(file_path))
            elif extension == ".txt":
                return self._load_text(str(file_path))
                
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")
            raise DocumentProcessingError(f"Document loading failed: {e}")
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF documents with OCR support."""
        documents = []
        
        try:
            ocr_config = {
                "mode": "elements",
                "strategy": "hi_res",
                "languages": ["eng"],
                "pdf_image_dpi": 96,
                "pdf_infer_table_structure": False,
                "extract_images": False,
                "include_page_breaks": False,
            }
            
            loader = UnstructuredPDFLoader(file_path, **ocr_config)
            elements = loader.load()
            
            # Group elements by page and source type
            page_groups = defaultdict(lambda: {"text": [], "text_from_image": []})
            
            for element in elements:
                metadata = element.metadata or {}
                page_num = metadata.get("page_number", 1) or 1
                
                if self.handler.is_text_from_image(metadata):
                    page_groups[page_num]["text_from_image"].append(element.page_content)
                else:
                    page_groups[page_num]["text"].append(element.page_content)
            
            # Create documents for each page
            for page_num in sorted(page_groups.keys()):
                page_data = page_groups[page_num]
                
                # Regular text content
                if page_data["text"]:
                    text_content = "\n".join(page_data["text"]).strip()
                    if text_content:
                        documents.append(Document(
                            page_content=text_content,
                            metadata={
                                "page_number": page_num,
                                "document_type": "pdf",
                                "reference_type": "page",
                                "source_type": "text",
                                "extraction_method": "unstructured_ocr",
                            }
                        ))
                
                # Text from images
                if page_data["text_from_image"]:
                    image_text = "\n".join(page_data["text_from_image"]).strip()
                    if image_text:
                        documents.append(Document(
                            page_content=image_text,
                            metadata={
                                "page_number": page_num,
                                "document_type": "pdf",
                                "reference_type": "page",
                                "source_type": "text_from_image",
                                "extraction_method": "unstructured_ocr",
                            }
                        ))
            
            logger.info(f"Successfully processed PDF: {len(documents)} sections")
            return documents
            
        except Exception as e:
            logger.error(f"PDF processing failed for {file_path}: {e}")
            return []
    
    def _load_word(self, file_path: str) -> List[Document]:
        """Load Word documents by converting to PDF first."""
        try:
            pdf_path = self.handler.office_to_pdf(file_path)
            return self._load_pdf(pdf_path)
        except Exception as e:
            logger.error(f"Word document processing failed: {e}")
            return []
    
    def _load_powerpoint(self, file_path: str) -> List[Document]:
        """Load PowerPoint documents."""
        try:
            extension = Path(file_path).suffix.lower()
            
            if extension == ".ppt":
                # Convert legacy PPT to PDF first
                pdf_path = self.handler.office_to_pdf(file_path)
                pdf_docs = self._load_pdf(pdf_path)
                
                # Convert page references to slide references
                processed = []
                for doc in pdf_docs:
                    metadata = dict(doc.metadata or {})
                    slide_num = int(metadata.get("page_number", 0) or 0)
                    
                    metadata.update({
                        "document_type": "powerpoint",
                        "slide_number": slide_num,
                        "reference_type": "slide",
                        "filename": Path(file_path).name,
                        "source": file_path,
                    })
                    
                    processed.append(Document(
                        page_content=doc.page_content,
                        metadata=self.handler.clean_metadata(metadata)
                    ))
                return processed
            else:
                # Use direct PowerPoint loader for modern formats
                ocr_config = {
                    "mode": "elements", 
                    "strategy": "hi_res", 
                    "languages": ["eng"]
                }
                
                loader = UnstructuredPowerPointLoader(file_path, **ocr_config)
                docs = loader.load()
                
                processed = []
                for doc in docs:
                    raw_metadata = dict(doc.metadata or {})
                    from_image = self.handler.is_text_from_image(raw_metadata)
                    
                    cleaned_metadata = self.handler.clean_metadata(raw_metadata)
                    cleaned_metadata.update({
                        "document_type": "powerpoint",
                        "slide_number": cleaned_metadata.get("page_number", 0),
                        "reference_type": "slide",
                        "source_type": "text_from_image" if from_image else "text",
                        "extraction_method": "unstructured_ocr"
                    })
                    
                    doc.metadata = cleaned_metadata
                    processed.append(doc)
                
                return processed
                
        except Exception as e:
            logger.error(f"PowerPoint processing failed: {e}")
            return []
    
    def _load_excel(self, file_path: str) -> List[Document]:
        """Load Excel documents with OCR support."""
        try:
            chunks = load_and_chunk_excel_with_ocr(file_path)
            for chunk in chunks:
                chunk.metadata = self.handler.clean_metadata(chunk.metadata)
                chunk.metadata.update({
                    "document_type": "excel",
                    "sheet_number": chunk.metadata.get("sheet", 0),
                    "reference_type": "sheet",
                    "source_type": chunk.metadata.get("object_type", "table_rows")
                })
            return chunks
        except Exception as e:
            logger.error(f"Excel processing failed: {e}")
            return []
    
    def _load_text(self, file_path: str) -> List[Document]:
        """Load plain text documents."""
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
            
            for doc in documents:
                doc.metadata = self.handler.clean_metadata(doc.metadata)
                doc.metadata.update({
                    "document_type": "text",
                    "page_number": 1,
                    "reference_type": "page",
                    "source_type": "text"
                })
            return documents
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return []

def load_and_process_document(file_path: str, chunk_size: int = 5000, chunk_overlap: int = 200) -> List[Document]:
    """
    Main entry point for document loading and processing.
    
    Args:
        file_path: Path to the document file
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of processed document chunks with standardized metadata
    """
    try:
        # Load documents
        loader = DocumentLoader()
        documents = loader.load_documents(file_path)
        
        if not documents:
            logger.warning(f"No content extracted from {file_path}")
            return []
        
        # Chunk documents
        chunker = DocumentChunker(chunk_size, chunk_overlap)
        chunks = chunker.chunk_documents(documents)
        
        # Log summary
        text_chunks = [c for c in chunks if c.metadata.get("source_type") == "text"]
        image_chunks = [c for c in chunks if c.metadata.get("source_type") == "text_from_image"]
        
        logger.info(f"Document processing complete:")
        logger.info(f"  - Total chunks: {len(chunks)}")
        logger.info(f"  - Text chunks: {len(text_chunks)}")
        logger.info(f"  - Image text chunks: {len(image_chunks)}")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Document processing failed for {file_path}: {e}")
        raise DocumentProcessingError(f"Failed to process document: {e}")
