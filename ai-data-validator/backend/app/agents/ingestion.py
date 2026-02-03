"""Data Ingestion Agent - handles file parsing, chunking, and embedding storage."""

import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import os

from ..utils.docparse import DocumentParser
from ..utils.chunker import chunk_document, TextChunk
from ..utils.embeddings import VectorStoreManager
from ..config import config

logger = logging.getLogger(__name__)


class DataIngestionAgent:
    """
    Data Ingestion Agent (DI) - Core agent for processing files and storing embeddings.
    
    Responsibilities:
    - File validation and parsing
    - Text extraction and normalization
    - Semantic chunking with metadata
    - Embedding generation and vector storage
    - Language detection and quality assessment
    """
    
    def __init__(self):
        self.parser = DocumentParser()
        self.vector_manager = VectorStoreManager()
        self.processed_files = []
        self.ingestion_stats = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "embeddings_stored": 0,
            "errors": []
        }
    
    async def ingest(self, 
                    files: List[str] = None, 
                    namespace: str = None, 
                    goal: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main ingestion pipeline.
        
        Args:
            files: List of file paths to process
            namespace: Pinecone namespace for storage
            goal: User goal context for metadata enhancement
            
        Returns:
            Ingestion result with statistics and success status
        """
        if not files:
            return self._create_error_result("No files provided for ingestion")
        
        if not namespace:
            namespace = f"session_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Starting ingestion of {len(files)} files into namespace: {namespace}")
        
        # Reset stats
        self._reset_stats()
        
        # Process each file
        all_chunks = []
        file_results = []
        
        for file_path in files:
            logger.info(f"Processing file: {file_path}")
            
            file_result = await self._process_single_file(file_path, goal)
            file_results.append(file_result)
            
            if file_result["success"]:
                all_chunks.extend(file_result["chunks"])
                self.ingestion_stats["files_processed"] += 1
                self.ingestion_stats["chunks_created"] += len(file_result["chunks"])
            else:
                self.ingestion_stats["files_failed"] += 1
                self.ingestion_stats["errors"].append({
                    "file": file_path,
                    "error": file_result["error"]
                })
        
        # Early exit if no successful processing
        if not all_chunks:
            return self._create_error_result(
                "No files were successfully processed",
                {"file_results": file_results, "stats": self.ingestion_stats}
            )
        
        # Store embeddings in vector database
        logger.info(f"Storing {len(all_chunks)} chunks in vector database")
        
        storage_result = await self.vector_manager.process_and_store_chunks(
            all_chunks, namespace
        )
        
        if storage_result["success"]:
            self.ingestion_stats["embeddings_stored"] = storage_result["upserted_count"]
        else:
            self.ingestion_stats["errors"].append({
                "stage": "embedding_storage",
                "error": storage_result.get("error", "Unknown storage error")
            })
        
        # Quick sanity check
        sanity_result = await self._perform_sanity_check(all_chunks, namespace)
        
        return {
            "success": storage_result["success"],
            "namespace": namespace,
            "stats": self.ingestion_stats,
            "file_results": file_results,
            "chunks": [self._chunk_to_dict(chunk) for chunk in all_chunks[:10]],  # Sample
            "sanity_check": sanity_result,
            "recommendations": self._generate_recommendations()
        }
    
    async def _process_single_file(self, file_path: str, goal: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a single file through the complete pipeline."""
        try:
            # Parse document
            parse_result = self.parser.parse_file(file_path)
            
            if not parse_result["success"]:
                return {
                    "success": False,
                    "file_path": file_path,
                    "error": parse_result["error"],
                    "chunks": []
                }
            
            # Extract text and metadata
            text = parse_result["text"]
            metadata = parse_result["metadata"]
            
            # Enhance metadata with goal context
            if goal:
                metadata.update({
                    "goal_domain": goal.get("domain", "general"),
                    "goal_description": goal.get("description", "")[:200],  # Truncate
                    "processing_timestamp": str(int(asyncio.get_event_loop().time()))
                })
            
            # Language detection and validation
            language = parse_result.get("language", "unknown")
            readability = parse_result.get("readability_score", 0)
            
            if language == "unknown":
                logger.warning(f"Could not detect language for {file_path}")
            
            # Check for OCR artifacts if it's a PDF
            has_ocr_artifacts = False
            ocr_artifact_ratio = 0.0
            
            if file_path.lower().endswith('.pdf'):
                has_ocr_artifacts, ocr_artifact_ratio = self.parser.detect_ocr_artifacts(text)
                if has_ocr_artifacts:
                    logger.warning(f"OCR artifacts detected in {file_path} (ratio: {ocr_artifact_ratio:.2f})")
            
            # Create chunks
            chunks = chunk_document(
                text=text,
                source=Path(file_path).name,
                metadata={
                    **metadata,
                    "language": language,
                    "readability_score": readability,
                    "has_ocr_artifacts": has_ocr_artifacts,
                    "ocr_artifact_ratio": ocr_artifact_ratio,
                    "original_file_path": file_path
                }
            )
            
            # Validate chunks
            valid_chunks = [chunk for chunk in chunks if self._validate_chunk(chunk)]
            
            if len(valid_chunks) < len(chunks):
                logger.warning(f"Filtered out {len(chunks) - len(valid_chunks)} invalid chunks from {file_path}")
            
            return {
                "success": True,
                "file_path": file_path,
                "chunks": valid_chunks,
                "metadata": {
                    "original_chunk_count": len(chunks),
                    "valid_chunk_count": len(valid_chunks),
                    "language": language,
                    "readability_score": readability,
                    "has_ocr_artifacts": has_ocr_artifacts,
                    "file_size": metadata.get("file_size", 0),
                    "file_type": metadata.get("file_extension", "unknown")
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                "success": False,
                "file_path": file_path,
                "error": str(e),
                "chunks": []
            }
    
    def _validate_chunk(self, chunk: TextChunk) -> bool:
        """Validate that a chunk meets minimum quality standards."""
        # Check minimum text length
        if len(chunk.text.strip()) < 10:
            return False
        
        # Check token count
        if chunk.token_count < 5:
            return False
        
        # Check for excessive special characters (possible OCR artifacts)
        special_char_ratio = sum(1 for c in chunk.text if not c.isalnum() and not c.isspace()) / len(chunk.text)
        if special_char_ratio > 0.5:
            return False
        
        # Check for excessive whitespace
        word_count = len(chunk.text.split())
        if word_count < 3:
            return False
        
        return True
    
    async def _perform_sanity_check(self, chunks: List[TextChunk], namespace: str) -> Dict[str, Any]:
        """Perform quick sanity checks on ingested data."""
        try:
            # Basic statistics
            total_chunks = len(chunks)
            total_tokens = sum(chunk.token_count for chunk in chunks)
            avg_chunk_size = total_tokens / total_chunks if total_chunks > 0 else 0
            
            # Language distribution
            languages = {}
            for chunk in chunks:
                lang = chunk.metadata.get("language", "unknown")
                languages[lang] = languages.get(lang, 0) + 1
            
            # Check embedding success rate by trying a simple query
            embedding_success = False
            query_result = None
            
            if total_chunks > 0:
                # Try to query the first chunk
                sample_text = chunks[0].text[:100]  # First 100 chars
                query_result = await self.vector_manager.query_knowledge_base(
                    sample_text, namespace, top_k=1
                )
                embedding_success = query_result.get("success", False)
            
            return {
                "total_chunks": total_chunks,
                "total_tokens": total_tokens,
                "avg_chunk_size": avg_chunk_size,
                "language_distribution": languages,
                "embedding_success": embedding_success,
                "sample_query_result": query_result.get("results", [])[:1] if query_result else [],
                "recommendations": []
            }
            
        except Exception as e:
            logger.error(f"Sanity check failed: {str(e)}")
            return {
                "error": str(e),
                "total_chunks": len(chunks),
                "recommendations": ["Manual review recommended due to sanity check failure"]
            }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on ingestion results."""
        recommendations = []
        
        stats = self.ingestion_stats
        
        if stats["files_failed"] > 0:
            recommendations.append(
                f"Review {stats['files_failed']} failed files. Consider different file formats or manual cleaning."
            )
        
        if stats["chunks_created"] == 0:
            recommendations.append(
                "No chunks created. Check file content and supported formats."
            )
        elif stats["chunks_created"] < 10:
            recommendations.append(
                "Very few chunks created. Consider adding more comprehensive documentation."
            )
        
        if stats["embeddings_stored"] < stats["chunks_created"]:
            missing = stats["chunks_created"] - stats["embeddings_stored"]
            recommendations.append(
                f"{missing} chunks failed embedding storage. Check API connectivity and quotas."
            )
        
        if len(stats["errors"]) > 0:
            recommendations.append(
                "Review error log for specific issues that need attention."
            )
        
        # Success case recommendations
        if stats["files_processed"] > 0 and len(recommendations) == 0:
            recommendations.append("Ingestion successful. Ready for quality analysis phase.")
        
        return recommendations
    
    def _chunk_to_dict(self, chunk: TextChunk) -> Dict[str, Any]:
        """Convert TextChunk to dictionary for JSON serialization."""
        return {
            "id": chunk.id,
            "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "source": chunk.source,
            "metadata": chunk.metadata
        }
    
    def _create_error_result(self, error_message: str, additional_data: Dict = None) -> Dict[str, Any]:
        """Create standardized error result."""
        result = {
            "success": False,
            "error": error_message,
            "stats": self.ingestion_stats
        }
        
        if additional_data:
            result.update(additional_data)
        
        return result
    
    def _reset_stats(self):
        """Reset ingestion statistics."""
        self.ingestion_stats = {
            "files_processed": 0,
            "files_failed": 0,
            "chunks_created": 0,
            "embeddings_stored": 0,
            "errors": []
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(DocumentParser.SUPPORTED_EXTENSIONS)
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connections to external services."""
        results = {}
        
        # Test OpenAI connection
        try:
            test_result = await self.vector_manager.embedding_generator.generate_embedding("test")
            results["openai"] = {
                "status": "success" if test_result.success else "failed",
                "error": test_result.error
            }
        except Exception as e:
            results["openai"] = {
                "status": "failed",
                "error": str(e)
            }
        
        # Test Pinecone connection
        try:
            stats = self.vector_manager.pinecone_manager.get_index_stats()
            results["pinecone"] = {
                "status": "success" if stats["success"] else "failed",
                "error": stats.get("error"),
                "index_info": stats if stats["success"] else None
            }
        except Exception as e:
            results["pinecone"] = {
                "status": "failed",
                "error": str(e)
            }
        
        return results


# Convenience function for external use
async def ingest_files(file_paths: List[str], 
                      namespace: str = None, 
                      goal: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function to ingest files.
    
    Args:
        file_paths: List of file paths to process
        namespace: Optional namespace (auto-generated if not provided)
        goal: Optional goal context
        
    Returns:
        Ingestion result
    """
    agent = DataIngestionAgent()
    return await agent.ingest(file_paths, namespace, goal)
