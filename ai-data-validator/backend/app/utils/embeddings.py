"""Embeddings generation and vector store utilities."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import time
import asyncio

from openai import OpenAI
import pinecone

from ..config import config
from .chunker import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    success: bool
    embedding: Optional[List[float]] = None
    error: Optional[str] = None
    token_count: int = 0


class EmbeddingGenerator:
    """Handles text embedding generation using OpenAI."""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.MODELS["embedding"]
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            return EmbeddingResult(success=False, error="Empty text")
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    encoding_format="float"
                )
                
                embedding = response.data[0].embedding
                token_count = response.usage.total_tokens
                
                return EmbeddingResult(
                    success=True,
                    embedding=embedding,
                    token_count=token_count
                )
                
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                else:
                    return EmbeddingResult(
                        success=False,
                        error=f"Failed after {self.max_retries} attempts: {str(e)}"
                    )
        
        return EmbeddingResult(success=False, error="Unknown error")
    
    async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts in batches."""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = []
            
            # Process batch items concurrently (but respect rate limits)
            tasks = [self.generate_embedding(text) for text in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in batch results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    batch_results[j] = EmbeddingResult(
                        success=False,
                        error=f"Batch processing error: {str(result)}"
                    )
            
            results.extend(batch_results)
            
            # Small delay between batches to respect rate limits
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return results


class PineconeManager:
    """Manages Pinecone vector database operations."""
    
    def __init__(self):
        self.api_key = config.PINECONE_API_KEY
        self.environment = config.PINECONE_ENVIRONMENT
        self.index_name = config.PINECONE_INDEX_NAME
        self.index = None
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index."""
        try:
            pinecone.init(
                api_key=self.api_key,
                environment=self.environment
            )
            
            # Check if index exists, create if not
            if self.index_name not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=3072,  # OpenAI text-embedding-3-large dimension
                    metric="cosine"
                )
                # Wait for index to be ready
                time.sleep(60)
            
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def upsert_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]], namespace: str) -> Dict[str, Any]:
        """Upsert text chunks with embeddings to Pinecone."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        try:
            # Prepare vectors for upsert
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                vector_data = {
                    "id": chunk.id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk.text[:1000],  # Limit text size in metadata
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                        "token_count": chunk.token_count,
                        "hash": chunk.hash,
                        **chunk.metadata
                    }
                }
                vectors.append(vector_data)
            
            # Upsert in batches
            batch_size = 100
            upserted_count = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                response = self.index.upsert(
                    vectors=batch,
                    namespace=namespace
                )
                upserted_count += response.upserted_count
                logger.info(f"Upserted batch {i//batch_size + 1}: {response.upserted_count} vectors")
            
            return {
                "success": True,
                "upserted_count": upserted_count,
                "namespace": namespace
            }
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "upserted_count": 0
            }
    
    def query_similar(self, 
                     query_embedding: List[float], 
                     namespace: str, 
                     top_k: int = None,
                     filter_dict: Dict = None) -> Dict[str, Any]:
        """Query for similar vectors."""
        top_k = top_k or config.RETRIEVAL_TOP_K
        
        try:
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict,
                include_metadata=True,
                include_values=False
            )
            
            results = []
            for match in response.matches:
                results.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                })
            
            return {
                "success": True,
                "results": results,
                "query_info": {
                    "top_k": top_k,
                    "namespace": namespace,
                    "filter": filter_dict
                }
            }
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    def delete_namespace(self, namespace: str) -> bool:
        """Delete all vectors in a namespace."""
        try:
            # Get all vector IDs in the namespace
            # Note: This is a simplified approach - for large namespaces, 
            # you'd want to implement pagination
            query_response = self.index.query(
                vector=[0.0] * 3072,  # Dummy vector
                top_k=10000,  # Maximum allowed
                namespace=namespace,
                include_metadata=False,
                include_values=False
            )
            
            # Extract IDs
            ids_to_delete = [match.id for match in query_response.matches]
            
            if ids_to_delete:
                self.index.delete(ids=ids_to_delete, namespace=namespace)
                logger.info(f"Deleted {len(ids_to_delete)} vectors from namespace {namespace}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete namespace {namespace}: {str(e)}")
            return False
    
    def get_index_stats(self, namespace: str = None) -> Dict[str, Any]:
        """Get statistics about the index."""
        try:
            stats = self.index.describe_index_stats()
            
            if namespace and namespace in stats.namespaces:
                return {
                    "success": True,
                    "total_vectors": stats.total_vector_count,
                    "namespace_vectors": stats.namespaces[namespace].vector_count,
                    "dimension": stats.dimension,
                    "index_fullness": stats.index_fullness
                }
            else:
                return {
                    "success": True,
                    "total_vectors": stats.total_vector_count,
                    "dimension": stats.dimension,
                    "index_fullness": stats.index_fullness,
                    "namespaces": list(stats.namespaces.keys()) if stats.namespaces else []
                }
                
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }


class VectorStoreManager:
    """High-level manager combining embedding generation and vector storage."""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.pinecone_manager = PineconeManager()
    
    async def process_and_store_chunks(self, chunks: List[TextChunk], namespace: str) -> Dict[str, Any]:
        """Process chunks: generate embeddings and store in vector database."""
        if not chunks:
            return {"success": False, "error": "No chunks provided"}
        
        logger.info(f"Processing {len(chunks)} chunks for namespace {namespace}")
        
        # Generate embeddings for all chunks
        texts = [chunk.text for chunk in chunks]
        embedding_results = await self.embedding_generator.generate_embeddings_batch(texts)
        
        # Filter successful embeddings
        successful_chunks = []
        successful_embeddings = []
        failed_count = 0
        
        for chunk, embedding_result in zip(chunks, embedding_results):
            if embedding_result.success:
                successful_chunks.append(chunk)
                successful_embeddings.append(embedding_result.embedding)
            else:
                failed_count += 1
                logger.warning(f"Failed to generate embedding for chunk {chunk.id}: {embedding_result.error}")
        
        if not successful_chunks:
            return {
                "success": False,
                "error": "No successful embeddings generated",
                "failed_count": failed_count
            }
        
        # Store in Pinecone
        storage_result = self.pinecone_manager.upsert_chunks(
            successful_chunks, 
            successful_embeddings, 
            namespace
        )
        
        return {
            "success": storage_result["success"],
            "processed_count": len(successful_chunks),
            "failed_count": failed_count,
            "upserted_count": storage_result.get("upserted_count", 0),
            "namespace": namespace,
            "error": storage_result.get("error")
        }
    
    async def query_knowledge_base(self, query: str, namespace: str, top_k: int = None) -> Dict[str, Any]:
        """Query the knowledge base with a text query."""
        # Generate embedding for query
        query_embedding_result = await self.embedding_generator.generate_embedding(query)
        
        if not query_embedding_result.success:
            return {
                "success": False,
                "error": f"Failed to generate query embedding: {query_embedding_result.error}"
            }
        
        # Query vector database
        search_result = self.pinecone_manager.query_similar(
            query_embedding_result.embedding,
            namespace,
            top_k
        )
        
        if search_result["success"]:
            # Enhance results with relevance information
            enhanced_results = []
            for result in search_result["results"]:
                enhanced_results.append({
                    "chunk_id": result["id"],
                    "relevance_score": result["score"],
                    "text": result["metadata"].get("text", ""),
                    "source": result["metadata"].get("source", ""),
                    "chunk_index": result["metadata"].get("chunk_index", 0),
                    "metadata": result["metadata"]
                })
            
            return {
                "success": True,
                "query": query,
                "results": enhanced_results,
                "result_count": len(enhanced_results)
            }
        else:
            return search_result
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {str(e)}")
            return 0.0
    
    def cleanup_namespace(self, namespace: str) -> bool:
        """Clean up a namespace (delete all vectors)."""
        return self.pinecone_manager.delete_namespace(namespace)


# Convenience functions
async def generate_embeddings(texts: List[str]) -> List[EmbeddingResult]:
    """Generate embeddings for a list of texts."""
    generator = EmbeddingGenerator()
    return await generator.generate_embeddings_batch(texts)


async def store_document_chunks(chunks: List[TextChunk], namespace: str) -> Dict[str, Any]:
    """Store document chunks in vector database."""
    manager = VectorStoreManager()
    return await manager.process_and_store_chunks(chunks, namespace)
