"""Text chunking utilities with semantic and rule-based approaches."""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib
import tiktoken

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    id: str
    text: str
    start_index: int
    end_index: int
    chunk_index: int
    source: str
    metadata: Dict
    token_count: int
    hash: str


class SemanticChunker:
    """Handles intelligent text chunking with semantic awareness."""
    
    def __init__(self, 
                 target_tokens: int = None,
                 overlap_tokens: int = None,
                 model: str = "gpt-4"):
        self.target_tokens = target_tokens or config.DEFAULT_CHUNK_SIZE
        self.overlap_tokens = overlap_tokens or config.DEFAULT_CHUNK_OVERLAP
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to a default tokenizer
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Sentence boundaries and section markers
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        self.paragraph_breaks = re.compile(r'\n\s*\n')
        self.section_headers = re.compile(r'\n\s*(?:[A-Z][A-Z\s]+|[\d]+\.?\s+[A-Z])')
    
    def chunk_text(self, text: str, source: str = "", metadata: Dict = None) -> List[TextChunk]:
        """
        Chunk text using semantic boundaries with overlap.
        
        Args:
            text: The text to chunk
            source: Source identifier (filename, etc.)
            metadata: Additional metadata to include
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Clean and normalize text
        normalized_text = self._normalize_text(text)
        
        # Try semantic chunking first
        semantic_chunks = self._semantic_chunk(normalized_text)
        
        # If semantic chunking fails or produces poor results, fall back to rule-based
        if not semantic_chunks or self._needs_rule_based_fallback(semantic_chunks):
            logger.info("Falling back to rule-based chunking")
            semantic_chunks = self._rule_based_chunk(normalized_text)
        
        # Create TextChunk objects with overlap
        for i, (chunk_text, start_idx, end_idx) in enumerate(semantic_chunks):
            # Add overlap from previous chunk if not the first chunk
            if i > 0 and self.overlap_tokens > 0:
                overlap_text = self._get_overlap_text(
                    semantic_chunks[i-1][0], 
                    chunk_text,
                    self.overlap_tokens
                )
                chunk_text = overlap_text + chunk_text
                start_idx -= len(overlap_text)
            
            # Create chunk
            chunk = self._create_chunk(
                text=chunk_text,
                start_index=start_idx,
                end_index=end_idx,
                chunk_index=i,
                source=source,
                metadata=metadata
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better chunking."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        return text.strip()
    
    def _semantic_chunk(self, text: str) -> List[Tuple[str, int, int]]:
        """Chunk text using semantic boundaries (paragraphs, sections)."""
        chunks = []
        
        # First, try to split by major sections
        sections = self._split_by_sections(text)
        
        for section_text, section_start in sections:
            # If section is small enough, keep as one chunk
            if self._count_tokens(section_text) <= self.target_tokens + 100:
                chunks.append((section_text, section_start, section_start + len(section_text)))
                continue
            
            # Otherwise, split section by paragraphs
            paragraphs = self._split_by_paragraphs(section_text, section_start)
            
            current_chunk = ""
            current_start = None
            
            for para_text, para_start in paragraphs:
                # Check if adding this paragraph would exceed target
                potential_chunk = current_chunk + "\n\n" + para_text if current_chunk else para_text
                
                if self._count_tokens(potential_chunk) <= self.target_tokens + 100:
                    # Add paragraph to current chunk
                    current_chunk = potential_chunk
                    if current_start is None:
                        current_start = para_start
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append((
                            current_chunk, 
                            current_start, 
                            current_start + len(current_chunk)
                        ))
                    
                    current_chunk = para_text
                    current_start = para_start
            
            # Don't forget the last chunk
            if current_chunk:
                chunks.append((
                    current_chunk,
                    current_start,
                    current_start + len(current_chunk)
                ))
        
        return chunks
    
    def _rule_based_chunk(self, text: str) -> List[Tuple[str, int, int]]:
        """Fallback rule-based chunking by token count."""
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Find the end position for this chunk
            end_pos = self._find_chunk_boundary(
                text, 
                current_pos, 
                self.target_tokens
            )
            
            chunk_text = text[current_pos:end_pos].strip()
            if chunk_text:
                chunks.append((chunk_text, current_pos, end_pos))
            
            # Move to next position (with some overlap consideration)
            overlap_offset = max(0, end_pos - current_pos - self.overlap_tokens)
            current_pos = current_pos + overlap_offset
            
            # Safety check to avoid infinite loop
            if current_pos >= end_pos:
                current_pos = end_pos
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[Tuple[str, int]]:
        """Split text by section headers."""
        sections = []
        
        # Find section boundaries
        section_matches = list(self.section_headers.finditer(text))
        
        if not section_matches:
            # No sections found, return entire text
            return [(text, 0)]
        
        # First section (before first header)
        if section_matches[0].start() > 0:
            sections.append((text[:section_matches[0].start()].strip(), 0))
        
        # Sections with headers
        for i, match in enumerate(section_matches):
            start = match.start()
            end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append((section_text, start))
        
        return sections
    
    def _split_by_paragraphs(self, text: str, offset: int = 0) -> List[Tuple[str, int]]:
        """Split text by paragraph boundaries."""
        paragraphs = []
        
        # Split by paragraph breaks
        para_parts = self.paragraph_breaks.split(text)
        current_pos = offset
        
        for part in para_parts:
            part = part.strip()
            if part:
                paragraphs.append((part, current_pos))
            current_pos += len(part) + 2  # Account for paragraph break
        
        return paragraphs
    
    def _find_chunk_boundary(self, text: str, start_pos: int, target_tokens: int) -> int:
        """Find a good boundary for chunking near the target token count."""
        # Estimate character position based on average tokens per character
        avg_chars_per_token = 4  # Rough estimate
        target_chars = target_tokens * avg_chars_per_token
        
        estimated_end = min(start_pos + target_chars, len(text))
        
        # Fine-tune by actual token count
        current_end = start_pos + target_chars // 2
        
        while current_end < len(text):
            chunk_text = text[start_pos:current_end]
            token_count = self._count_tokens(chunk_text)
            
            if token_count >= target_tokens:
                # Find a good breaking point (sentence end, paragraph break)
                break_pos = self._find_break_point(text, current_end, start_pos + target_chars)
                return break_pos if break_pos > start_pos else current_end
            
            current_end = min(current_end + 100, len(text))
        
        return len(text)
    
    def _find_break_point(self, text: str, preferred_pos: int, max_pos: int) -> int:
        """Find the best breaking point near the preferred position."""
        search_window = min(200, max_pos - preferred_pos)
        
        # Look for sentence endings first
        search_text = text[preferred_pos:preferred_pos + search_window]
        sentence_matches = list(self.sentence_endings.finditer(search_text))
        
        if sentence_matches:
            return preferred_pos + sentence_matches[0].end()
        
        # Look for paragraph breaks
        para_matches = list(self.paragraph_breaks.finditer(search_text))
        if para_matches:
            return preferred_pos + para_matches[0].end()
        
        # Look for other natural breaks (commas, semicolons)
        for char in ['\n', ';', ',', ' ']:
            pos = search_text.rfind(char)
            if pos > len(search_text) // 2:  # Don't break too early
                return preferred_pos + pos + 1
        
        # No good break point found
        return preferred_pos + search_window
    
    def _get_overlap_text(self, prev_chunk: str, current_chunk: str, overlap_tokens: int) -> str:
        """Get overlap text from the previous chunk."""
        prev_tokens = self.tokenizer.encode(prev_chunk)
        if len(prev_tokens) <= overlap_tokens:
            return prev_chunk + " "
        
        # Take the last N tokens from previous chunk
        overlap_token_ids = prev_tokens[-overlap_tokens:]
        overlap_text = self.tokenizer.decode(overlap_token_ids)
        
        # Find a good starting point in the overlap
        sentences = self.sentence_endings.split(overlap_text)
        if len(sentences) > 1:
            # Start from the beginning of the last complete sentence
            overlap_text = " ".join(sentences[-2:])
        
        return overlap_text + " " if overlap_text else ""
    
    def _create_chunk(self, 
                     text: str, 
                     start_index: int, 
                     end_index: int,
                     chunk_index: int,
                     source: str,
                     metadata: Dict) -> TextChunk:
        """Create a TextChunk object with metadata."""
        token_count = self._count_tokens(text)
        chunk_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Generate unique chunk ID
        chunk_id = f"{source}_{chunk_index}_{chunk_hash[:8]}"
        
        # Enhance metadata
        enhanced_metadata = {
            **metadata,
            "chunk_type": self._classify_chunk_type(text),
            "has_code": self._contains_code(text),
            "has_table": self._contains_table(text),
            "sentence_count": len(self.sentence_endings.split(text)),
            "paragraph_count": len(self.paragraph_breaks.split(text)),
        }
        
        return TextChunk(
            id=chunk_id,
            text=text,
            start_index=start_index,
            end_index=end_index,
            chunk_index=chunk_index,
            source=source,
            metadata=enhanced_metadata,
            token_count=token_count,
            hash=chunk_hash
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback to rough estimation
            return len(text.split()) * 1.3  # Rough conversion
    
    def _needs_rule_based_fallback(self, semantic_chunks: List) -> bool:
        """Determine if we need to fall back to rule-based chunking."""
        if not semantic_chunks:
            return True
        
        # Check if chunks are too large or too small
        for chunk_text, _, _ in semantic_chunks:
            token_count = self._count_tokens(chunk_text)
            if token_count > self.target_tokens * 2 or token_count < self.target_tokens * 0.3:
                return True
        
        return False
    
    def _classify_chunk_type(self, text: str) -> str:
        """Classify the type of content in the chunk."""
        text_lower = text.lower()
        
        # Check for different content types
        if re.search(r'\|.*\|.*\|', text) or 'table' in text_lower:
            return "table"
        elif re.search(r'```|def |class |import |function', text):
            return "code"
        elif re.search(r'^\s*\d+\.|\*\s+|-\s+', text, re.MULTILINE):
            return "list"
        elif re.search(r'^#{1,6}\s+', text, re.MULTILINE):
            return "heading"
        else:
            return "text"
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code snippets."""
        code_patterns = [
            r'```',  # Code blocks
            r'\bdef\s+\w+\(',  # Python functions
            r'\bclass\s+\w+',  # Class definitions
            r'\bimport\s+\w+',  # Import statements
            r'{\s*\w+\s*:\s*\w+\s*}',  # JSON-like objects
            r'<\w+.*?>.*?</\w+>',  # HTML/XML tags
        ]
        
        return any(re.search(pattern, text) for pattern in code_patterns)
    
    def _contains_table(self, text: str) -> bool:
        """Check if text contains table-like structures."""
        table_patterns = [
            r'\|.*\|.*\|',  # Pipe-separated tables
            r'\t.*\t.*\t',  # Tab-separated tables
            r'^\s*\w+\s+\w+\s+\w+\s*$',  # Space-separated columns
        ]
        
        return any(re.search(pattern, text, re.MULTILINE) for pattern in table_patterns)


def chunk_document(text: str, source: str = "", metadata: Dict = None, **kwargs) -> List[TextChunk]:
    """Convenience function to chunk a document."""
    chunker = SemanticChunker(**kwargs)
    return chunker.chunk_text(text, source, metadata)
