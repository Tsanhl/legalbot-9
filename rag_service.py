"""
RAG (Retrieval-Augmented Generation) Service for Legal AI
Extracts text from documents, creates embeddings, and retrieves relevant content
"""
import os
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings

# PDF and DOCX extraction
import fitz  # PyMuPDF
from docx import Document

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    id: str
    text: str
    source_file: str
    category: str
    chunk_index: int

class RAGService:
    """Service for document retrieval using embeddings"""
    
    def __init__(self, persist_directory: str = None):
        """Initialize the RAG service"""
        if persist_directory is None:
            persist_directory = os.path.join(os.path.dirname(__file__), 'chroma_db')
        
        self.persist_directory = persist_directory
        self.chunk_size = 1500  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="law_resources",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.is_indexed = self.collection.count() > 0
        print(f"📚 RAG Service initialized. Documents indexed: {self.collection.count()}")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"Error extracting PDF {file_path}: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from a DOCX file"""
        try:
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        except Exception as e:
            print(f"Error extracting DOCX {file_path}: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from a TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from a file based on its extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        elif ext in ['.txt', '.md']:
            return self.extract_text_from_txt(file_path)
        else:
            return ""
    
    def chunk_text(self, text: str, source_file: str, category: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks"""
        if not text or len(text) < 100:
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within last 200 chars
                for i in range(min(200, end - start)):
                    if text[end - i - 1] in '.!?\n':
                        end = end - i
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text and len(chunk_text) > 50:
                # Create unique ID
                chunk_id = hashlib.md5(f"{source_file}_{chunk_index}".encode()).hexdigest()
                
                chunks.append(DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    source_file=source_file,
                    category=category,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def index_documents(self, resources_path: str, progress_callback=None) -> Dict[str, int]:
        """Index all documents in the resources folder"""
        stats = {"processed": 0, "chunks": 0, "errors": 0, "skipped": 0}
        
        # Clear existing collection
        try:
            self.client.delete_collection("law_resources")
            self.collection = self.client.create_collection(
                name="law_resources",
                metadata={"hnsw:space": "cosine"}
            )
        except:
            pass
        
        all_chunks = []
        
        # Walk through all files
        for root, dirs, files in os.walk(resources_path):
            # Skip temp files
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                # Skip temp files and hidden files
                if file.startswith('~$') or file.startswith('.'):
                    stats["skipped"] += 1
                    continue
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, resources_path)
                category = rel_path.split(os.sep)[0] if os.sep in rel_path else "General"
                
                # Extract text
                text = self.extract_text(file_path)
                
                if text:
                    chunks = self.chunk_text(text, rel_path, category)
                    all_chunks.extend(chunks)
                    stats["processed"] += 1
                    stats["chunks"] += len(chunks)
                else:
                    stats["errors"] += 1
                
                if progress_callback:
                    progress_callback(stats["processed"], file)
        
        # Add chunks to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            
            self.collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[{
                    "source_file": c.source_file,
                    "category": c.category,
                    "chunk_index": c.chunk_index
                } for c in batch]
            )
        
        self.is_indexed = True
        print(f"✅ Indexing complete: {stats['processed']} documents, {stats['chunks']} chunks")
        
        return stats
    
    def search(self, query: str, n_results: int = 5, category_filter: str = None) -> List[Dict]:
        """Search for relevant document chunks"""
        if not self.is_indexed or self.collection.count() == 0:
            return []
        
        # Build where clause for category filter
        where = None
        if category_filter:
            where = {"category": {"$eq": category_filter}}
        
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        # Format results
        formatted = []
        if results and results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results['distances'] else 0
                
                formatted.append({
                    "text": doc,
                    "source_file": metadata.get('source_file', 'Unknown'),
                    "category": metadata.get('category', 'Unknown'),
                    "relevance_score": 1 - distance  # Convert distance to similarity
                })
        
        return formatted
    
    def get_context_for_query(self, query: str, max_chunks: int = 8) -> str:
        """Get relevant context for a query to include in the prompt"""
        results = self.search(query, n_results=max_chunks)
        
        if not results:
            return ""
        
        context_parts = [
            "\n================================================================================",
            "RELEVANT EXCERPTS FROM YOUR LAW RESOURCES KNOWLEDGE BASE:",
            "================================================================================\n"
        ]
        
        for i, result in enumerate(results, 1):
            relevance_pct = int(result['relevance_score'] * 100)
            context_parts.append(f"""
--- Document {i}: {result['source_file']} (Relevance: {relevance_pct}%) ---
Category: {result['category']}

{result['text'][:1200]}{'...' if len(result['text']) > 1200 else ''}
""")
        
        context_parts.append("""
================================================================================
INSTRUCTION: Use the above excerpts as PRIMARY sources. Cite them using OSCOLA format.
If the excerpts don't fully answer the question, supplement with your trained knowledge.
================================================================================
""")
        
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """Get statistics about the indexed documents"""
        return {
            "total_chunks": self.collection.count(),
            "is_indexed": self.is_indexed,
            "persist_directory": self.persist_directory
        }


# Global instance
_rag_service: Optional[RAGService] = None

def get_rag_service() -> RAGService:
    """Get or create the RAG service instance"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

def get_relevant_context(query: str, max_chunks: int = 8) -> str:
    """Convenience function to get relevant context for a query"""
    service = get_rag_service()
    return service.get_context_for_query(query, max_chunks)
