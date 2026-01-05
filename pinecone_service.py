"""
Pinecone RAG Service for Legal AI
Cloud-based vector database that persists permanently (free tier available)
Replaces local ChromaDB for production deployment
"""
import os
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# PDF and DOCX extraction
import fitz  # PyMuPDF
from docx import Document

# Pinecone
try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("⚠️ Pinecone not installed. Run: pip install pinecone")

# We'll use a simple embedding approach - either Gemini embeddings or sentence-transformers
try:
    from google import genai
    GEMINI_EMBEDDINGS = True
except ImportError:
    GEMINI_EMBEDDINGS = False

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    id: str
    text: str
    source_file: str
    category: str
    chunk_index: int

class PineconeRAGService:
    """Service for document retrieval using Pinecone cloud vector database"""
    
    def __init__(self, pinecone_api_key: str, gemini_api_key: str = None, index_name: str = "legal-ai-docs"):
        """Initialize the Pinecone RAG service"""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone library not installed. Run: pip install pinecone")
        
        self.pinecone_api_key = pinecone_api_key
        self.gemini_api_key = gemini_api_key
        self.index_name = index_name
        self.chunk_size = 1500  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks
        self.embedding_dimension = 768  # Dimension for embeddings
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Initialize Gemini client for embeddings if available
        if gemini_api_key and GEMINI_EMBEDDINGS:
            os.environ['GOOGLE_API_KEY'] = gemini_api_key
            self.genai_client = genai.Client()
        else:
            self.genai_client = None
        
        # Create or get index
        self._ensure_index()
        
        print(f"📚 Pinecone RAG Service initialized. Index: {self.index_name}")
    
    def _ensure_index(self):
        """Ensure the Pinecone index exists"""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
        
        self.index = self.pc.Index(self.index_name)
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using Gemini"""
        if self.genai_client:
            try:
                response = self.genai_client.models.embed_content(
                    model="text-embedding-004",
                    contents=text[:8000]  # Limit text length
                )
                return response.embeddings[0].values
            except Exception as e:
                print(f"Embedding error: {e}")
                # Fallback to simple hash-based embedding (not ideal but works)
                return self._simple_embedding(text)
        else:
            return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str) -> List[float]:
        """Simple fallback embedding (hash-based, not semantic)"""
        import random
        # Create deterministic random embedding from text hash
        random.seed(hash(text))
        return [random.uniform(-1, 1) for _ in range(self.embedding_dimension)]
    
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
        """Index all documents in the resources folder to Pinecone"""
        stats = {"processed": 0, "chunks": 0, "errors": 0, "skipped": 0}
        
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
        
        # Upload to Pinecone in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            
            vectors = []
            for chunk in batch:
                embedding = self._get_embedding(chunk.text)
                vectors.append({
                    "id": chunk.id,
                    "values": embedding,
                    "metadata": {
                        "text": chunk.text[:1000],  # Store first 1000 chars in metadata
                        "source_file": chunk.source_file,
                        "category": chunk.category,
                        "chunk_index": chunk.chunk_index,
                        "full_text": chunk.text  # Store full text
                    }
                })
            
            self.index.upsert(vectors=vectors)
            
            if progress_callback:
                progress_callback(i + len(batch), f"Uploading batch {i // batch_size + 1}")
        
        print(f"✅ Indexing complete: {stats['processed']} documents, {stats['chunks']} chunks")
        return stats
    
    def search(self, query: str, n_results: int = 5, category_filter: str = None) -> List[Dict]:
        """Search for relevant document chunks"""
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Build filter if category specified
        filter_dict = None
        if category_filter:
            filter_dict = {"category": {"$eq": category_filter}}
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Format results
        formatted = []
        for match in results.matches:
            metadata = match.metadata or {}
            formatted.append({
                "text": metadata.get('full_text', metadata.get('text', '')),
                "source_file": metadata.get('source_file', 'Unknown'),
                "category": metadata.get('category', 'Unknown'),
                "relevance_score": match.score
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
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "is_indexed": stats.total_vector_count > 0,
                "index_name": self.index_name
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {
                "total_vectors": 0,
                "is_indexed": False,
                "index_name": self.index_name
            }
    
    def clear_index(self):
        """Clear all vectors from the index"""
        try:
            self.index.delete(delete_all=True)
            print("✅ Index cleared")
        except Exception as e:
            print(f"Error clearing index: {e}")


# Global instance
_pinecone_service: Optional[PineconeRAGService] = None

def get_pinecone_service(pinecone_api_key: str = None, gemini_api_key: str = None) -> Optional[PineconeRAGService]:
    """Get or create the Pinecone RAG service instance"""
    global _pinecone_service
    
    if _pinecone_service is None and pinecone_api_key:
        _pinecone_service = PineconeRAGService(pinecone_api_key, gemini_api_key)
    
    return _pinecone_service

def get_pinecone_context(query: str, max_chunks: int = 8) -> str:
    """Convenience function to get relevant context for a query"""
    if _pinecone_service:
        return _pinecone_service.get_context_for_query(query, max_chunks)
    return ""
