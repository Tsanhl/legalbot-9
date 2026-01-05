"""
Gemini Files Service for Legal AI
Uploads documents to Google's Gemini Files API for persistent cloud storage
Files persist for 48 hours and can be referenced in queries
"""
import os
import json
import time
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta

# Try to import the new google-genai library
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-genai library not available for Files API")

# File tracking storage
UPLOAD_TRACKING_FILE = os.path.join(os.path.dirname(__file__), '.gemini_files_cache.json')

class GeminiFilesService:
    """Service for managing documents in Gemini Files API"""
    
    def __init__(self, api_key: str):
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai library required for Files API")
        
        os.environ['GOOGLE_API_KEY'] = api_key
        self.client = genai.Client()
        self.uploaded_files: Dict[str, Dict] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cached file upload info"""
        try:
            if os.path.exists(UPLOAD_TRACKING_FILE):
                with open(UPLOAD_TRACKING_FILE, 'r') as f:
                    data = json.load(f)
                    # Filter out expired entries (older than 47 hours to be safe)
                    cutoff = datetime.now() - timedelta(hours=47)
                    self.uploaded_files = {
                        k: v for k, v in data.items()
                        if datetime.fromisoformat(v.get('uploaded_at', '2000-01-01')) > cutoff
                    }
        except Exception as e:
            print(f"Could not load file cache: {e}")
            self.uploaded_files = {}
    
    def _save_cache(self):
        """Save file upload info to cache"""
        try:
            with open(UPLOAD_TRACKING_FILE, 'w') as f:
                json.dump(self.uploaded_files, f, indent=2)
        except Exception as e:
            print(f"Could not save file cache: {e}")
    
    def upload_file(self, file_path: str, display_name: str = None) -> Optional[str]:
        """Upload a single file to Gemini Files API"""
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None
            
            # Check if already uploaded and still valid
            cache_key = file_path
            if cache_key in self.uploaded_files:
                cached = self.uploaded_files[cache_key]
                uploaded_at = datetime.fromisoformat(cached.get('uploaded_at', '2000-01-01'))
                if datetime.now() - uploaded_at < timedelta(hours=47):
                    # Still valid, verify it exists
                    try:
                        file_info = self.client.files.get(name=cached['name'])
                        if file_info.state.name == 'ACTIVE':
                            return cached['name']
                    except:
                        pass  # File expired or not found, re-upload
            
            # Upload the file
            display = display_name or os.path.basename(file_path)
            
            # Determine mime type
            ext = os.path.splitext(file_path)[1].lower()
            mime_types = {
                '.pdf': 'application/pdf',
                '.txt': 'text/plain',
                '.md': 'text/markdown',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.doc': 'application/msword',
            }
            mime_type = mime_types.get(ext, 'application/octet-stream')
            
            # Upload to Gemini
            with open(file_path, 'rb') as f:
                uploaded_file = self.client.files.upload(
                    file=f,
                    config={
                        'display_name': display,
                        'mime_type': mime_type
                    }
                )
            
            # Wait for processing if needed
            while uploaded_file.state.name == 'PROCESSING':
                time.sleep(1)
                uploaded_file = self.client.files.get(name=uploaded_file.name)
            
            if uploaded_file.state.name == 'ACTIVE':
                # Cache the upload info
                self.uploaded_files[cache_key] = {
                    'name': uploaded_file.name,
                    'display_name': display,
                    'uploaded_at': datetime.now().isoformat(),
                    'uri': uploaded_file.uri
                }
                self._save_cache()
                print(f"✅ Uploaded: {display}")
                return uploaded_file.name
            else:
                print(f"❌ Upload failed for {display}: {uploaded_file.state.name}")
                return None
                
        except Exception as e:
            print(f"Error uploading {file_path}: {e}")
            return None
    
    def upload_directory(self, directory_path: str, progress_callback=None) -> Dict[str, int]:
        """Upload all supported documents from a directory"""
        stats = {'uploaded': 0, 'skipped': 0, 'errors': 0, 'cached': 0}
        supported_extensions = {'.pdf', '.txt', '.md', '.docx', '.doc'}
        
        all_files = []
        for root, dirs, files in os.walk(directory_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                # Skip hidden/temp files
                if file.startswith('.') or file.startswith('~$'):
                    continue
                
                ext = os.path.splitext(file)[1].lower()
                if ext in supported_extensions:
                    all_files.append(os.path.join(root, file))
        
        total = len(all_files)
        for i, file_path in enumerate(all_files):
            rel_path = os.path.relpath(file_path, directory_path)
            
            # Check cache first
            if file_path in self.uploaded_files:
                cached = self.uploaded_files[file_path]
                uploaded_at = datetime.fromisoformat(cached.get('uploaded_at', '2000-01-01'))
                if datetime.now() - uploaded_at < timedelta(hours=47):
                    stats['cached'] += 1
                    if progress_callback:
                        progress_callback(i + 1, total, rel_path, 'cached')
                    continue
            
            result = self.upload_file(file_path, display_name=rel_path)
            
            if result:
                stats['uploaded'] += 1
            else:
                stats['errors'] += 1
            
            if progress_callback:
                progress_callback(i + 1, total, rel_path, 'uploaded' if result else 'error')
        
        return stats
    
    def get_uploaded_files(self) -> List[Dict]:
        """Get list of currently uploaded files"""
        valid_files = []
        for path, info in self.uploaded_files.items():
            uploaded_at = datetime.fromisoformat(info.get('uploaded_at', '2000-01-01'))
            if datetime.now() - uploaded_at < timedelta(hours=47):
                valid_files.append({
                    'path': path,
                    'name': info['name'],
                    'display_name': info['display_name'],
                    'uploaded_at': info['uploaded_at']
                })
        return valid_files
    
    def get_file_references(self, limit: int = 20) -> List[Any]:
        """Get file references to include in Gemini requests"""
        if not GENAI_AVAILABLE:
            return []
        
        valid_files = self.get_uploaded_files()[:limit]
        references = []
        
        for file_info in valid_files:
            try:
                # Create file reference for the API
                references.append(types.Part.from_uri(
                    file_uri=self.uploaded_files[file_info['path']]['uri'],
                    mime_type='application/pdf'  # Gemini handles this
                ))
            except Exception as e:
                print(f"Could not create reference for {file_info['display_name']}: {e}")
        
        return references
    
    def list_all_gemini_files(self) -> List[Dict]:
        """List all files currently in Gemini storage"""
        try:
            files = self.client.files.list()
            return [
                {
                    'name': f.name,
                    'display_name': f.display_name,
                    'state': f.state.name,
                    'size_bytes': f.size_bytes
                }
                for f in files
            ]
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    def delete_all_files(self):
        """Delete all uploaded files from Gemini"""
        try:
            files = self.client.files.list()
            for f in files:
                self.client.files.delete(name=f.name)
                print(f"Deleted: {f.display_name}")
            self.uploaded_files = {}
            self._save_cache()
        except Exception as e:
            print(f"Error deleting files: {e}")


# Global instance
_files_service: Optional[GeminiFilesService] = None

def get_files_service(api_key: str) -> Optional[GeminiFilesService]:
    """Get or create the files service instance"""
    global _files_service
    if _files_service is None and GENAI_AVAILABLE:
        _files_service = GeminiFilesService(api_key)
    return _files_service

def upload_law_resources(api_key: str, resources_path: str, progress_callback=None) -> Dict[str, int]:
    """Upload all law resources to Gemini Files API"""
    service = get_files_service(api_key)
    if service:
        return service.upload_directory(resources_path, progress_callback)
    return {'error': 'Files service not available'}
