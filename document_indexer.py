from typing import List, Dict, Any, Optional
import os
import hashlib
import json
from pathlib import Path
from datetime import datetime
try:
    from langchain.indexes import SQLRecordManager, index
except ImportError:
    # Fallback for newer versions
    from langchain_community.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config import Config

class DocumentIndexer:
    """Manages document indexing with automatic sync for add/update/delete operations"""
    
    def __init__(self, namespace: str = "multimodal_rag"):
        self.namespace = namespace
        self.index_db_path = f"{Config.PERSIST_DIRECTORY}/index_records.db"
        
        # Initialize record manager for tracking document changes
        self.record_manager = SQLRecordManager(
            namespace=self.namespace,
            db_url=f"sqlite:///{self.index_db_path}"
        )
        self.record_manager.create_schema()
        
        # Track file metadata
        self.metadata_path = Path(Config.PERSIST_DIRECTORY) / "file_metadata.json"
        self.file_metadata = self._load_file_metadata()
    
    def _load_file_metadata(self) -> Dict[str, Dict]:
        """Load file metadata from disk"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_file_metadata(self):
        """Save file metadata to disk"""
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.file_metadata, f, indent=2)
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file for change detection"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_file_stats(self, file_path: str) -> Dict[str, Any]:
        """Get file statistics"""
        stat = os.stat(file_path)
        return {
            "path": file_path,
            "hash": self._compute_file_hash(file_path),
            "size": stat.st_size,
            "modified_time": stat.st_mtime,
            "last_indexed": datetime.now().isoformat()
        }
    
    def is_file_modified(self, file_path: str) -> bool:
        """Check if file has been modified since last indexing"""
        if file_path not in self.file_metadata:
            return True
        
        current_hash = self._compute_file_hash(file_path)
        stored_hash = self.file_metadata[file_path].get("hash")
        
        return current_hash != stored_hash
    
    def is_file_new(self, file_path: str) -> bool:
        """Check if file is new (not in metadata)"""
        return file_path not in self.file_metadata
    
    def index_documents(
        self,
        documents: List[Document],
        vector_store: Chroma,
        file_path: str,
        cleanup: str = "incremental"
    ) -> Dict[str, Any]:
        """
        Index documents with automatic sync
        
        Args:
            documents: List of Document objects to index
            vector_store: Chroma vector store instance
            file_path: Source file path
            cleanup: "incremental" (update changed), "full" (replace all), or None
        
        Returns:
            Dictionary with indexing statistics
        """
        # Add source file to document metadata
        for doc in documents:
            doc.metadata["source_file"] = file_path
            doc.metadata["indexed_at"] = datetime.now().isoformat()
        
        # Use LangChain indexing API for automatic sync
        result = index(
            docs_source=documents,
            record_manager=self.record_manager,
            vector_store=vector_store,
            cleanup=cleanup,
            source_id_key="source_file"
        )
        
        # Update file metadata
        self.file_metadata[file_path] = self._get_file_stats(file_path)
        self._save_file_metadata()
        
        return {
            "num_added": result.get("num_added", 0),
            "num_updated": result.get("num_updated", 0),
            "num_deleted": result.get("num_deleted", 0),
            "num_skipped": result.get("num_skipped", 0)
        }
    
    def remove_file_from_index(self, file_path: str, vector_stores: Dict[str, Chroma]) -> int:
        """Remove all documents associated with a file from all vector stores"""
        total_deleted = 0
        
        for store_name, vector_store in vector_stores.items():
            try:
                # Get all documents from this file
                results = vector_store.get(where={"source_file": file_path})
                
                if results and results['ids']:
                    # Delete documents
                    vector_store.delete(ids=results['ids'])
                    total_deleted += len(results['ids'])
                    print(f"  Deleted {len(results['ids'])} chunks from {store_name} store")
            except Exception as e:
                print(f"  Error removing from {store_name} store: {e}")
        
        # Remove from metadata
        if file_path in self.file_metadata:
            del self.file_metadata[file_path]
            self._save_file_metadata()
        
        return total_deleted
    
    def sync_folder(
        self,
        folder_path: str,
        current_files: List[str]
    ) -> Dict[str, List[str]]:
        """
        Detect changes in a folder and categorize files
        
        Args:
            folder_path: Path to folder
            current_files: List of current file paths in folder
        
        Returns:
            Dictionary with 'new', 'modified', 'deleted', and 'unchanged' file lists
        """
        current_set = set(current_files)
        indexed_set = set(self.file_metadata.keys())
        
        # Find new files
        new_files = [f for f in current_set - indexed_set]
        
        # Find deleted files
        deleted_files = [f for f in indexed_set if f.startswith(folder_path) and f not in current_set]
        
        # Find modified and unchanged files
        modified_files = []
        unchanged_files = []
        
        for file_path in current_set & indexed_set:
            if self.is_file_modified(file_path):
                modified_files.append(file_path)
            else:
                unchanged_files.append(file_path)
        
        return {
            "new": new_files,
            "modified": modified_files,
            "deleted": deleted_files,
            "unchanged": unchanged_files
        }
    
    def get_indexed_files(self) -> List[Dict[str, Any]]:
        """Get list of all indexed files with metadata"""
        return [
            {
                "path": path,
                **metadata
            }
            for path, metadata in self.file_metadata.items()
        ]
    
    def get_indexing_stats(self) -> Dict[str, Any]:
        """Get overall indexing statistics"""
        total_files = len(self.file_metadata)
        total_size = sum(meta.get("size", 0) for meta in self.file_metadata.values())
        
        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "index_db_path": self.index_db_path,
            "metadata_path": str(self.metadata_path)
        }
