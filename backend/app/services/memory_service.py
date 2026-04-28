import os
import logging

logger = logging.getLogger(__name__)

try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    logger.warning("chromadb not installed — LTM will use in-memory fallback")


class MemoryService:
    def __init__(self):
        self.collection = None
        if not _CHROMA_AVAILABLE:
            return
        # Store the vector db in the backend directory
        db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
        try:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection_name = "workspace_memory"
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            self.collection = None


    def store_memory(self, workspace_id: str, content: str, metadata: dict = None):
        """Store a new memory fragment for a workspace."""
        if not self.collection:
            return
            
        import uuid
        doc_id = str(uuid.uuid4())
        
        if metadata is None:
            metadata = {}
        metadata["workspace_id"] = workspace_id
        
        try:
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
            logger.info(f"Stored memory {doc_id} for workspace {workspace_id}")
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")

    def query_memory(self, query: str, workspace_id: str = None, n_results: int = 3):
        """Query memory for semantically similar past experiences."""
        if not self.collection:
            return []
        try:
            # Clamp n_results to the actual count to avoid ChromaDB error when < n_results docs exist
            doc_count = self.collection.count()
            if doc_count == 0:
                return []
            effective_n = min(n_results, doc_count)
            results = self.collection.query(
                query_texts=[query],
                n_results=effective_n,
            )
            if results and results.get("documents") and len(results["documents"]) > 0:
                return results["documents"][0]  # List of top-k matching strings
            return []
        except Exception as e:
            logger.error(f"Failed to query memory: {e}")
            return []

memory_service = MemoryService()
