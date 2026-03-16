# base.py

from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar, Optional, List

import numpy as np

# Represents a raw chunk of text from a source document.
TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

# Generic TypeVar for Key-Value storage.
T = TypeVar("T")

@dataclass
class QueryParam:
    """
    Defines parameters for a retrieval query, adapted for a dynamic, layered memory system.
    """
    # Specifies which memory layers to search. If None, searches all available memory.
    
    # --- Context Truncation Parameters ---
    # Max tokens for the final list of source text chunks.
    max_token_for_text_unit: int = 2000
    # Max tokens for the final list of hyperedge/memory contexts.
    max_token_for_relations: int = 2000
    # Max tokens for the final list of entity descriptions.
    max_token_for_entities: int = 2000


@dataclass
class StorageNameSpace:
    """Base class providing a namespace and global configuration to all storage implementations."""
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        """Commit the storage operations after an indexing or insertion batch."""
        pass

    async def query_done_callback(self):
        """Commit any storage operations after a query (e.g., updating access times)."""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    """Abstract base class for vector database storage."""
    embedding_func = None
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int, memory_types: Optional[list[str]] = None, query_embedding: Optional[np.ndarray] = None) -> list[dict]:
        """
        Query the vector store for similar items, with optional filtering by memory type.
        
        Args:
            query: Query text string
            top_k: Number of top results to return
            memory_types: Optional filter for memory types
            query_embedding: Optional pre-computed query embedding vector. If provided, will skip embedding computation.
        """
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """
        Insert or update vector data.
        'content' field is used for embedding. Key is used as the document ID.
        """
        raise NotImplementedError

    async def update_attributes(self, doc_id: str, updates: dict) -> bool:
        """
        Efficiently update the metadata attributes of a specific document without re-embedding.
        Returns True on success, False if the document ID is not found.
        """
        raise NotImplementedError

    async def increment_call_count(self, doc_id: str) -> bool:
        """
        Atomically increments the 'call_count' metadata field for a given document.
        Returns True on success, False if the document ID is not found.
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    """Abstract base class for a simple Key-Value store."""
    embedding_func = None # KV stores typically don't need embeddings.

    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """Given a list of keys, return the subset of keys that do not exist in the store."""
        raise NotImplementedError

    async def upsert(self, data: dict):
        """Insert new key-value pairs or update existing ones."""
        raise NotImplementedError

    async def drop(self):
        """Delete all data from the store."""
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    """Abstract base class for graph database storage, enhanced for dynamic memory."""
    embedding_func = None

    # --- Read Operations ---
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    # --- Write/Update Operations ---
    async def upsert_node(self, node_id: str, node_data: dict):
        """Insert a new node or update all attributes of an existing node."""
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict
    ):
        """Insert a new edge or update all attributes of an existing edge."""
        raise NotImplementedError

    async def update_node_attributes(self, node_id: str, updates: dict) -> bool:
        """Efficiently update specific attributes of a node."""
        raise NotImplementedError

    async def update_edge_attributes(self, source_id: str, target_id: str, updates: dict) -> bool:
        """Efficiently update specific attributes of an edge."""
        raise NotImplementedError
        
    async def increment_node_call_count(self, node_id: str) -> bool:
        """Atomically increments the 'call_count' attribute for a given node."""
        raise NotImplementedError

    async def increment_edge_call_count(self, source_id: str, target_id: str) -> bool:
        """Atomically increments the 'call_count' attribute for a given edge."""
        raise NotImplementedError

    async def delete_node(self, node_id: str):
        raise NotImplementedError