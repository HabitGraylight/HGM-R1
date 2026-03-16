# storage.py
import asyncio
import os
import logging
from dataclasses import dataclass, field
from typing import Any, Union, Optional, List, Tuple, Dict, Set
import networkx as nx
import numpy as np
import json
from datetime import datetime, timezone

from hypergraphmem.utils import (
    logger,
    load_json,
    write_json,
    compute_mdhash_id,
)

from hypergraphmem.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)

from nano_vectordb import NanoVectorDB

@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        if not self.global_config:
            raise ValueError(f"JsonKVStorage ({self.namespace}) requires global_config with 'working_dir'.")
            
        working_dir = self.global_config.get("working_dir")
        if not working_dir:
            raise ValueError(f"JsonKVStorage ({self.namespace}): 'working_dir' is missing in configuration.")
        
        os.makedirs(working_dir, exist_ok=True)
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        
        self._data = load_json(self._file_name) or {}
        logger.info(f"JsonKVStorage initialized for {self.namespace} at {self._file_name} (records={len(self._data)})")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        try:
            write_json(self._data, self._file_name)
        except Exception as e:
            logger.exception(f"JsonKVStorage failed to write file {self._file_name}: {e}")

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        self._data.update(data)
        return data

    async def drop(self):
        self._data = {}


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    """
    Persistent vector storage backed by nano-vectordb.
    Strict implementation based on official dbs.py:
    - Removes redundant caching (NanoVectorDB is already in-memory).
    - Handles read-modify-write for metadata updates.
    """
    embedding_func: Optional[Any] = None

    _vector_db: NanoVectorDB = field(init=False)
    _storage_file: str = field(init=False)

    def __post_init__(self):
        if not self.global_config:
            raise ValueError(f"NanoVectorDBStorage ({self.namespace}) requires global_config.")

        working_dir = self.global_config.get("working_dir")
        if not working_dir:
            raise ValueError(f"NanoVectorDBStorage ({self.namespace}): 'working_dir' missing in config.")

        nano_cfg = self.global_config.get("nano_vectordb", {})
        
        storage_dir = nano_cfg.get("storage_dir")
        if not storage_dir:
            storage_dir = os.path.join(working_dir, "nano_vector_store")
        
        os.makedirs(storage_dir, exist_ok=True)
        self._storage_file = os.path.join(storage_dir, f"{self.namespace}.json")

        # Determine embedding dimension
        embedding_dim = 2560 
        if self.embedding_func:
            embedding_dim = getattr(self.embedding_func, "embedding_dim", None)
        
        if not embedding_dim:
             embedding_cfg = self.global_config.get("embedding", {})
             val = (
                 nano_cfg.get("embedding_dim")
                 or self.global_config.get("embedding_dim")
                 or (embedding_cfg.get("embedding_dim") if isinstance(embedding_cfg, dict) else None)
             )
             if val:
                 embedding_dim = int(val)

        self.embedding_dim = embedding_dim
        
        # Initialize DB
        # NanoVectorDB automatically loads data from _storage_file into memory on init
        self._vector_db = NanoVectorDB(embedding_dim=self.embedding_dim, storage_file=self._storage_file)
        
        logger.info(
            f"NanoVectorDBStorage initialized for {self.namespace} at {self._storage_file} (dim={self.embedding_dim})"
        )

    async def upsert(self, data: dict[str, dict]):
        """
        Calculates embeddings for new content and inserts into DB.
        According to dbs.py, data MUST contain '__id__' and '__vector__'.
        """
        if not data: return []
        if self.embedding_func is None:
            return []

        valid_ids = []
        valid_contents = []
        
        # 1. Filter items that have content requiring embedding
        for _id, item in data.items():
            content = item.get("content", "")
            if content and isinstance(content, str) and content.strip():
                valid_ids.append(_id)
                valid_contents.append(content)

        if not valid_ids:
            return []
        
        try:
            # 2. Compute Embeddings
            emb_list = await self.embedding_func(valid_contents)
            emb_arr = np.asarray(emb_list, dtype="float32")
            if emb_arr.ndim == 1: emb_arr = emb_arr.reshape(1, -1)

            if emb_arr.shape[0] != len(valid_ids):
                logger.error(f"Embedding count mismatch. Expected {len(valid_ids)}, got {emb_arr.shape[0]}")
                return []

            # 3. Construct records for NanoVectorDB
            records = []
            for idx, _id in enumerate(valid_ids):
                vec = emb_arr[idx]
                if vec.ndim == 2: vec = vec.reshape(-1)
                record = data[_id].copy()
                record["__id__"] = _id
                record["__vector__"] = vec
                records.append(record)

            # 4. Call underlying upsert
            report = await asyncio.to_thread(self._vector_db.upsert, records)
            
            return (report.get("update") or []) + (report.get("insert") or [])
            
        except Exception as e:
            logger.error(f"Vector upsert failed: {e}", exc_info=True)
            return []

    async def update_attributes(self, doc_id: str, updates: dict) -> bool:
        """
        Updates metadata without re-embedding.
        Logic: Get -> Merge -> Upsert (with existing vector).
        """
        if not doc_id or not updates: return False

        try:
            # 1. Fetch existing record (contains __vector__)
            # dbs.py's get returns a list of dicts
            results = await asyncio.to_thread(self._vector_db.get, [doc_id])
            
            if not results or not isinstance(results, list) or len(results) == 0:
                logger.warning(f"[Nano] update_attributes failed: ID {doc_id} not found.")
                return False
            
            current_record = results[0].copy() # Copy to avoid mutating internal state implicitly
            
            # 2. Apply updates (exclude reserved fields)
            for k, v in updates.items():
                if k not in ["__id__", "__vector__", "__metrics__"]:
                    current_record[k] = v
            
            # 3. Write back
            # upsert expects a list of dicts
            await asyncio.to_thread(self._vector_db.upsert, [current_record])
            return True
            
        except Exception as e:
            logger.error(f"[Nano] update_attributes error for {doc_id}: {e}")
            return False

    async def increment_call_count(self, doc_id: str) -> bool:
        """
        Atomically increments 'call_count'.
        """
        try:
            # Reuse update_attributes logic to avoid code duplication
            # But we need to read the current value first
            results = await asyncio.to_thread(self._vector_db.get, [doc_id])
            if not results: return False
            
            current_count = results[0].get("call_count", 0)
            return await self.update_attributes(doc_id, {"call_count": current_count + 1})
        except Exception:
            return False

    async def query(self, query: str, top_k: int, memory_types: Optional[list[str]] = None, query_embedding: Optional[np.ndarray] = None) -> list[dict]:
        # 1. Prepare Query Vector
        q_vec = None
        if query_embedding is not None:
            q_vec = np.asarray(query_embedding, dtype="float32").reshape(-1)
        elif self.embedding_func is not None:
            try:
                if not query or not query.strip(): return []
                q_emb_list = await self.embedding_func([query])
                if not q_emb_list: return []
                q_vec = np.asarray(q_emb_list[0], dtype="float32")
            except Exception: pass
        
        if q_vec is None: return []

        # 2. Dimension Check
        if q_vec.shape[0] != self.embedding_dim:
             logger.warning(f"Query dim ({q_vec.shape[0]}) != DB dim ({self.embedding_dim})")
             return []

        # 3. Construct Filter
        _filter = None
        if memory_types:
            # NanoVectorDB accepts a lambda for filtering
            def _filter_lambda(data_dict):
                return data_dict.get("memory_type") in memory_types
            _filter = _filter_lambda

        # 4. Execute Query
        raw_results = await asyncio.to_thread(
            self._vector_db.query,
            query=q_vec,
            top_k=top_k,
            better_than_threshold=None,
            filter_lambda=_filter,
        )

        # 5. Format Results
        formatted = []
        for item in raw_results:
            doc_id = item.get("__id__")
            score = item.get("__metrics__", 0.0)
            
            # Convert cosine score to distance (1 - score)
            distance = 1.0 - float(score)
            
            # Clean reserved fields
            clean_item = {k: v for k, v in item.items() if k not in ["__id__", "__vector__", "__metrics__"]}
            
            formatted.append({"id": doc_id, "distance": distance, **clean_item})
            
        return formatted

    async def delete_entity(self, entity_name: str):
        """
        Delete an entity by ID.
        Note: BaseVectorStorage abstract class implies 'upsert' but strictly doesn't enforce 'delete',
        however, higher-level logic (manager) often requires it.
        """
        if not entity_name: return False
        try:
            await asyncio.to_thread(self._vector_db.delete, [entity_name])
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    async def index_done_callback(self):
        """Persist data to disk."""
        try:
            await asyncio.to_thread(self._vector_db.save)
        except Exception as e:
            logger.error(f"Save failed: {e}")


@dataclass
class NetworkXStorage(BaseGraphStorage):
    def __post_init__(self):
        if not self.global_config:
            raise ValueError(f"NetworkXStorage ({self.namespace}) requires global_config.")

        working_dir = self.global_config.get("working_dir")
        if not working_dir:
            raise ValueError(f"NetworkXStorage ({self.namespace}): 'working_dir' missing in config.")
            
        os.makedirs(working_dir, exist_ok=True)
        self._graphml_xml_file = os.path.join(
            working_dir, f"graph_{self.namespace}.graphml"
        )
        
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            try:
                if isinstance(preloaded_graph, nx.DiGraph):
                    self._graph = preloaded_graph.to_undirected()
                else:
                    self._graph = preloaded_graph.copy()
            except Exception:
                self._graph = nx.Graph()
        else:
            self._graph = nx.Graph()

        logger.info(f"NetworkXStorage initialized at {self._graphml_xml_file}")

    @staticmethod
    def load_nx_graph(file_name) -> Optional[nx.Graph]:
        if os.path.exists(file_name):
            try:
                g = nx.read_graphml(file_name)
                if isinstance(g, nx.DiGraph): g = g.to_undirected()
                return g
            except Exception: return None
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        try:
            g_copy = graph.copy()
            for nid in list(g_copy.nodes()):
                attrs = g_copy.nodes[nid]
                for k, v in list(attrs.items()):
                    if isinstance(v, (list, dict)):
                         try: g_copy.nodes[nid][k] = json.dumps(v)
                         except: g_copy.nodes[nid].pop(k, None)
            nx.write_graphml(g_copy, file_name)
        except Exception as e:
            logger.error(f"Graph save failed: {e}")

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)
    
    async def has_node(self, node_id: str) -> bool: return self._graph.has_node(node_id)
    async def get_node(self, node_id: str) -> Union[dict, None]: return self._graph.nodes.get(node_id)
    
    async def upsert_node(self, node_id: str, node_data: dict): 
        if not self._graph.has_node(node_id): 
            self._graph.add_node(node_id, **node_data)
        else: 
            self._graph.nodes[node_id].update(node_data)
        self._graph.nodes[node_id]['last_accessed'] = datetime.now(timezone.utc).isoformat()

    async def upsert_edge(self, s, t, d): 
        self._graph.add_edge(s, t, **d)

    async def update_node_attributes(self, n, u): 
        if self._graph.has_node(n): 
            self._graph.nodes[n].update(u)
            self._graph.nodes[n]['last_accessed'] = datetime.now(timezone.utc).isoformat()
            return True
        return False

    async def increment_node_call_count(self, n):
        if self._graph.has_node(n): 
            self._graph.nodes[n]['call_count'] = self._graph.nodes[n].get('call_count', 0) + 1
            self._graph.nodes[n]['last_accessed'] = datetime.now(timezone.utc).isoformat()

    async def delete_node(self, n): 
        if self._graph.has_node(n): self._graph.remove_node(n)

    async def get_node_edges(self, n): 
        return list(self._graph.edges(n, data=True)) if self._graph.has_node(n) else []

    async def get_adjacent_nodes(self, n, target_prefix):
        if not self._graph.has_node(n): return set()
        return {nbr for nbr in self._graph.neighbors(n) if nbr.startswith(target_prefix)}

VECTOR_BACKENDS = {
    "nano": NanoVectorDBStorage,
}
DEFAULT_VECTOR_BACKEND = "nano"