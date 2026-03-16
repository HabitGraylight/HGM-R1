# hypergraphmem.py
import asyncio
import os
import shutil
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import networkx as nx

from hypergraphmem.base import BaseVectorStorage
from hypergraphmem.utils import compute_mdhash_id
from hypergraphmem.operation import (
    add_memory_unit,
    extract_memories_from_text,
    retrieve_hyperedge,
    retrieve_chunk,
    ner_from_text,
    delete_memory_node,
    update_memory_node
)
from hypergraphmem.storage import (
    JsonKVStorage,
    NetworkXStorage,
    VECTOR_BACKENDS,
    DEFAULT_VECTOR_BACKEND,
)

# [CRITICAL]: 导入无状态 embedding 函数
from hypergraphmem.embedding import get_embeddings

logger = logging.getLogger(__name__)

@dataclass
class LayeredMemoryManager:
    working_dir: str = None 
    config: Dict[str, Any] = field(default_factory=dict)
    
    kv_cls = JsonKVStorage
    vector_cls = None
    graph_cls = NetworkXStorage

    initialized: bool = False
    entity_vdb: Optional[BaseVectorStorage] = None
    hyperedge_vdb: Optional[BaseVectorStorage] = None
    text_chunks_kv: Optional[JsonKVStorage] = None
    kg: Optional[NetworkXStorage] = None
    chunk_vdb: Optional[BaseVectorStorage] = None
    
    def __post_init__(self):
        # 1. 配置防御性拷贝
        if not self.config:
            self.config = {}
        self.config = self.config.copy()

        # 2. 确定 Working Directory (单一事实来源)
        if self.working_dir:
            self.config["working_dir"] = self.working_dir
        else:
            self.working_dir = self.config.get("working_dir", "./default_cache")

        # 确保目录存在
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir, exist_ok=True)
        
        abs_work_dir = os.path.abspath(self.working_dir)
        self.config["working_dir"] = abs_work_dir

        # [LOGGING] 初始化路径确认
        logger.info(f"========== [MemoryManager Init] ==========")
        logger.info(f"📂 Root Working Dir : {abs_work_dir}")

        # 3. 准备传递给 Storage 层的配置
        storage_section = self.config.get("storage", {}).copy()
        operation_section = self.config.get("operation", {}).copy()
        final_storage_config = {**storage_section, **operation_section}
        final_storage_config["working_dir"] = abs_work_dir
        
        if "embedding" in self.config:
            final_storage_config["embedding"] = self.config["embedding"]

        # 4. 选择向量后端
        _pref = (final_storage_config.get("vector_backend_preference") or DEFAULT_VECTOR_BACKEND).lower()
        if _pref not in VECTOR_BACKENDS:
            logger.warning(f"⚠️ Unsupported backend {_pref}, using default")
            self.vector_cls = VECTOR_BACKENDS[DEFAULT_VECTOR_BACKEND]
        else:
            self.vector_cls = VECTOR_BACKENDS[_pref]
        
        # 5. 创建 Embedding Wrapper
        embedding_wrapper = lambda texts: get_embeddings(texts, self.config)

        # 6. 初始化存储层
        logger.info(f"🛠️  Initializing Storage Components...")

        # KV Storage
        self.text_chunks_kv = self.kv_cls(namespace="chunks", global_config=final_storage_config)
        logger.info(f"   - KV Storage (Chunks)  : Ready")
        
        # Vector Storage
        self.entity_vdb = self.vector_cls(namespace="entity", global_config=final_storage_config, embedding_func=embedding_wrapper)
        self.hyperedge_vdb = self.vector_cls(namespace="hyperedge", global_config=final_storage_config, embedding_func=embedding_wrapper)
        self.chunk_vdb = self.vector_cls(namespace="chunks", global_config=final_storage_config, embedding_func=embedding_wrapper)
        logger.info(f"   - Vector DB (Nano)     : Entity / Hyperedge / Chunks Ready")
        
        # Graph Storage
        self.kg = self.graph_cls(namespace="chunk_entity_relation", global_config=final_storage_config)
        logger.info(f"   - Knowledge Graph      : Ready (NetworkX)")

        self.initialized = True
        logger.info(f"✅ MemoryManager Initialization Complete.")
        logger.info(f"==========================================")

    # ----------------- Monitoring Helper (Enhanced) -----------------
    def _try_log_graph_stats(self, operation_tag: str):
        """
        尝试读取并打印图谱的统计信息。
        增加了对 NetworkXStorage 内部 _graph 属性的直接访问尝试，以确保能读到数据。
        """
        try:
            num_nodes = -1
            num_edges = -1
            
            # 尝试 1: 直接从 NetworkXStorage 对象获取（如果它暴露了方法）
            if hasattr(self.kg, "number_of_nodes"):
                 num_nodes = self.kg.number_of_nodes()
            
            # 尝试 2: 访问受保护的 _graph 属性 (Pythonic way for white-box monitoring)
            # storage.py 中 NetworkXStorage 把图存在 self._graph
            if num_nodes == -1 and hasattr(self.kg, "_graph"):
                g = self.kg._graph
                if isinstance(g, nx.Graph):
                    num_nodes = g.number_of_nodes()
                    num_edges = g.number_of_edges()

            if num_nodes >= 0:
                logger.info(f"[📊 Graph Stats] After '{operation_tag}': Nodes={num_nodes} | Edges={num_edges}")
            else:
                logger.debug(f"[Graph Stats] Could not retrieve stats for {operation_tag}")

        except Exception as e:
            logger.warning(f"[Graph Stats] Error logging stats: {e}")

    # ----------------- Write Operations -----------------

    async def ingest_source_chunk(self, text: str, timestamp: str = None) -> str:
        """Store raw chunk."""
        if not text or not text.strip(): return None

        content = text.strip()
        chunk_id = compute_mdhash_id(content, prefix="chunk-")
        
        existing_list = await self.text_chunks_kv.get_by_ids([chunk_id])
        
        if not existing_list or existing_list[0] is None:
            chunk_data = {
                "content": content,
                "chunk_id": chunk_id,
                "created_at": datetime.utcnow().isoformat(),
                "happened_at": timestamp,
                "metadata": {"source": "agent_rollout"}
            }
            # [LOGGING] 提升为 INFO
            logger.info(f"[💾 Ingest] New Source Chunk detected. ID: {chunk_id[:8]}... | Len: {len(content)}")
            
            await self.text_chunks_kv.upsert({chunk_id: chunk_data})
            if self.chunk_vdb:
                await self.chunk_vdb.upsert({chunk_id: chunk_data})
        else:
            # [LOGGING] Debug 级别即可，但也打印一下确认存在
            logger.info(f"[⏭️ Ingest] Chunk already exists. ID: {chunk_id[:8]}...")
            
        return chunk_id

    async def add_memory_fact(self, content: str, memory_type: str, timestamp: str, source_chunk_id: str):
        if not content: return
        
        # [LOGGING] 开始提取
        logger.info(f"[🧠 Add Fact] Processing: '{content[:40]}...' (Type: {memory_type})")
        
        try:
            entities = await ner_from_text(content, self.config)
            # [LOGGING] 实体提取结果
            if entities:
                logger.info(f"   -> Found Entities: {entities}")
            else:
                logger.info(f"   -> No specific entities found.")
        except Exception as e:
            logger.warning(f"   -> NER failed: {e}")
            entities = []

        mu_dict = {
            "hyperedge_content": content,
            "memory_type": memory_type,
            "source_chunk_id": source_chunk_id,
            "entities": entities,
            "created_at": datetime.utcnow().isoformat(),
            "happened_at": timestamp,
        }

        await add_memory_unit(
            mu_dict, 
            self.kg, 
            self.entity_vdb, 
            self.hyperedge_vdb, 
            self.text_chunks_kv
        )
        # [LOGGING] 关键：操作后的图统计
        self._try_log_graph_stats("Add Fact")

    # ----------------- Update / Delete Operations -----------------

    async def delete_memory_by_id(self, memory_id: str) -> bool:
        if not memory_id: return False
        logger.info(f"[🗑️ Delete] Request to remove node: {memory_id}")
        
        res = await delete_memory_node(
            memory_id, self.kg, self.entity_vdb, self.hyperedge_vdb
        )
        
        if res: 
            logger.info(f"   -> Delete Successful.")
            self._try_log_graph_stats("Delete Memory")
        else:
            logger.warning(f"   -> Delete Failed (Node not found or error).")
        return res

    async def update_memory_content(self, memory_id: str, new_content: str = None, new_type: str = None, new_happened_at: str = None) -> bool:
        if not memory_id: return False
        
        updates_log = []
        if new_content: updates_log.append("Content")
        if new_type: updates_log.append(f"Type->{new_type}")
        if new_happened_at: updates_log.append(f"Time->{new_happened_at}")
        
        logger.info(f"[📝 Update] Updating {memory_id}. Fields: {', '.join(updates_log)}")

        res = await update_memory_node(
            memory_id, 
            self.kg, 
            self.hyperedge_vdb,
            new_content=new_content,
            new_type=new_type,
            new_happened_at=new_happened_at
        )
        if res: 
            logger.info(f"   -> Update Successful.")
            self._try_log_graph_stats("Update Memory")
        else:
            logger.warning(f"   -> Update Failed.")
        return res

    # ----------------- Read / Retrieval Operations -----------------

    async def retrieve_context_dict(self, query: str) -> Dict[str, Any]:
        logger.info(f"[🔍 Retrieve] Query: '{query[:50]}...'")
        
        hyperedges = await retrieve_hyperedge(
            query, self.kg, self.entity_vdb, self.hyperedge_vdb, None, self.config
        )
        hyperedge_ids = {h["id"] for h in hyperedges}
        
        chunks = await retrieve_chunk(
            query, hyperedge_ids, self.config, self.text_chunks_kv, None
        )
        
        logger.info(f"   -> Result: {len(hyperedges)} Hyperedges, {len(chunks)} Source Chunks.")
        return {"hyperedges": hyperedges, "sources": chunks}

    async def retrieve_hyperedges_list(self, query: str) -> List[Dict]:
        res = await retrieve_hyperedge(
            query, self.kg, self.entity_vdb, self.hyperedge_vdb, None, self.config
        )
        # 增加 Debug 级别的详细日志
        logger.debug(f"[Retrieval List] Query='{query}' -> Found {len(res)} items")
        return res

    async def aextract_memories_from_text(self, text: str, chunk_key: str, timestamp: str = None) -> list[dict]:
        # Log 在 executor 已经有了，但这里加一层保险
        res = await extract_memories_from_text(text, chunk_key, self.config, timestamp=timestamp)
        logger.info(f"[⛏️ Extract] Chunk {chunk_key[:8]}... -> Extracted {len(res)} Raw Facts.")
        return res

    # ----------------- Lifecycle -----------------

    async def close(self):
        logger.info(f"[🛑 Closing] Saving indexes to disk...")
        tasks = []
        if self.text_chunks_kv: tasks.append(self.text_chunks_kv.index_done_callback())
        if self.entity_vdb: tasks.append(self.entity_vdb.index_done_callback())
        if self.hyperedge_vdb: tasks.append(self.hyperedge_vdb.index_done_callback())
        if self.chunk_vdb: tasks.append(self.chunk_vdb.index_done_callback())
        if self.kg: tasks.append(self.kg.index_done_callback())
        
        if tasks:
            await asyncio.gather(*tasks)
        logger.info(f"[🛑 Closing] All indexes saved.")

    async def destroy(self):
        logger.warning(f"[💥 Destroy] Clearing working directory: {self.working_dir}")
        try:
            await self.close()
        except Exception as e:
            logger.warning(f"Error closing before destroy: {e}")
        
        # if os.path.exists(self.working_dir):
        #     try:
        #         shutil.rmtree(self.working_dir)
        #         logger.info(f"[💥 Destroy] Directory removed.")
        #     except Exception as e:
        #         logger.error(f"[💥 Destroy] Failed to remove dir: {e}")