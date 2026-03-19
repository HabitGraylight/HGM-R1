# hypergraphmem.py
import asyncio
import os
import shutil
import logging
import re
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
from hypergraphmem.llm import close_global_session

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
    _last_memory_id: Optional[str] = None
    _last_memory_session_id: Optional[str] = None
    
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

    @staticmethod
    def _extract_dia_session_id(content: str) -> Optional[int]:
        m = re.search(r"\[(?:D(\d+):\d+)\]", content or "")
        if not m:
            return None
        try:
            d_main = int(m.group(1))
        except Exception:
            return None
        return ((d_main - 1) // 3) + 1

    @staticmethod
    def _extract_time_bucket(happened_at: Optional[str]) -> Optional[str]:
        ts = (happened_at or "").strip()
        if not ts:
            return None

        # Fast path for common LoCoMo/HGM timestamps: YYYY-MM-DD or YYYY-MM-DD HH:MM
        m = re.search(r"(\d{4}-\d{2}-\d{2})", ts)
        if m:
            return m.group(1)

        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.date().isoformat()
        except Exception:
            return None

    def _resolve_temporal_bucket(self, content: str, happened_at: Optional[str]) -> Optional[str]:
        dia_session = self._extract_dia_session_id(content)
        if dia_session is not None:
            return f"S{dia_session}"

        day_bucket = self._extract_time_bucket(happened_at)
        if day_bucket:
            return f"D{day_bucket}"
        return None

    def _graph_cfg(self) -> Dict[str, Any]:
        g = self.config.get("graph", {})
        return g if isinstance(g, dict) else {}

    async def _maybe_link_temporal_edge(
        self,
        current_mem_id: Optional[str],
        content: str,
        happened_at: Optional[str] = None,
    ) -> None:
        if not current_mem_id:
            return

        gcfg = self._graph_cfg()
        tcfg = gcfg.get("temporal_link", {}) if isinstance(gcfg.get("temporal_link", {}), dict) else {}
        enabled = bool(tcfg.get("enabled", True))

        curr_bucket = self._resolve_temporal_bucket(content, happened_at)

        if not enabled:
            self._last_memory_id = current_mem_id
            self._last_memory_session_id = curr_bucket
            return

        link_across_sessions = bool(tcfg.get("link_across_sessions", False))
        should_link = self._last_memory_id is not None

        if should_link and not link_across_sessions:
            # Strict temporal guard: if bucket is unknown on either side, do not force-link.
            if self._last_memory_session_id is None or curr_bucket is None:
                should_link = False
            else:
                should_link = str(self._last_memory_session_id) == str(curr_bucket)

        if should_link:
            try:
                await self.kg.upsert_edge(
                    self._last_memory_id,
                    current_mem_id,
                    {
                        "relation": "temporal_next",
                        "created_at": datetime.utcnow().isoformat(),
                    },
                )
            except Exception as e:
                logger.warning(f"[Graph] Failed temporal link {self._last_memory_id}->{current_mem_id}: {e}")

        self._last_memory_id = current_mem_id
        self._last_memory_session_id = curr_bucket

    async def _maybe_link_memory_chunk(self, current_mem_id: Optional[str], source_chunk_id: Optional[str]) -> None:
        if not current_mem_id or not source_chunk_id:
            return

        gcfg = self._graph_cfg()
        enabled = bool(gcfg.get("link_memory_to_chunk", True))
        if not enabled:
            return

        try:
            chunk = None
            chunks = await self.text_chunks_kv.get_by_ids([source_chunk_id])
            if chunks and chunks[0]:
                chunk = chunks[0]
            chunk_content = chunk.get("content", "") if chunk else ""
            await self.kg.upsert_node(
                source_chunk_id,
                {
                    "content": chunk_content,
                    "node_type": "chunk",
                    "source_id": source_chunk_id,
                },
            )
            await self.kg.upsert_edge(
                current_mem_id,
                source_chunk_id,
                {
                    "relation": "source_chunk",
                    "created_at": datetime.utcnow().isoformat(),
                },
            )
        except Exception as e:
            logger.warning(f"[Graph] Failed memory-chunk link for {current_mem_id}: {e}")

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
        
        gcfg = self._graph_cfg()
        use_entity_links = bool(gcfg.get("use_entity_links", True))

        if use_entity_links:
            try:
                entities = await ner_from_text(content, self.config)
                if entities:
                    logger.info(f"   -> Found Entities: {entities}")
                else:
                    logger.info("   -> No specific entities found.")
            except Exception as e:
                logger.warning(f"   -> NER failed: {e}")
                entities = []
        else:
            entities = []
            logger.info("   -> Entity linking disabled by graph.use_entity_links=false")

        mu_dict = {
            "hyperedge_content": content,
            "memory_type": memory_type,
            "source_chunk_id": source_chunk_id,
            "entities": entities,
            "created_at": datetime.utcnow().isoformat(),
            "happened_at": timestamp,
        }

        mem_id = await add_memory_unit(
            mu_dict,
            self.kg,
            self.entity_vdb,
            self.hyperedge_vdb,
            self.text_chunks_kv,
        )

        await self._maybe_link_temporal_edge(mem_id, content, timestamp)
        await self._maybe_link_memory_chunk(mem_id, source_chunk_id)

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

        qcfg = self.config.get("query", {}) if isinstance(self.config.get("query", {}), dict) else {}
        max_chunk_candidates = int(qcfg.get("max_chunk_candidates", 64))
        source_chunk_neighbors_per_hyperedge = int(qcfg.get("source_chunk_neighbors_per_hyperedge", 3))
        chunk_vdb_top_k = int(qcfg.get("chunk_vdb_top_k", 0))

        chunk_scores: Dict[str, float] = {}

        def _add_chunk_candidate(chunk_id: str, score: float) -> None:
            if not isinstance(chunk_id, str) or not chunk_id.startswith("chunk-"):
                return
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + float(score)

        # 1) Direct source chunk ids on retrieved memories.
        for h in hyperedges:
            _add_chunk_candidate(h.get("source_id"), 3.0)

        # 2) Expand source_chunk neighbors from retrieved memory nodes.
        if source_chunk_neighbors_per_hyperedge > 0:
            mem_ids = [
                h.get("id")
                for h in hyperedges
                if isinstance(h.get("id"), str) and h.get("id", "").startswith("mem-")
            ]
            if mem_ids:
                edge_lists = await asyncio.gather(*[self.kg.get_node_edges(mid) for mid in mem_ids], return_exceptions=True)
                for edges in edge_lists:
                    if isinstance(edges, Exception) or not isinstance(edges, list):
                        continue
                    local: List[tuple[str, float]] = []
                    for s, t, d in edges:
                        s_id = s if isinstance(s, str) else ""
                        t_id = t if isinstance(t, str) else ""
                        if s_id.startswith("chunk-"):
                            other = s_id
                        elif t_id.startswith("chunk-"):
                            other = t_id
                        else:
                            continue
                        rel = d.get("relation", "") if isinstance(d, dict) else ""
                        w = 2.0 if rel == "source_chunk" else 1.0
                        local.append((other, w))

                    local.sort(key=lambda x: (-x[1], x[0]))
                    if source_chunk_neighbors_per_hyperedge > 0:
                        local = local[:source_chunk_neighbors_per_hyperedge]
                    for cid, w in local:
                        _add_chunk_candidate(cid, w)

        # 3) Optional semantic chunk expansion from chunk vector DB.
        if chunk_vdb_top_k > 0 and self.chunk_vdb is not None:
            try:
                extra_chunks = await self.chunk_vdb.query(query, top_k=chunk_vdb_top_k)
                for item in extra_chunks or []:
                    cid = item.get("id") if isinstance(item, dict) else None
                    dist = item.get("distance", 1.0) if isinstance(item, dict) else 1.0
                    try:
                        sim = 1.0 - float(dist)
                    except Exception:
                        sim = 0.0
                    _add_chunk_candidate(cid, max(0.25, sim))
            except Exception as e:
                logger.warning(f"[Retrieve] chunk_vdb expansion failed: {e}")

        ranked_chunk_ids = sorted(chunk_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if max_chunk_candidates > 0:
            ranked_chunk_ids = ranked_chunk_ids[:max_chunk_candidates]
        chunk_ids = {cid for cid, _ in ranked_chunk_ids}

        chunks = await retrieve_chunk(
            query, chunk_ids, self.config, self.text_chunks_kv, None
        )

        logger.info(
            f"   -> Result: {len(hyperedges)} Hyperedges, {len(chunks)} Source Chunks "
            f"(chunk_candidates={len(chunk_ids)})."
        )
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
        try:
            await close_global_session()
        except Exception as e:
            logger.debug(f"[Close] ignore llm session close error: {e}")
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