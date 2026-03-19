# operation.py
import asyncio
import re
import json
import logging
from typing import Optional, List, Dict, Any, Set
from datetime import datetime, timezone
import numpy as _np

from hypergraphmem.utils import (
    normalize_entity,
    compute_mdhash_id,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
)
from hypergraphmem.prompt import PROMPTS, JSON_SCHEMAS
from hypergraphmem.llm import llm_chat_request
from hypergraphmem.reranker import rerank_texts

logger = logging.getLogger(__name__)


def _get_query_setting(global_config: dict, key: str, default: Any) -> Any:
    if key in global_config:
        return global_config.get(key, default)

    query_cfg = global_config.get("query", {})
    if isinstance(query_cfg, dict):
        return query_cfg.get(key, default)

    return default


def _month_to_num(name: str) -> Optional[int]:
    token = (name or "").strip().lower()
    months = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    return months.get(token)


def _build_datetime(y: int, mo: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0) -> Optional[datetime]:
    try:
        return datetime(y, mo, d, hh, mm, ss)
    except Exception:
        return None


def _extract_date_from_text(text: str) -> Optional[datetime]:
    s = (text or "").strip()
    if not s:
        return None

    # YYYY-MM-DD / YYYY-MM-DD HH:MM / YYYY-MM-DDTHH:MM:SS
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hh = int(m.group(4) or 0)
        mm = int(m.group(5) or 0)
        ss = int(m.group(6) or 0)
        return _build_datetime(y, mo, d, hh, mm, ss)

    # Month DD, YYYY (e.g., March 26, 2023)
    m = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b", s)
    if m:
        mo = _month_to_num(m.group(1))
        d = int(m.group(2))
        y = int(m.group(3))
        if mo is not None:
            return _build_datetime(y, mo, d)

    # DD Month YYYY (e.g., 26 March 2023)
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s*,?\s*(\d{4})\b", s)
    if m:
        d = int(m.group(1))
        mo = _month_to_num(m.group(2))
        y = int(m.group(3))
        if mo is not None:
            return _build_datetime(y, mo, d)

    # MM/DD/YYYY
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", s)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return _build_datetime(y, mo, d)

    return None


def _parse_happened_at(ts: Any) -> Optional[datetime]:
    s = ("" if ts is None else str(ts)).strip()
    if not s:
        return None

    dt = _extract_date_from_text(s)
    if dt is not None:
        return dt

    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _memory_event_time(mem: dict) -> Optional[datetime]:
    dt = _parse_happened_at(mem.get("happened_at"))
    if dt is not None:
        return dt
    return _extract_date_from_text(str(mem.get("content", "")))


def _temporal_query_mode(query: str) -> Optional[str]:
    q = (query or "").lower()

    if any(k in q for k in ["first", "earliest", "start", "started", "begin", "initial"]):
        return "earliest"
    if any(k in q for k in ["latest", "last", "most recent", "recently", "recent"]):
        return "latest"
    if re.search(r"\b(when|date|day|month|year|time)\b", q):
        return "time"

    return None


# =============================================================================
#  Formatter & Helper Functions
# =============================================================================

def simplify_time(ts):
    if ts and isinstance(ts, str) and "T" in ts:
        return ts.split("T")[0]
    return ts

def _format_single_hyperedge(h: dict) -> str | None:
    content = (h.get("content") or "").strip()
    if not content:
        return None

    memory_type = h.get("memory_type", "short_term")
    call_count = h.get("call_count", 0)
    mem_id_short = h.get("id", "")

    happened_at = h.get("happened_at")
    last_called = h.get("last_called") 

    happened_str = simplify_time(happened_at)
    last_accessed_str = simplify_time(last_called)

    metadata = f"type={memory_type} | id={mem_id_short} | calls={call_count} | happened={happened_str} | last_accessed={last_accessed_str}"
    return f"- CONTENT: {content}\n  METADATA: {metadata}"


def format_memory_context(context: dict, include_sources: bool = True) -> str:
    if not context:
        return "(No memory context available)"

    hyperedges = context.get("hyperedges", []) or []
    sources = context.get("sources", []) or []

    long_term_memories = []
    short_term_memories = []

    for h in hyperedges:
        entry = _format_single_hyperedge(h)
        if entry:
            if h.get("memory_type") == "long_term":
                long_term_memories.append(entry)
            else:
                short_term_memories.append(entry)

    def build_section(tag: str, items: list[str]) -> str:
        lines = [f"<{tag} count={len(items)} >"]
        if items:
            lines.extend(items)
        else:
            lines.append("(empty)")
        lines.append(f"</{tag}>")
        return "\n".join(lines)

    parts = [
        "",
        build_section("long_term_memory", long_term_memories),
        "",
        build_section("short_term_memory", short_term_memories)
    ]

    if include_sources and sources:
        source_entries = [f"[Source {i+1}]: {s.get('content','').strip()}" for i, s in enumerate(sources)]
        parts.append("")
        parts.append(build_section("source_documents", source_entries))

    return "\n".join(parts)


def chunking_by_token_size(content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"):
    try:
        tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    except Exception:
        # Fallback if encoding fails
        parts = [p.strip() for p in content.split('.') if p.strip()]
        return [{"tokens": len(p.split()), "content": p, "chunk_order_index": i} for i, p in enumerate(parts)]

    results = []
    step = max_token_size - overlap_token_size
    if step <= 0: step = max_token_size
    
    for index, start in enumerate(range(0, len(tokens), step)):
        chunk_tokens = tokens[start: start + max_token_size]
        try:
            chunk_content = decode_tokens_by_tiktoken(chunk_tokens, model_name=tiktoken_model)
        except Exception:
            chunk_content = ""
        results.append({
            "tokens": min(max_token_size, len(tokens) - start),
            "content": chunk_content.strip(),
            "chunk_order_index": index,
        })
    return results

# =============================================================================
#  LLM Extraction Functions (Stateless)
# =============================================================================

async def extract_memories_from_text(
    text: str, 
    chunk_key: str, 
    global_config: dict, 
    timestamp: str = None
) -> List[Dict]:
    """
    Uses stateless llm_chat_request to extract knowledge units.
    """
    # [LOG]
    logger.info(f"[Extract] Processing chunk {chunk_key} (len={len(text)})")
    
    extraction_template = PROMPTS.get("extraction", "")
    extract_prompt = extraction_template.format(passage=text, reference_time=timestamp or "Unknown")

    try:
        # [FIX]: 直接调用无状态函数，传递 config
        raw = await llm_chat_request(
            prompt=extract_prompt,
            config=global_config,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": JSON_SCHEMAS.get("extraction"),
            },
        )
    except Exception as e:
        logger.warning(f"[Extract] Failed for {chunk_key}: {e}")
        return []

    knowledge_units = []
    cleaned_raw = raw.strip()
    if cleaned_raw.startswith("```json"): cleaned_raw = cleaned_raw[7:]
    if cleaned_raw.endswith("```"): cleaned_raw = cleaned_raw[:-3]
    
    try:
        data = json.loads(cleaned_raw)
        if isinstance(data, dict) and "knowledge_units" in data:
            for u in data["knowledge_units"]:
                if isinstance(u, dict):
                    content = str(u.get("content", "")).strip()
                    ts = str(u.get("timestamp", timestamp)).strip()
                    if content:
                        knowledge_units.append({"content": content, "timestamp": ts})
    except json.JSONDecodeError:
        pass

    # [LOG]
    logger.info(f"[Extract] Done chunk {chunk_key}. Found {len(knowledge_units)} facts.")
    return knowledge_units


def _looks_like_time_entity(entity: str) -> bool:
    e = (entity or "").lower().strip()
    return bool(
        re.search(r"\b(am|pm)\b", e)
        or re.search(r"\b\d{1,2}:\d{2}\b", e)
        or re.search(r"\b\d{1,2}\s+[a-z]+,\s*\d{4}\b", e)
    )


def _extract_speaker_name(text: str) -> str:
    m = re.search(r"\]\s*([^:\n]{1,40}):", text or "")
    return (m.group(1).strip().lower() if m else "")


def _filter_entities(entities: list[str], text: str, global_config: dict) -> list[str]:
    cfg = (global_config or {}).get("entity_filter", {})
    if not isinstance(cfg, dict):
        cfg = {}

    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        # keep stable behavior by default
        out = []
        seen = set()
        for e in entities:
            if not e:
                continue
            n = e.strip().lower()
            if n and n not in seen:
                seen.add(n)
                out.append(n)
        return out

    drop_time_entities = bool(cfg.get("drop_time_entities", True))
    drop_speaker_name = bool(cfg.get("drop_speaker_name", True))
    min_entity_len = int(cfg.get("min_entity_len", 2))

    drop_stop_entities = cfg.get("drop_stop_entities", ["calvin", "dave"])
    if not isinstance(drop_stop_entities, list):
        drop_stop_entities = ["calvin", "dave"]
    stop_set = {str(x).strip().lower() for x in drop_stop_entities if str(x).strip()}

    speaker_name = _extract_speaker_name(text) if drop_speaker_name else ""

    out = []
    seen = set()
    for ent in entities:
        n = (ent or "").strip().lower()
        if not n:
            continue
        if len(n) < min_entity_len:
            continue
        if n in stop_set:
            continue
        if speaker_name and n == speaker_name:
            continue
        if drop_time_entities and _looks_like_time_entity(n):
            continue
        if n in seen:
            continue
        seen.add(n)
        out.append(n)

    return out


async def ner_from_text(text: str, global_config: dict) -> list[str]:
    """
    Uses stateless llm_chat_request to extract entities.
    """
    ner_template = PROMPTS.get("ner", "")
    try:
        # [FIX]: 直接调用无状态函数
        ner_raw = await llm_chat_request(
            prompt=ner_template.format(passage=text),
            config=global_config,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": JSON_SCHEMAS.get("ner"),
            },
        )
        data = json.loads(ner_raw)
        entities = []
        if isinstance(data, dict) and "named_entities" in data:
            for ent in data["named_entities"]:
                norm = normalize_entity(ent)
                if norm:
                    entities.append(norm)

        filtered = _filter_entities(entities, text, global_config)

        # [LOG]
        if len(filtered) > 0:
            logger.info(f"[NER] Extracted {len(filtered)} entities from text (raw={len(entities)}, len={len(text)}).")

        return filtered
    except Exception as e:
        logger.warning(f"[NER] Failed: {e}")
        return []

# =============================================================================
#  Storage Atomic Operations
# =============================================================================

def _extract_dia_meta_from_content(content: str) -> tuple[Optional[str], Optional[int], Optional[int]]:
    m = re.search(r"\[(D(\d+):(\d+))\]", content or "")
    if not m:
        return None, None, None
    dia_id = m.group(1)
    d_main = int(m.group(2))
    d_turn = int(m.group(3))
    session_id = ((d_main - 1) // 3) + 1
    return dia_id, session_id, d_turn


async def add_memory_unit(
    memory_unit: dict,
    knowledge_graph: Any, 
    entity_vdb: Any,     
    hyperedge_vdb: Any,   
    text_chunks_db: Optional[Any] = None,
):  
    hyperedge_name = memory_unit["hyperedge_content"]
    source_id = memory_unit.get("source_chunk_id")
    memory_type = memory_unit.get("memory_type")
    created_at = memory_unit.get("created_at")   
    happened_at = memory_unit.get("happened_at")
    now_ts = datetime.now(timezone.utc).isoformat()
    hyperedge_id = compute_mdhash_id(hyperedge_name, prefix="mem-")
    dia_id, dia_session_id, dia_turn_id = _extract_dia_meta_from_content(hyperedge_name)

    hyperedge_attrs = {
        "content": hyperedge_name,
        "source_id": source_id,
        "memory_type": memory_type,
        "participants": [],
        "call_count": 0,
        "created_at": created_at,
        "last_called": now_ts,
        "happened_at": happened_at,
        "dia_id": dia_id,
        "dia_session_id": dia_session_id,
        "dia_turn_id": dia_turn_id,
    }

    # [1] 顺序执行：先写图
    await knowledge_graph.upsert_node(hyperedge_id, hyperedge_attrs)
    
    # [2] 顺序执行：再写向量库
    # 传递完整属性以保证元数据一致性
    await hyperedge_vdb.upsert({hyperedge_id: hyperedge_attrs})
    
    participant_ids = []
    entities = memory_unit.get("entities", [])
    
    # [3] 顺序执行：处理实体
    # 以前这里是收集 tasks 然后 gather，现在改成循环内直接 await
    for entity_name in entities:
        if not entity_name: continue
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        participant_ids.append(entity_id)

        entity_attrs = {"content": entity_name, "source_id": source_id, "call_count": 0, "created_at": created_at}
        
        # 串行写入：绝对安全
        await knowledge_graph.upsert_node(entity_id, entity_attrs)
        await entity_vdb.upsert({entity_id: entity_attrs})
        await knowledge_graph.upsert_edge(hyperedge_id, entity_id, {"created_at": created_at})

    # [4] 最后更新参与者关系
    if participant_ids:
        await knowledge_graph.update_node_attributes(hyperedge_id, {"participants": participant_ids})

    # [LOG]
    try:
        if hasattr(knowledge_graph, "_graph"):
            num_nodes = knowledge_graph._graph.number_of_nodes()
            logger.info(f"[GraphStats] Nodes: {num_nodes}. Added Mem: {hyperedge_name[:30]}...")
    except Exception:
        pass

    return hyperedge_id

# --- Delete / Update Operations ---

async def delete_memory_node(
    node_id: str,
    knowledge_graph: Any,
    entity_vdb: Any,
    hyperedge_vdb: Any
) -> bool:
    try:
        if not await knowledge_graph.get_node(node_id): return False
        await knowledge_graph.delete_node(node_id)
        if node_id.startswith("mem-"):
            await hyperedge_vdb.delete_entity(node_id)
        elif node_id.startswith("ent-"):
            await entity_vdb.delete_entity(node_id)
        logger.info(f"[Storage] Deleted node: {node_id}")
        return True
    except Exception as e:
        logger.error(f"Delete failed {node_id}: {e}")
        return False

async def update_memory_node(
    node_id: str,
    knowledge_graph: Any,
    hyperedge_vdb: Any,
    new_content: str = None,
    new_type: str = None,
    new_happened_at: str = None
) -> bool:
    """
    更新记忆节点。
    策略：
    1. 优先更新 Graph (Source of Truth)。
    2. 根据是否修改 Content，决定 VectorDB 是执行 Upsert (全量替换+重新Embedding) 还是 update_attributes (仅更新元数据)。
    """
    try:
        # --- Step 1: 准备更新数据 ---
        updates = {}
        if new_content is not None: 
            updates["content"] = new_content
        if new_type is not None: 
            updates["memory_type"] = new_type
        if new_happened_at is not None: 
            updates["happened_at"] = new_happened_at
        
        # 如果没有任何更新，直接返回
        if not updates: 
            return False

        # --- Step 2: 更新 Graph (作为元数据的 Source of Truth) ---
        # 这一步必须先做，确保 Graph 里存的是最新、最全的状态
        graph_update_success = await knowledge_graph.update_node_attributes(node_id, updates)
        if not graph_update_success:
            logger.warning(f"[Operation] Failed to update graph node {node_id}. Aborting VDB update.")
            return False
        
        # --- Step 3: 更新 Vector DB ---
        # 仅针对记忆节点 (mem-) 进行向量库同步，实体节点 (ent-) 若有需要也可放开
        if node_id.startswith("mem-"):
            
            # Scenario A: Content 发生了变化 (混合/单独 均由此路径处理)
            # 必须 Re-embed，且必须防止 Metadata 丢失 (Critique 1 & 4)
            if new_content is not None:
                # 1. 从 Graph 中获取刚刚更新后的【完整】节点属性
                #    包含：content, memory_type, happened_at, created_at, call_count, last_called 等
                full_node_attrs = await knowledge_graph.get_node(node_id)
                
                if full_node_attrs:
                    vdb_payload = full_node_attrs.copy()
                    vdb_payload["content"] = new_content 
                    await hyperedge_vdb.upsert({node_id: vdb_payload})
                    logger.info(f"[Operation] VDB Upsert (Re-embed) for {node_id}")
                else:
                    logger.error(f"[Operation] Graph node {node_id} missing after update. VDB unsync.")

            # Scenario B: Content 没变，只变了 Metadata (Type 或 Time 或 混合) (Critique 2 & 3)
            # 走轻量级更新，不重新计算 Embedding
            else:
                meta_updates = {}
                if new_type is not None: 
                    meta_updates["memory_type"] = new_type
                if new_happened_at is not None: 
                    meta_updates["happened_at"] = new_happened_at
                
                if meta_updates:
                    # storage.py 的 update_attributes 内部实现了 Read-Modify-Write，安全。
                    if hasattr(hyperedge_vdb, "update_attributes"):
                        await hyperedge_vdb.update_attributes(node_id, meta_updates)
                        logger.info(f"[Operation] VDB Metadata Update for {node_id}: {list(meta_updates.keys())}")
                    else:
                        logger.warning(f"VectorDB backend does not support metadata update for {node_id}")

        logger.info(f"[Storage] Successfully updated node {node_id}: {list(updates.keys())}")
        return True

    except Exception as e:
        logger.error(f"Update failed for {node_id}: {e}", exc_info=True)
        return False

# =============================================================================
#  Retrieval Logic (Stateless)
# =============================================================================

async def increment_single_memory(mem_id: str, knowledge_graph: Any):
    now_ts = datetime.now(timezone.utc).isoformat()
    try:
        node = await knowledge_graph.get_node(mem_id)
        if node:
            prev = int(node.get("call_count", 0))
            await knowledge_graph.update_node_attributes(mem_id, {
                "call_count": prev + 1,
                "last_called": now_ts
            })
    except Exception:
        logger.debug("increment_single_memory update error")

async def increment_memory_usage(memory_ids: List[str], knowledge_graph: Any):
    tasks = [increment_single_memory(mem_id, knowledge_graph) for mem_id in memory_ids]
    await asyncio.gather(*tasks)

async def retrieve_hyperedge(
    query: str, 
    knowledge_graph: Any,
    entity_vdb: Any,
    hyperedge_vdb: Any,
    reranker_placeholder: Any, # [FIX] 为了保持接口兼容保留占位符，但实际上不使用对象
    global_config: dict,
) -> list[dict]:
    
    # [LOG]
    logger.info(f"[Retrieve] Start query: '{query[:50]}...'")

    # 1. Embedding Search
    # VDB 内部已经封装了 embedding lambda (在 MemoryManager.__post_init__ 中定义的纯函数闭包)
    query_embedding = None
    if hasattr(hyperedge_vdb, "embedding_func") and hyperedge_vdb.embedding_func:
        try:
            q_emb = await hyperedge_vdb.embedding_func([query])
            query_embedding = _np.asarray(q_emb[0], dtype="float32")
        except Exception: pass

    # 2. NER & Search
    graph_cfg = global_config.get("graph", {}) if isinstance(global_config.get("graph", {}), dict) else {}
    use_entity_links = bool(graph_cfg.get("use_entity_links", True))

    if use_entity_links:
        query_entities = await ner_from_text(query, global_config)
    else:
        query_entities = []

    entity_top_k = int(_get_query_setting(global_config, "cos_top_k_entity", 3))
    hyperedge_top_k = int(_get_query_setting(global_config, "cos_top_k_hyperedge", 5))

    entity_tasks = []
    if use_entity_links and entity_top_k > 0 and query_entities:
        entity_tasks = [entity_vdb.query(e, top_k=entity_top_k) for e in query_entities]

    he_task = hyperedge_vdb.query(query, top_k=hyperedge_top_k, query_embedding=query_embedding)

    results = await asyncio.gather(*entity_tasks, he_task, return_exceptions=True)
    
    candidates = []
    seen = set()
    for res_list in results:
        if isinstance(res_list, list):
            for item in res_list:
                if item["id"] not in seen:
                    candidates.append(item)
                    seen.add(item["id"])

    # [LOG]
    logger.info(f"[Retrieve] Step 1 (Vector/NER) found {len(candidates)} candidates.")
    if not candidates: return []

    # 3. Graph Traversal (1-Hop & 2-Hop)
    ents = [c for c in candidates if c["id"].startswith("ent-")]
    mems = [c for c in candidates if c["id"].startswith("mem-")]
    
    hop1_task = _graph_hop(knowledge_graph, [e["id"] for e in ents], "mem-")
    hop2_task = _graph_hop_two_step(knowledge_graph, [m["id"] for m in mems], "ent-", "mem-")

    use_temporal_edges = bool(_get_query_setting(global_config, "use_temporal_edges", True))
    temporal_neighbors_per_seed = int(_get_query_setting(global_config, "temporal_neighbors_per_seed", 2))
    graph_expand_max_candidates = int(_get_query_setting(global_config, "graph_expand_max_candidates", 0))
    graph_degree_penalty = float(_get_query_setting(global_config, "graph_degree_penalty", 0.0))

    hop_tasks = [hop1_task, hop2_task]
    if use_temporal_edges:
        hop_tasks.append(
            _graph_temporal_hop(
                knowledge_graph,
                [m["id"] for m in mems],
                max_neighbors_per_seed=temporal_neighbors_per_seed,
            )
        )

    hop_results = await asyncio.gather(*hop_tasks)
    hop1_ids = hop_results[0] if len(hop_results) > 0 else set()
    hop2_ids = hop_results[1] if len(hop_results) > 1 else set()
    temporal_ids = hop_results[2] if (use_temporal_edges and len(hop_results) > 2) else set()

    direct_mem_ids = {m["id"] for m in mems}
    candidate_ids = hop1_ids | hop2_ids | temporal_ids | direct_mem_ids
    candidate_order = sorted(candidate_ids)

    if graph_expand_max_candidates > 0 and len(candidate_ids) > graph_expand_max_candidates:
        priority: Dict[str, float] = {}
        for mid in direct_mem_ids:
            priority[mid] = priority.get(mid, 0.0) + 3.0
        for mid in hop1_ids:
            priority[mid] = priority.get(mid, 0.0) + 2.0
        for mid in hop2_ids:
            priority[mid] = priority.get(mid, 0.0) + 1.0
        for mid in temporal_ids:
            priority[mid] = priority.get(mid, 0.0) + 0.75
        for mid in candidate_ids:
            priority.setdefault(mid, 0.0)

        if graph_degree_penalty > 0 and candidate_ids:
            mids_for_degree = list(candidate_ids)
            edge_lists = await asyncio.gather(
                *[knowledge_graph.get_node_edges(mid) for mid in mids_for_degree],
                return_exceptions=True,
            )
            for mid, edges in zip(mids_for_degree, edge_lists):
                deg = len(edges) if isinstance(edges, list) else 0
                priority[mid] = priority.get(mid, 0.0) - graph_degree_penalty * min(deg, 50)

        ranked = sorted(priority.items(), key=lambda kv: (-kv[1], kv[0]))
        candidate_order = [mid for mid, _ in ranked[:graph_expand_max_candidates]]
        candidate_ids = set(candidate_order)

    # [LOG]
    logger.info(
        f"[Retrieve] Step 2 (Graph Expansion) result: {len(candidate_ids)} candidates "
        f"(temporal={len(temporal_ids)})."
    )

    # 4. Fetch & Rerank (Stateless)
    final_mems = []
    if candidate_ids:
        nodes = await asyncio.gather(*[knowledge_graph.get_node(nid) for nid in candidate_order])

        texts, meta_list = [], []
        for nid, node in zip(candidate_order, nodes):
            if node and node.get("content"):
                meta_list.append({
                    "id": nid,
                    "content": node.get("content", ""),
                    "source_id": node.get("source_id", ""),
                    "dia_id": node.get("dia_id", ""),
                    "dia_session_id": node.get("dia_session_id", None),
                    "call_count": node.get("call_count", 0),
                    "memory_type": node.get("memory_type", ""),
                    "created_at": node.get("created_at", ""),
                    "last_called": node.get("last_called", ""), # System Access Time
                    "happened_at": node.get("happened_at", ""), # Semantic Content Time
                })
                texts.append(node.get("content"))
        
        if texts:
            # [FIX]: 使用无状态函数调用进行 Rerank
            ranked = await rerank_texts(
                query=query,
                texts=texts,
                config=global_config,
                top_k=int(_get_query_setting(global_config, "rerank_top_k_hyperedge", 5)),
            )
            # ranked return: List[Tuple[int, str, float]]
            final_mems = [meta_list[idx] for idx, _, _ in ranked]

            # Temporal post-ordering for when/first/latest questions.
            t_mode = _temporal_query_mode(query)
            if t_mode and final_mems:
                with_ts: List[tuple[datetime, dict]] = []
                without_ts: List[dict] = []
                for mem in final_mems:
                    dt = _memory_event_time(mem)
                    if dt is None:
                        without_ts.append(mem)
                    else:
                        with_ts.append((dt, mem))

                if with_ts:
                    if t_mode == "earliest":
                        with_ts.sort(key=lambda x: x[0])
                    elif t_mode == "latest":
                        with_ts.sort(key=lambda x: x[0], reverse=True)
                    else:
                        # Generic "when" query: keep rerank order, only prioritize timestamped entries.
                        pass

                    final_mems = [m for _, m in with_ts] + without_ts

            # Update access stats
            await increment_memory_usage([m["id"] for m in final_mems], knowledge_graph)

    # [LOG]
    logger.info(f"[Retrieve] Step 3 (Rerank) Final result: {len(final_mems)} hyperedges.")
    return final_mems

async def retrieve_chunk(
    query: str, 
    chunk_ids: set, 
    global_config: dict,
    text_chunks_db: Any,
    reranker_placeholder: Any # [FIX] 占位符
) -> list[dict]:
    if not chunk_ids: return []
    
    chunks = await text_chunks_db.get_by_ids(list(chunk_ids))
    valid_chunks = [c for c in chunks if c and c.get("content")]
    
    if not valid_chunks: return []
    
    texts = [c["content"] for c in valid_chunks]
    
    # [FIX]: 使用无状态函数调用进行 Rerank
    ranked = await rerank_texts(
        query=query,
        texts=texts,
        config=global_config,
        top_k=int(_get_query_setting(global_config, "rerank_top_k_chunk", 3))
    )
    
    logger.info(f"[Retrieve] Chunks: {len(valid_chunks)} -> Reranked to {len(ranked)}")
    return [valid_chunks[idx] for idx, _, _ in ranked]

async def _graph_temporal_hop(storage: Any, start_ids, max_neighbors_per_seed: int = 2) -> Set[str]:
    if not start_ids:
        return set()

    out: Set[str] = set()
    for nid in start_ids:
        try:
            edges = await storage.get_node_edges(nid)
        except Exception:
            edges = []

        local = []
        for s, t, d in edges:
            other = t if s == nid else s
            if not (isinstance(other, str) and other.startswith("mem-")):
                continue
            if isinstance(d, dict) and d.get("relation") == "temporal_next":
                local.append(other)

        local = sorted(set(local))
        if max_neighbors_per_seed > 0:
            local = local[:max_neighbors_per_seed]
        out.update(local)

    return out


async def _graph_hop(storage: Any, start_ids, target_prefix) -> Set[str]:
    if not start_ids: return set()
    tasks = [storage.get_adjacent_nodes(nid, target_prefix=target_prefix) for nid in start_ids]
    results = await asyncio.gather(*tasks)
    return set().union(*results)

async def _graph_hop_two_step(storage: Any, start_ids, p1, p2) -> Set[str]:
    step1 = await _graph_hop(storage, start_ids, p1)
    if not step1: return set()
    return await _graph_hop(storage, list(step1), p2)
