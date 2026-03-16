# hypergraphmem_executor.py
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import copy

from hypergraphmem.hypergraphmem import LayeredMemoryManager
from hypergraphmem.operation import format_memory_context

logger = logging.getLogger(__name__)

class HyperGraphMemExecutor:
    def __init__(self, memory_manager: LayeredMemoryManager):
        self.memory_manager = memory_manager
        logger.info("[Executor] Ready (Stateless Mode).")

    async def extract_and_retrieve(self, chunk: str, timestamp: str) -> Dict[str, Any]:
        facts_struct = []
        retrieved_context = ""
        stored_chunk_id = None
        id_map = {}

        if chunk and chunk.strip():
            try:
                # 1. Ingest raw chunk first (Creates Source)
                stored_chunk_id = await self.memory_manager.ingest_source_chunk(
                    text=chunk, timestamp=timestamp
                )

                # 2. Extract facts from it
                # Operation 层会内部调用无状态的 LLM 函数，无需 Executor 干预
                extracted_dicts = await self.memory_manager.aextract_memories_from_text(
                    chunk, str(stored_chunk_id), timestamp=timestamp
                )
                facts_struct = extracted_dicts 
                
                # [LOGGING] Trace extracted facts count
                logger.info(f"[Executor] Chunk {stored_chunk_id}: Extracted {len(facts_struct)} facts.")

            except Exception as e:
                logger.error(f"[Executor] Ingest/Extract failed: {e}", exc_info=True)

        # 3. Retrieve related context based on facts
        if facts_struct:
            try:
                fact_contents = [d["content"] for d in facts_struct]
                tasks = [self.memory_manager.retrieve_hyperedges_list(f) for f in fact_contents]
                results = await asyncio.gather(*tasks)

                unique_hyperedges = []
                seen = set()
                for res_list in results:
                    for he in res_list:
                        if he["id"] not in seen:
                            unique_hyperedges.append(he)
                            seen.add(he["id"])
                
                # [LOGGING] Trace retrieved context size
                logger.info(f"[Executor] Retrieved context: {len(unique_hyperedges)} unique hyperedges.")
                
                display_hyperedges = []
                for idx, he in enumerate(unique_hyperedges):
                    short_id = str(idx)
                    long_id = he["id"]
                    id_map[short_id] = long_id
                    
                    he_display = copy.deepcopy(he)
                    he_display["id"] = short_id 
                    display_hyperedges.append(he_display)
                
                retrieved_context = format_memory_context(
                    {"hyperedges": display_hyperedges}, include_sources=False
                )

            except Exception as e:
                logger.error(f"[Executor] Retrieval failed: {e}", exc_info=True)

        return {
            "facts": facts_struct,
            "retrieved_context": retrieved_context,
            "stored_chunk_id": stored_chunk_id,
            "id_map": id_map
        }

    async def execute_operations(self, operations: List[Dict], timestamp: str, source_chunk_id: str) -> List[Dict]:
        results = []
        success_count = 0
        
        for op in operations:
            action = op.get("action", "").upper()
            
            # 提取通用字段
            mem_id = op.get("memory_id")
            
            # 使用 op 自带的 happened_at，如果没有则回退到当前参考时间
            op_happened_at = op.get("happened_at") or timestamp
            
            success = False
            
            try:
                if action == "ADD":
                    content = op.get("content", "").strip()
                    # 从 JSON 中获取 type，默认为 short_term
                    m_type = op.get("type", "short_term") 
                    
                    if content:
                        await self.memory_manager.add_memory_fact(
                            content=content,
                            memory_type=m_type,
                            timestamp=op_happened_at,
                            source_chunk_id=source_chunk_id
                        )
                        success = True
                    else:
                        logger.warning(f"[Executor] ADD action missing content.")

                elif action == "DELETE":
                    if mem_id:
                        success = await self.memory_manager.delete_memory_by_id(mem_id)
                    else:
                        logger.warning(f"[Executor] DELETE action missing memory_id.")

                elif action == "UPDATE":
                    if mem_id:
                        # 提取 UPDATE 特有的可选字段
                        new_content = op.get("new_content")
                        new_type = op.get("new_type")
                        new_happened_at = op.get("new_happened_at") # 可能为 None

                        # 如果没有提供 new_happened_at，不要覆盖原有的时间，传 None 即可
                        success = await self.memory_manager.update_memory_content(
                            memory_id=mem_id, 
                            new_content=new_content,
                            new_type=new_type,
                            new_happened_at=new_happened_at
                        )
                    else:
                        logger.warning(f"[Executor] UPDATE action missing memory_id.")
                
                else:
                    logger.warning(f"[Executor] Unknown action: {action}")

            except Exception as e:
                logger.error(f"[Executor] Op failed: {action}: {e}", exc_info=True)

            if success:
                success_count += 1
            results.append({"action": action, "success": success, "details": str(op)})
            
        # [LOGGING] Trace execution success rate
        if operations:
            logger.info(f"[Executor] Executed {len(operations)} ops. Success: {success_count}/{len(operations)}.")

        # [DEBUG ADDITION] Force save after every execution batch to visualize files
        try:
            await self.memory_manager.close()
            logger.info("[Executor] 💾 Debug Snapshot saved to disk.")
        except Exception as e:
            logger.warning(f"Debug save failed: {e}")
            
        return results

    async def retrieve_for_eval(self, query: str, eval_mode: str = "hybrid") -> str:
        try:
            ctx = await self.memory_manager.retrieve_context_dict(query)
            
            # [LOGGING] Trace eval retrieval stats
            num_edges = len(ctx.get("hyperedges", []))
            num_sources = len(ctx.get("sources", []))
            logger.info(f"[Executor Eval] Query: '{query[:30]}...' -> Edges: {num_edges}, Sources: {num_sources}")

            if eval_mode == "hyperedge":
                ctx["sources"] = []
            elif eval_mode == "chunk":
                ctx["hyperedges"] = []
            
            return format_memory_context(ctx, include_sources=True)
        except Exception as e:
            logger.error(f"Eval retrieval failed: {e}")
            return ""

    async def destroy(self):
        await self.memory_manager.destroy()