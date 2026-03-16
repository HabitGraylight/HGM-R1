# hypergraphmem_agent_loop.py
# Copyright 2025 Bytedance Ltd. and/or its affiliates
import asyncio
import copy
import json
import logging
import os
from pathlib import Path
from enum import Enum
from typing import Any, List, Dict
from dataclasses import dataclass, field
from uuid import uuid4

from hypergraphmem.hypergraphmem import LayeredMemoryManager
from hypergraphmem_agent.agent.hypergraphmem_executor import HyperGraphMemExecutor
from hypergraphmem_agent.agent.prompt import build_policy_prompt, POLICY_JSON_SCHEMA

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.experimental.agent_loop.agent_loop import DictConfigWrap, AsyncLLMServerManager
from transformers import AutoTokenizer, AutoProcessor

import yaml 
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HGM_CONFIG_PATH = REPO_ROOT / "hypergraphmem_config.yaml"

@dataclass
class AgentData:
    chunks: List[Any]
    request_id: str
    executor: HyperGraphMemExecutor
    memory_manager: LayeredMemoryManager
    
    # 存储整个 Episode 的 Token 序列，用于 PPO 训练
    prompt_ids: List[int] = field(default_factory=list)
    response_ids: List[int] = field(default_factory=list)
    response_mask: List[int] = field(default_factory=list)
    response_logprobs: List[float] = field(default_factory=list)

    turn_lengths: List[int] = field(default_factory=list)
    
    # 当前轮次的临时状态
    current_turn_input_ids: List[int] = field(default_factory=list)
    current_turn_response_ids: List[int] = field(default_factory=list)
    
    # 游标与上下文
    current_chunk_idx: int = 0
    current_facts: List[Dict[str, Any]] = field(default_factory=list)
    current_retrieved_context: str = ""
    current_reference_time: str = None
    current_source_chunk_id: str = None 
    current_id_map: Dict[str, str] = field(default_factory=dict)
    
    metrics: Dict[str, Any] = field(default_factory=dict)
    op_trace: List[Dict[str, Any]] = field(default_factory=list)


@register("hypergraphmem_tool_agent")
class HyperGraphMemToolAgentLoop(AgentLoopBase):
    
    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        self.raw_config = trainer_config.config
        
        # Load HGM Config
        self.hgm_config = {}
        if hasattr(self.raw_config, "hypergraphmem_config"):
            self.hgm_config = self.raw_config.hypergraphmem_config
        
        config_path = Path(os.getenv("HYPERGRAPHMEM_CONFIG_PATH", str(DEFAULT_HGM_CONFIG_PATH)))
        if not self.hgm_config:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        self.hgm_config = yaml.safe_load(f)
                except Exception as e:
                    logger.error(f"[Agent] Config load failed: {e}")

        if not isinstance(self.hgm_config, dict):
            try: self.hgm_config = dict(self.hgm_config)
            except: self.hgm_config = {}

        self.memory_root_dir = self.hgm_config.get("working_dir", "./memory_workspace")
        # 确保目录存在
        os.makedirs(self.memory_root_dir, exist_ok=True)
        
        logger.info(f"[Agent] Ready. Root: {self.memory_root_dir}")

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        chunks = kwargs.get("chunks")
        qa_pairs = kwargs.get("qa_pairs")
        uid = kwargs.get("instance_id", "unknown")
        # 为每个 Episode 创建唯一的 Instance ID
        instance_id = f"{uid}_{uuid4().hex}"

        # 解析输入
        if isinstance(chunks, str):
            chunks = json.loads(chunks)
        if isinstance(qa_pairs, str):
            qa_pairs = json.loads(qa_pairs)
        
        if not chunks: 
            return AgentLoopOutput(
                prompt_ids=[], response_ids=[], response_mask=[], 
                response_logprobs=[], num_turns=0, metrics={}, extra_fields={"error": "no_chunks"}
            )

        # 为该 Instance 创建独立的 Working Directory
        working_dir = os.path.join(self.memory_root_dir, f"inst_{instance_id}")
        
        # 初始化 Memory Manager 和 Executor
        mm = LayeredMemoryManager(working_dir=working_dir, config=self.hgm_config)
        executor = HyperGraphMemExecutor(mm)
        
        agent_data = AgentData(
            chunks=chunks,
            request_id=instance_id,
            executor=executor,
            memory_manager=mm
        )

        logger.info(f"[Agent Loop] Starting instance {instance_id} with {len(chunks)} chunks.")

        # --- Phase 1: Build Memory Loop (Process Chunks) ---
        # 这里的核心逻辑修改为遍历所有 chunks，每一轮都是独立的 prompt 构建
        try:
            for idx, chunk_raw in enumerate(chunks):
                agent_data.current_chunk_idx = idx
                
                # 1. Prepare: Extract Facts & Retrieve Context (Memory State)
                await self._step_prepare_context(agent_data, chunk_raw)
                
                # 2. Generate: Build Fresh Prompt & Call Policy
                # 这里的 Prompt 是 [System, Memory_State(Context), Chunk]
                await self._step_generate_ops(agent_data, sampling_params)
                
                # 3. Execute: Parse & Apply Operations
                await self._step_execute_ops(agent_data)
                
        except Exception as e:
            logger.error(f"[Agent Run] Loop Error at chunk {agent_data.current_chunk_idx}: {e}", exc_info=True)

        # --- Phase 2: Retrieve Context for Evaluation ---
        # 基于构建好的 Memory Graph，为每个问题检索上下文
        eval_contexts = []
        if qa_pairs:
            logger.info(f"[{instance_id}] Retrieving context for {len(qa_pairs)} QA pairs...")
            for qa in qa_pairs:
                question = qa.get("question")
                if question:
                    try:
                        context = await executor.retrieve_for_eval(question)
                        eval_contexts.append(context)
                    except Exception as e:
                        logger.warning(f"Retrieval failed for Q: {question[:30]}... {e}")
                        eval_contexts.append("")
                else:
                    eval_contexts.append("")
        
        # Cleanup
        await executor.destroy()

        # Pack Data for Reward Manager
        extra_fields = {
            "instance_id": instance_id,
            "qa_pairs": qa_pairs,              
            "eval_contexts": eval_contexts,  
            "op_trace": agent_data.op_trace,
            "chunks_processed": agent_data.current_chunk_idx + 1,
            "turn_lengths": agent_data.turn_lengths,
        }

        # [LOGGING] Final summary
        total_tokens = len(agent_data.prompt_ids) + len(agent_data.response_ids)
        logger.info(f"[Agent Loop] Finished {instance_id}. Processed Chunks: {len(chunks)}. Total Tokens: {total_tokens}.")

        output = AgentLoopOutput(
            prompt_ids=agent_data.prompt_ids,
            response_ids=agent_data.response_ids,
            response_mask=agent_data.response_mask,
            response_logprobs=agent_data.response_logprobs,
            num_turns=agent_data.current_chunk_idx + 1,
            metrics=agent_data.metrics,
            extra_fields=extra_fields
        )
        return output

    async def _step_prepare_context(self, agent_data: AgentData, chunk_raw: Any):
        """
        Step 1: 摄入 Raw Chunk，提取事实，并基于事实检索当前 Memory 状态。
        """
        chunk_text = chunk_raw if isinstance(chunk_raw, str) else chunk_raw.get("text", "")
        chunk_ts = None if isinstance(chunk_raw, str) else chunk_raw.get("timestamp")
        agent_data.current_reference_time = chunk_ts

        with simple_timer("extract_and_retrieve", agent_data.metrics):
            # 调用 Executor 获取当前状态
            context = await agent_data.executor.extract_and_retrieve(chunk_text, chunk_ts)
        
        agent_data.current_facts = context["facts"]
        agent_data.current_retrieved_context = context["retrieved_context"]
        agent_data.current_source_chunk_id = context["stored_chunk_id"]
        agent_data.current_id_map = context.get("id_map", {})
        
        logger.info(f"[Step {agent_data.current_chunk_idx}] Context Prepared. "
                    f"Facts extracted: {len(context['facts'])}, "
                    f"Memory State Size: {len(context['retrieved_context'])} chars.")

    async def _step_generate_ops(self, agent_data: AgentData, sampling_params: dict):
        """
        Step 2: 构建 Prompt 并生成。
        关键点：Prompt 是根据当前状态全新构建的，不包含之前的 Chat History Token。
        """
        from vllm.sampling_params import GuidedDecodingParams

        # 1. 构建全新的 Prompt
        # 这确保了模型看到的是 [System, Current_Mem_State, Current_Chunk]
        prompt_str = build_policy_prompt(
            reference_time=agent_data.current_reference_time,
            system_time=datetime.now().isoformat(),
            facts=agent_data.current_facts,
            retrieved_context=agent_data.current_retrieved_context, # 这是 Mem_State
        )
        
        # 2. Tokenize (注意：不加 Special Tokens 防止多次 BOS，或者视 Tokenizer 而定)
        current_turn_input_ids = await self._tokenize(prompt_str)
        agent_data.current_turn_input_ids = current_turn_input_ids

        agent_data.response_ids.extend(current_turn_input_ids)
        agent_data.response_mask.extend([0] * len(current_turn_input_ids))
        agent_data.response_logprobs.extend([0.0] * len(current_turn_input_ids))

        # 4. 配置采样参数 (JSON Schema)
        json_schema = POLICY_JSON_SCHEMA.get("json_schema", {}).get("schema", {})
        op_sampling_params = copy.deepcopy(sampling_params)
        op_sampling_params["guided_decoding"] = GuidedDecodingParams(json=json_schema)
        op_sampling_params["temperature"] = 0.2 
        
        # 5. 调用模型生成
        # 注意：这里我们只传入 current_turn_input_ids，实现了上下文的独立性
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=current_turn_input_ids,
            sampling_params=op_sampling_params
        )
        
        generated_ids = output.token_ids
        agent_data.current_turn_response_ids = generated_ids
        
        agent_data.response_ids.extend(generated_ids)
        agent_data.response_mask.extend([1] * len(generated_ids))
        
        if output.log_probs:
            agent_data.response_logprobs.extend(output.log_probs)
        else:
            agent_data.response_logprobs.extend([0.0] * len(generated_ids))

        current_block_len = len(current_turn_input_ids) + len(generated_ids)
        agent_data.turn_lengths.append(current_block_len)

        # [LOGGING]
        logger.info(f"[Step {agent_data.current_chunk_idx}] Generated {len(generated_ids)} tokens.")

    async def _step_execute_ops(self, agent_data: AgentData):
        """
        Step 3: 执行操作，更新 Memory 状态。
        """
        if not agent_data.current_turn_response_ids:
            return

        current_response_text = self.tokenizer.decode(
            agent_data.current_turn_response_ids, 
            skip_special_tokens=True
        )
        
        raw_ops = self._parse_json_from_text(current_response_text)
        
        resolved_ops = []
        for op in raw_ops:
            new_op = op.copy()
            raw_mem_id = new_op.get("memory_id")
            action = new_op.get("action", "").upper()
            # ID 映射 (Short ID -> UUID)
            if raw_mem_id:
                short_id = str(raw_mem_id)
                if short_id in agent_data.current_id_map:
                    new_op["memory_id"] = agent_data.current_id_map[short_id]
                    resolved_ops.append(new_op)
                else:
                    logger.warning(f"[Agent Loop] Invalid memory_id '{short_id}' ignored.")
            elif action == "ADD":
                resolved_ops.append(new_op)
            else:
                logger.warning(f"[Agent Loop] Action {action} missing memory_id. Skipped.")

        # 执行并记录 Trace
        with simple_timer("execute_ops", agent_data.metrics):
            results = await agent_data.executor.execute_operations(
                operations=resolved_ops,
                timestamp=agent_data.current_reference_time,
                source_chunk_id=agent_data.current_source_chunk_id
            )
            
        agent_data.op_trace.extend(results)
        
        # 清理当前轮次状态
        agent_data.current_turn_input_ids = []
        agent_data.current_turn_response_ids = []
        agent_data.current_id_map = {}

    async def _tokenize(self, text: str) -> List[int]:
        return await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(text, add_special_tokens=False)
        )

    def _parse_json_from_text(self, text: str) -> List[Dict]:
        try:
            text = text.strip()
            # 简单的 JSON 提取逻辑
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                candidate = text[start : end + 1]
                data = json.loads(candidate)
                return data.get("operations", [])
            data = json.loads(text)
            if isinstance(data, dict): return data.get("operations", [])
        except Exception: 
            pass
        return []