import os
import json
import logging
import asyncio
import random
import aiohttp
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# =============================================================================
# 全局单例配置 (Singleton Configuration)
# =============================================================================
# 控制并发数：防止并发过高导致 429 或者本地 TCP 耗尽
# 建议根据你的 API 供应商限制调整，例如 OpenAI Tier 1 一般能承受 20-50 并发
MAX_CONCURRENT_REQUESTS = 32 
_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# 全局 Session：复用 TCP 连接，显著减少握手耗时
_GLOBAL_SESSION: Optional[aiohttp.ClientSession] = None

async def get_global_session():
    global _GLOBAL_SESSION
    if _GLOBAL_SESSION is None or _GLOBAL_SESSION.closed:
        # 设置连接池参数
        connector = aiohttp.TCPConnector(
            limit=MAX_CONCURRENT_REQUESTS + 10, # 连接池限制略大于并发限制
            ttl_dns_cache=300,
            enable_cleanup_closed=True
        )
        # 设置通用的超时，防止僵尸连接
        timeout = aiohttp.ClientTimeout(total=300, connect=10)
        _GLOBAL_SESSION = aiohttp.ClientSession(connector=connector, timeout=timeout)
    return _GLOBAL_SESSION

async def close_global_session():
    """程序退出时调用"""
    global _GLOBAL_SESSION
    if _GLOBAL_SESSION and not _GLOBAL_SESSION.closed:
        await _GLOBAL_SESSION.close()

# =============================================================================
# LLM Request Function
# =============================================================================

async def llm_chat_request(
    prompt: str,
    config: Dict[str, Any],
    system_prompt: Optional[str] = None,
    history_messages: Optional[List[dict]] = None,
    **kwargs
) -> str:
    """
    Stateless LLM Call using raw HTTP with Semaphore and Session Reuse.
    """
    llm_cfg = config.get("llm", {})
    base_url = llm_cfg.get("base_url")
    api_key = llm_cfg.get("api_key")
    model_name = llm_cfg.get("model_name")

    # Retry & Timeout Configuration
    max_retries = int(llm_cfg.get("max_retries", 5))
    base_delay = float(llm_cfg.get("retry_delay", 2.0))
    # 这里的 timeout 是指“请求发出后”等待响应的最长时间
    request_timeout = float(llm_cfg.get("timeout", 120)) 

    if not base_url:
        raise ValueError("LLM base_url is not configured.")

    # Determine generation parameters
    defaults = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048}
    gen_max = int(llm_cfg.get("max_token", 2048))
    defaults["max_tokens"] = gen_max
    
    # Merge parameters
    payload_params = {**defaults, **kwargs}
    extra_body = payload_params.pop("extra_body", None)

    # Construct Messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_name,
        "messages": messages,
        **payload_params
    }
    if extra_body:
        payload.update(extra_body)

    # Raw HTTP Request Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    target_url = f"{base_url.rstrip('/')}/chat/completions"
    if "v1" not in target_url and "v1" not in base_url:
         target_url = f"{base_url.rstrip('/')}/v1/chat/completions"

    # 获取全局 Session
    session = await get_global_session()

    # [CRITICAL] 核心逻辑：使用 Semaphore 限制并发
    # 只有拿到“令牌”的任务才会进入网络请求阶段
    # 排队的任务会在这里 await，不会触发 HTTP Timeout，也不会占用连接数
    async with _SEMAPHORE:
        for attempt in range(max_retries):
            try:
                # 使用 ClientTimeout 覆盖 Session 默认超时
                # post 请求复用 session
                async with session.post(
                    target_url, 
                    json=payload, 
                    headers=headers, 
                    timeout=aiohttp.ClientTimeout(total=request_timeout)
                ) as resp:
                    
                    if resp.status == 200:
                        res_json = await resp.json()
                        if "choices" in res_json and len(res_json["choices"]) > 0:
                            return res_json["choices"][0]["message"]["content"]
                        return ""
                    
                    # Handle Non-200 responses
                    # 如果是 429 (Rate Limit)，通常应该多等一会儿
                    if resp.status == 429:
                        logger.warning(f"[LLM] Rate Limit Hit (429). Attempt {attempt+1}")
                        # 429 强制增加延迟
                        await asyncio.sleep(5 + random.uniform(0, 2))
                        continue

                    err = await resp.text()
                    logger.warning(f"[LLM] Request Failed (Attempt {attempt+1}/{max_retries}). Status: {resp.status} | Error: {err[:200]}")
                    
            except asyncio.TimeoutError:
                 logger.warning(f"[LLM] Timeout after {request_timeout}s (Attempt {attempt+1}/{max_retries}). Server might be overloaded.")
            except aiohttp.ClientError as e:
                logger.warning(f"[LLM] Connection Error (Attempt {attempt+1}/{max_retries}): {e}")
            except Exception as e:
                logger.error(f"[LLM] Unexpected Error: {e}", exc_info=True)

            # Exponential Backoff with Jitter
            if attempt < max_retries - 1:
                delay = (base_delay * (2 ** attempt)) + (random.uniform(0, 1))
                # logger.info(f"[LLM] Retrying in {delay:.2f}s...") 
                await asyncio.sleep(delay)

    logger.error("[LLM] Max retries exhausted. Returning empty string.")
    return ""