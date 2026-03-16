# utils.py

import asyncio
import html
import json
import logging
import os
import re
from dataclasses import dataclass
from hashlib import md5
from typing import Any, Optional, Callable

import numpy as np
import tiktoken

# --- 1. 日志与并发控制 ---

logger = logging.getLogger("hypergraphrag_agent")
# If no handlers are configured (e.g., in tests or when imported as a library),
# add a default console StreamHandler so important INFO/DEBUG messages are visible
# on the terminal. This avoids silently dropping logs when set_logger() is not used.
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)
ENCODER = None

class UnlimitedSemaphore:
    async def __aenter__(self): pass
    async def __aexit__(self, exc_type, exc, tb): pass

def limit_async_func_call(concurrency: int):
    """Return a decorator that limits concurrent calls to an async or sync function.

    Usage:
        wrapped = limit_async_func_call(8)(some_func)

    The returned wrapper is always async. If the original function is sync, it
    will be executed in the default threadpool via run_in_executor.
    """
    sem = asyncio.Semaphore(concurrency) if concurrency and concurrency > 0 else None

    def decorator(fn):
        if asyncio.iscoroutinefunction(fn):
            async def _wrapped(*args, **kwargs):
                if sem:
                    async with sem:
                        return await fn(*args, **kwargs)
                return await fn(*args, **kwargs)

            return _wrapped

        # sync function -> wrap to async and run in threadpool
        async def _wrapped_sync(*args, **kwargs):
            loop = asyncio.get_event_loop()
            if sem:
                async with sem:
                    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
            return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

        return _wrapped_sync

    return decorator

# --- 2. 核心工具函数 (哈希, Token处理, 字符串操作) ---

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """根据内容计算MD5哈希值，生成确定性的ID。"""
    return prefix + md5(content.encode('utf-8')).hexdigest()

def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o") -> list[int]:
    """使用tiktoken将字符串编码为token ID列表。"""
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    return ENCODER.encode(content)

def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o") -> str:
    """使用tiktoken将token ID列表解码为字符串。"""
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    return ENCODER.decode(tokens)

def clean_str(input_val: Any) -> str:
    """清理字符串，移除HTML转义、控制字符等，并进行基本规范化。
    
    - 去除首尾空白
    - 解码HTML转义字符
    - 移除控制字符
    - 转换为小写
    """
    if not isinstance(input_val, str):
        return str(input_val) if input_val is not None else ""
    result = html.unescape(input_val.strip())
    result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
    return result.lower().strip()


def normalize_entity(entity_name: Any) -> str:
    """规范化实体名称，用于统一entity的存储和检索。
        normalize_entity("Amy's") -> "amy"
        normalize_entity('"Beijing"') -> "beijing"
        normalize_entity("New  York") -> "new york"
    """
    if entity_name is None:
        return ""
    
    result = clean_str(entity_name)
    
    # 移除所有格: 's 或 's
    result = re.sub(r"[''']s\b", "", result)
    
    # 移除首尾引号
    result = re.sub(r'^["\'\'\"]+|["\'\'\"]+$', '', result)
    
    # 移除其他常见噪声符号，但保留基本标点
    result = re.sub(r'[""„‟«»‹›]', '', result)
    
    # 合并连续空格
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

# --- 3. 数据格式化与文件I/O ---

def load_json(file_name: str):
    """从文件加载JSON对象。"""
    if not os.path.exists(file_name):
        return None
    with open(file_name, 'r', encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name: str):
    """将JSON对象写入文件。

    This function is defensive: it will ensure the parent directory exists,
    write to a temporary file and atomically replace the destination. Any
    exception is logged and re-raised so callers can observe failures.
    """
    try:
        dir_name = os.path.dirname(file_name)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        tmp_name = file_name + ".tmp"
        with open(tmp_name, 'w', encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2, ensure_ascii=False)
        # atomic replace
        os.replace(tmp_name, file_name)
    except Exception as e:
        # log the full traceback via logger
        try:
            logger.exception(f"Failed to write JSON to {file_name}: {e}")
        except Exception:
            # if logger itself fails for some reason, fallback to stderr
            import sys
            print(f"Failed to write JSON to {file_name}: {e}", file=sys.stderr)
        raise