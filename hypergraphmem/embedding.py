# embedding.py
import aiohttp
import asyncio
import random
import logging
import numpy as np
from typing import List, Union, Dict, Any

logger = logging.getLogger(__name__)

async def get_embeddings(texts: Union[str, List[str]], config: Dict[str, Any]) -> np.ndarray:
    """
    Stateless Embedding Call with Exponential Backoff Retry.
    """
    embedding_cfg = config.get("embedding", {})
    base_url = embedding_cfg.get("base_url")
    model_name = embedding_cfg.get("model_name")
    
    # Retry Configuration
    max_retries = embedding_cfg.get("max_retries", 5)
    base_delay = embedding_cfg.get("retry_delay", 1.0)
    timeout_sec = embedding_cfg.get("timeout", 60)
    
    if not base_url:
        raise ValueError("Embedding base_url not configured.")

    if isinstance(texts, str):
        texts = [texts]

    payload = {
        "model": model_name,
        "input": texts,
        "encoding_format": "float"
    }

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(base_url, json=payload, timeout=timeout_sec) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Handle standard OpenAI embedding format
                        embeddings = [item["embedding"] for item in data["data"]]
                        return np.array(embeddings)
                    
                    # If server is busy (503) or rate limited (429), we should retry
                    # If it's a 4xx error (other than 429), it might be a bad request, but we'll log it.
                    err_text = await resp.text()
                    logger.warning(f"[Embedding] Request failed (Attempt {attempt+1}/{max_retries}). Status: {resp.status}, Error: {err_text}")
                    
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"[Embedding] Connection error (Attempt {attempt+1}/{max_retries}): {e}")
        except Exception as e:
            logger.error(f"[Embedding] Unexpected error: {e}", exc_info=True)
            # For unexpected errors, we might stop, but here we keep retrying to be robust
            pass
        
        # Exponential Backoff with Jitter
        if attempt < max_retries - 1:
            delay = (base_delay * (2 ** attempt)) + (random.uniform(0, 1))
            logger.info(f"[Embedding] Retrying in {delay:.2f}s...")
            await asyncio.sleep(delay)
    
    # If all retries fail
    raise RuntimeError(f"Embedding failed after {max_retries} attempts.")