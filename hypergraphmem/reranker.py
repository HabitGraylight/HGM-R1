# reranker.py
import aiohttp
import logging
import asyncio
import random
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

async def rerank_texts(
    query: str, 
    texts: List[str], 
    config: Dict[str, Any], 
    top_k: int = None
) -> List[Tuple[int, str, float]]:
    """
    Stateless Rerank Call with Exponential Backoff Retry.
    """
    if not texts:
        return []

    reranker_cfg = config.get("reranker", {})
    base_url = reranker_cfg.get("base_url")
    model_name = reranker_cfg.get("model_name", "bge-reranker")
    api_key = reranker_cfg.get("api_key", "empty")
    
    # Retry Configuration
    max_retries = int(reranker_cfg.get("max_retries", 5))
    base_delay = float(reranker_cfg.get("retry_delay", 1.0))
    timeout_sec = float(reranker_cfg.get("timeout", 60))

    if not base_url:
        logger.warning("Reranker base_url missing.")
        return [(i, t, 0.0) for i, t in enumerate(texts)][:top_k]

    # Support both legacy reranker payload (text_1/text_2) and wrapper payload (query/documents).
    if str(base_url).rstrip("/").endswith("/v1/rerank"):
        payload = {
            "model": model_name,
            "query": query,
            "documents": texts,
            "top_k": top_k or len(texts),
        }
    else:
        payload = {
            "model": model_name,
            "text_1": query,
            "text_2": texts,
        }
    headers = {"Content-Type": "application/json"}
    if api_key != "empty":
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(base_url, json=payload, headers=headers, timeout=timeout_sec) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        scores = []
                        
                        # Parsing logic
                        if isinstance(data, list):
                            for i, score in enumerate(data):
                                val = score if isinstance(score, (int, float)) else score.get("score", 0.0)
                                scores.append((i, texts[i], float(val)))
                        elif isinstance(data, dict):
                            results = data.get("results", []) or data.get("data", [])
                            if results:
                                for item in results:
                                    idx = item.get("index")
                                    score = item.get("relevance_score", item.get("score", 0.0))
                                    if idx is not None and idx < len(texts):
                                        scores.append((idx, texts[idx], float(score)))
                        
                        scores.sort(key=lambda x: x[2], reverse=True)
                        if top_k:
                            return scores[:top_k]
                        return scores
                    
                    # Log failure and continue to retry
                    err_text = await resp.text()
                    logger.warning(f"[Reranker] Request failed (Attempt {attempt+1}/{max_retries}). Status: {resp.status}, Error: {err_text}")

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"[Reranker] Connection error (Attempt {attempt+1}/{max_retries}): {e}")
        except Exception as e:
            logger.error(f"[Reranker] Unexpected error: {e}", exc_info=True)

        # Backoff logic
        if attempt < max_retries - 1:
            delay = (base_delay * (2 ** attempt)) + (random.uniform(0, 1))
            logger.info(f"[Reranker] Retrying in {delay:.2f}s...")
            await asyncio.sleep(delay)

    logger.error("[Reranker] Max retries exhausted. Returning original order.")
    return [(i, t, 0.0) for i, t in enumerate(texts)][:top_k]