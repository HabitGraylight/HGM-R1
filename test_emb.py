import asyncio
import os
from pathlib import Path
import yaml
import numpy as np
import aiohttp
from typing import List, Union, Dict, Any

# =============================================================================
# 1. 临时定义 get_embeddings 函数 (为了独立运行，不需要依赖其他文件)
# =============================================================================
async def get_embeddings(texts: Union[str, List[str]], config: Dict[str, Any]) -> np.ndarray:
    embedding_cfg = config.get("embedding", {})
    base_url = embedding_cfg.get("base_url")
    model_name = embedding_cfg.get("model_name")
    
    if not base_url:
        raise ValueError("Embedding base_url is empty in config.")

    if isinstance(texts, str):
        texts = [texts]

    # 构造标准 OpenAI 格式请求
    payload = {
        "model": model_name,
        "input": texts,
        "encoding_format": "float"
    }

    print(f"[Request] POST {base_url}")
    print(f"[Payload] model={model_name}, input_len={len(texts)}")

    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, json=payload, timeout=30) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Embedding API Failed [{resp.status}]: {text}")
            
            data = await resp.json()
            # 兼容 OpenAI 格式
            if "data" in data:
                embeddings = [item["embedding"] for item in data["data"]]
            else:
                # 某些非标接口可能直接返回列表
                print("[Warning] Response format might be non-standard, trying raw parsing...")
                embeddings = data
                
            return np.array(embeddings)

# =============================================================================
# 2. 主测试逻辑
# =============================================================================
async def main():
    # --- 配置加载逻辑 ---
    config_path = Path(os.getenv("HYPERGRAPHMEM_CONFIG_PATH", Path(__file__).resolve().with_name("hypergraphmem_config.yaml")))
    config = {}

    # A. 尝试从文件加载
    if config_path.exists():
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            # 提取 hypergraphmem_config 下的 embedding 部分，或者是根目录下的 embedding
            if "hypergraphmem_config" in full_config:
                config = full_config["hypergraphmem_config"]
            elif "embedding" in full_config:
                config = full_config
            else:
                config = {"embedding": {}} 
                print("Warning: Could not find 'embedding' section in yaml.")
    else:
        print(f"Config file not found at {config_path}")

    # B. [重要] 如果没读到，或者你想手动强制测试，请在这里修改！
    # 如果你的 yaml 里没写 base_url，这里是最后的救命稻草
    if not config.get("embedding", {}).get("base_url"):
        print("\n!!! Config missing base_url. Using manual defaults/env vars !!!")
        config["embedding"] = {
            # 替换成你的真实 API 地址，例如 vllm 的地址
            "base_url": os.getenv("EMBEDDING_BASE_URL", "http://localhost:8000/v1/embeddings"),
            "model_name": os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct") 
        }

    print(f"Testing with config: {config.get('embedding')}")

    # --- 执行测试 ---
    try:
        test_text = "Testing embedding dimension."
        emb = await get_embeddings(test_text, config)
        
        print("\n" + "="*40)
        print(f"RESULT:")
        print(f"Input: '{test_text}'")
        print(f"Output Shape: {emb.shape}")
        
        dim = emb.shape[1]
        print(f"Detected Dimension: {dim}")
        
        if dim == 1536:
            print("✅ Dimension matches default (1536).")
        elif dim == 2560:
            print("⚠️ Dimension is 2560! (Likely Qwen2.5-14B/32B or similar?)")
            print("👉 You MUST update your NanoVectorDBStorage logic to handle this.")
        elif dim == 4096:
            print("⚠️ Dimension is 4096! (Likely Llama-3-70B or Qwen-72B?)")
        else:
            print(f"⚠️ Dimension {dim} is non-standard.")
            
        print("="*40 + "\n")

    except Exception as e:
        print(f"\n❌ Test Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())