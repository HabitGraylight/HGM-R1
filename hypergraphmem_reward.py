# hypergraphmem_reward.py
# Copyright 2025
#
# Verl 0.7.0 compatible reward manager for HyperGraph Memory.

from __future__ import annotations

import inspect
import logging
import os
import re
import string
from pathlib import Path
from collections import Counter
from typing import Any, Dict

import numpy as np
from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

logger = logging.getLogger(__name__)

DEFAULT_HGM_CONFIG_PATH = Path(__file__).resolve().with_name("hypergraphmem_config.yaml")


def normalize_answer(answer_text: Any) -> str:
    text = str(answer_text).lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def f1_score(prediction: Any, ground_truth: Any) -> tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    zero_metric = (0.0, 0.0, 0.0)

    special_answers = {"yes", "no", "noanswer"}
    if (
        normalized_prediction in special_answers or normalized_ground_truth in special_answers
    ) and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common_tokens = sum(common_tokens.values())

    if num_common_tokens == 0:
        return zero_metric

    precision = num_common_tokens / max(1, len(prediction_tokens))
    recall = num_common_tokens / max(1, len(ground_truth_tokens))
    f1 = (2 * precision * recall) / max(1e-8, precision + recall)
    return float(f1), float(precision), float(recall)


def _extract_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
    return value


@register("hypergraphmem")
class HyperGraphMemRewardManager(RewardManagerBase):
    """
    Verl official reward-loop manager style:
    - base class: RewardManagerBase
    - registry: verl.experimental.reward_loop.reward_manager.register
    - run_single return: {"reward_score": float, "reward_extra_info": dict}
    """

    def __init__(
        self,
        config: DictConfig,
        tokenizer: AutoTokenizer | None,
        compute_score=None,
        reward_router_address: str | None = None,
        reward_model_tokenizer: AutoTokenizer | None = None,
    ):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score) if self.compute_score else False
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        self.num_examine = int(config.get("num_examine", 0))
        self._printed = 0
        self.reward_mode = str(config.get("reward_mode", "f1"))

        # Optional external file path is kept for compatibility, not required by default.
        self.hgm_config_path = config.get(
            "hgm_config_path", os.getenv("HYPERGRAPHMEM_CONFIG_PATH", str(DEFAULT_HGM_CONFIG_PATH))
        )

    def _extract_prediction(self, data_item: DataProto) -> str:
        # Preferred minimal format for custom training scripts.
        if "predicted_answer" in data_item.non_tensor_batch:
            return str(_extract_scalar(data_item.non_tensor_batch["predicted_answer"]))

        # Generic fallback from extra_info.
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        if isinstance(extra_info, dict):
            if "generated_answer" in extra_info:
                return str(extra_info["generated_answer"])
            if "gen_ans" in extra_info:
                return str(extra_info["gen_ans"])

        # Final fallback: decode response tokens if tokenizer is available.
        if self.tokenizer is not None and "responses" in data_item.batch and "attention_mask" in data_item.batch:
            response_ids = data_item.batch["responses"]
            response_length = response_ids.shape[-1]
            valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
            valid_response_ids = response_ids[:valid_response_length]
            return str(self.tokenizer.decode(valid_response_ids, skip_special_tokens=True))

        return ""

    def _extract_ground_truth(self, data_item: DataProto) -> str:
        if "ground_truth" in data_item.non_tensor_batch:
            return str(_extract_scalar(data_item.non_tensor_batch["ground_truth"]))

        reward_model = data_item.non_tensor_batch.get("reward_model", {})
        if isinstance(reward_model, dict) and "ground_truth" in reward_model:
            return str(reward_model["ground_truth"])

        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        if isinstance(extra_info, dict) and "answer" in extra_info:
            return str(extra_info["answer"])

        return ""

    async def run_single(self, data: DataProto) -> Dict[str, Any]:
        assert len(data) == 1, "HyperGraphMemRewardManager only supports single-item DataProto in run_single."
        data_item = data[0]

        prediction = self._extract_prediction(data_item)
        ground_truth = self._extract_ground_truth(data_item)

        # Optional user custom compute_score callback (verl-style extension point).
        if self.compute_score is not None:
            kwargs = {
                "solution_str": prediction,
                "ground_truth": ground_truth,
                "extra_info": data_item.non_tensor_batch.get("extra_info", {}),
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.is_async_reward_score:
                result = await self.compute_score(**kwargs)
            else:
                result = await self.loop.run_in_executor(None, lambda: self.compute_score(**kwargs))

            if isinstance(result, dict):
                score = float(result.get("score", 0.0))
                reward_extra_info = result
            else:
                score = float(result)
                reward_extra_info = {"acc": score}
        else:
            # Default F1 reward (stable, no external LLM dependency).
            f1, precision, recall = f1_score(prediction, ground_truth)
            score = f1 if self.reward_mode == "f1" else precision
            reward_extra_info = {
                "acc": float(score),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "predicted_answer": prediction,
                "ground_truth": ground_truth,
            }

        if self._printed < self.num_examine:
            logger.info(
                "[HGM Reward] score=%.4f | pred=%s | gt=%s",
                score,
                prediction[:120],
                ground_truth[:120],
            )
            self._printed += 1

        return {"reward_score": float(score), "reward_extra_info": reward_extra_info}

