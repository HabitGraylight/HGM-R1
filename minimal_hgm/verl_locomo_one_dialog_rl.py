#!/usr/bin/env python3
"""
Verl-style RL training demo on one LoCoMo dialogue.

Key points:
- Uses official verl reward-manager interface (RewardManagerBase + register).
- Uses one dialogue from official LoCoMo-10 dataset.
- Trains a tiny policy model to choose memory actions:
  WRITE_SHORT / RETRIEVE / CONSOLIDATE
- Evaluates with HyperGraphMemRewardManager through DataProto.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.distributions import Categorical

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import get_reward_manager_cls

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import for registration side effect: @register("hypergraphmem")
import hypergraphmem_reward  # noqa: F401,E402

ACTIONS = ["WRITE_SHORT", "RETRIEVE", "CONSOLIDATE"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
ID_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        best_idx, best_free = 0, -1
        for line in out.splitlines():
            idx, free_mb = [int(x.strip()) for x in line.split(",")]
            if free_mb > best_free:
                best_idx, best_free = idx, free_mb
        if best_idx < torch.cuda.device_count():
            return torch.device(f"cuda:{best_idx}")
    except Exception:
        pass
    return torch.device("cuda:0")


def normalize_text(text: Any) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = " ".join(text.split())
    return text


def tokenize_set(text: str) -> set[str]:
    return set(normalize_text(text).split())


class CharTokenizer:
    def __init__(self, stoi: Dict[str, int]):
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}
        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]

    @classmethod
    def build(cls, texts: List[str]) -> "CharTokenizer":
        chars = sorted(set("".join(texts)))
        vocab = ["<pad>", "<unk>"] + chars
        return cls({ch: i for i, ch in enumerate(vocab)})

    def encode(self, text: str, max_len: int) -> tuple[List[int], List[int]]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text][:max_len]
        attn = [1] * len(ids)
        if len(ids) < max_len:
            pad_n = max_len - len(ids)
            ids += [self.pad_id] * pad_n
            attn += [0] * pad_n
        return ids, attn

    def dumps(self) -> Dict[str, int]:
        return self.stoi


class TinyPolicyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, len(ACTIONS))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        key_padding_mask = ~attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


@dataclass
class MemoryRecord:
    mem_id: str
    text: str
    memory_type: str
    call_count: int


class HypergraphMemory:
    def __init__(self, short_to_long_threshold: int = 2):
        self.graph = nx.Graph()
        self.records: Dict[str, MemoryRecord] = {}
        self.mem_seq = 0
        self.short_to_long_threshold = short_to_long_threshold

    def add_short_memory(self, text: str) -> str:
        mem_id = f"mem-{self.mem_seq}"
        self.mem_seq += 1
        self.records[mem_id] = MemoryRecord(mem_id, text, "short_term", 0)
        self.graph.add_node(mem_id, kind="memory")
        return mem_id

    def consolidate(self, text: str) -> str:
        mem_id = self.add_short_memory(text)
        self.records[mem_id].memory_type = "long_term"
        return mem_id

    def retrieve(self, query: str, top_k: int = 1) -> List[MemoryRecord]:
        q_tokens = tokenize_set(query)
        scored: List[tuple[float, str]] = []
        for mem_id, rec in self.records.items():
            rec_tokens = tokenize_set(rec.text)
            overlap = len(q_tokens & rec_tokens)
            score = float(overlap) + (0.5 if rec.memory_type == "long_term" else 0.0) + 0.1 * rec.call_count
            scored.append((score, mem_id))
        scored.sort(reverse=True)
        out: List[MemoryRecord] = []
        for _, mem_id in scored[:top_k]:
            rec = self.records[mem_id]
            rec.call_count += 1
            if rec.memory_type == "short_term" and rec.call_count >= self.short_to_long_threshold:
                rec.memory_type = "long_term"
            out.append(rec)
        return out

    def answer_question(self, question: str) -> str:
        hits = self.retrieve(question, top_k=1)
        return hits[0].text if hits else ""

    def stats(self) -> Dict[str, int]:
        short_n = sum(1 for r in self.records.values() if r.memory_type == "short_term")
        long_n = sum(1 for r in self.records.values() if r.memory_type == "long_term")
        return {
            "memory_nodes": len(self.records),
            "short_term": short_n,
            "long_term": long_n,
        }


def flatten_locomo_turns(dialogue_obj: dict, max_turns: int) -> List[dict]:
    conv = dialogue_obj["conversation"]
    session_ids = []
    for k, v in conv.items():
        if k.startswith("session_") and isinstance(v, list):
            try:
                session_ids.append(int(k.split("_")[1]))
            except Exception:
                continue
    session_ids.sort()

    turns = []
    for sid in session_ids:
        date = conv.get(f"session_{sid}_date_time", "")
        for turn in conv[f"session_{sid}"]:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            fact = f"{speaker} ({date}): {text}"
            turns.append({"speaker": speaker, "text": text, "date": date, "fact": fact})
            if len(turns) >= max_turns:
                return turns
    return turns


def select_retrievable_qas(dialogue_obj: dict, turns: List[dict], qa_limit: int) -> List[dict]:
    corpus = [normalize_text(t["fact"]) for t in turns]
    picked = []
    for qa in dialogue_obj.get("qa", []):
        answer = normalize_text(qa.get("answer", ""))
        if answer and any(answer in c for c in corpus):
            picked.append(qa)
        if len(picked) >= qa_limit:
            break
    if not picked:
        return dialogue_obj.get("qa", [])[:qa_limit]
    return picked


def run_async_with_manager_loop(reward_manager, coro):
    try:
        return reward_manager.loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def score_with_reward_manager(reward_manager, predicted_answer: str, ground_truth: str) -> float:
    data = DataProto.from_single_dict(
        {
            "predicted_answer": np.array([predicted_answer], dtype=object),
            "ground_truth": np.array([ground_truth], dtype=object),
        }
    )
    out = run_async_with_manager_loop(reward_manager, reward_manager.run_single(data))
    return float(out["reward_score"])


def build_reward_manager() -> Any:
    reward_cls = get_reward_manager_cls("hypergraphmem")
    cfg = OmegaConf.create({"reward_mode": "f1", "num_examine": 1})
    return reward_cls(cfg, tokenizer=None)


def train_one_dialogue(
    model: TinyPolicyTransformer,
    tokenizer: CharTokenizer,
    turns: List[dict],
    qas: List[dict],
    reward_manager,
    device: torch.device,
    epochs: int,
    lr: float,
    gamma: float,
    max_len: int,
) -> List[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    baseline = 0.0
    logs: List[dict] = []

    for epoch in range(1, epochs + 1):
        memory = HypergraphMemory(short_to_long_threshold=2)
        log_probs = []
        rewards = []
        action_counter = {k: 0 for k in ACTIONS}

        model.train()
        for t in turns:
            ids, attn = tokenizer.encode(t["fact"], max_len=max_len)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            attention_mask = torch.tensor([attn], dtype=torch.long, device=device)
            logits = model(input_ids, attention_mask)
            dist = Categorical(logits=logits)
            action_id = int(dist.sample().item())
            action = ID_TO_ACTION[action_id]
            action_counter[action] += 1
            log_probs.append(dist.log_prob(torch.tensor(action_id, device=device)))

            step_reward = 0.0
            if action == "WRITE_SHORT":
                memory.add_short_memory(t["fact"])
                step_reward += 0.01
            elif action == "CONSOLIDATE":
                memory.consolidate(t["fact"])
                step_reward += 0.012
            else:
                _ = memory.retrieve(t["text"], top_k=1)
                step_reward += -0.005
            rewards.append(step_reward)

        qa_scores = []
        model.eval()
        for qa in qas:
            q = qa.get("question", "")
            gt = str(qa.get("answer", ""))
            pred = memory.answer_question(q)
            qa_scores.append(score_with_reward_manager(reward_manager, pred, gt))

        qa_mean = float(np.mean(qa_scores)) if qa_scores else 0.0
        rewards[-1] += qa_mean

        returns = []
        g = 0.0
        for r in reversed(rewards):
            g = r + gamma * g
            returns.append(g)
        returns.reverse()
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        baseline = 0.9 * baseline + 0.1 * float(returns_t.mean().item())
        advantages = returns_t - baseline
        log_probs_t = torch.stack(log_probs)
        loss = -(log_probs_t * advantages.detach()).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        stat = memory.stats()
        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "qa_f1_mean": qa_mean,
            "episode_return": float(returns_t.mean().item()),
            "actions": action_counter,
            "memory_stats": stat,
        }
        logs.append(row)
        print(
            f"[Epoch {epoch}] loss={row['loss']:.4f} qa_f1={qa_mean:.4f} "
            f"return={row['episode_return']:.4f} actions={action_counter}"
        )

    return logs


def main() -> None:
    parser = argparse.ArgumentParser("Verl-style RL training on one LoCoMo dialogue")
    parser.add_argument("--locomo-path", type=str, default=str(ROOT / "data" / "locomo10.json"))
    parser.add_argument("--dialogue-index", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=180)
    parser.add_argument("--qa-limit", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "output" / "verl_locomo_one_dialog"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    with open(args.locomo_path, "r", encoding="utf-8") as f:
        all_dialogues = json.load(f)
    if not (0 <= args.dialogue_index < len(all_dialogues)):
        raise ValueError(f"dialogue-index={args.dialogue_index} out of range (0..{len(all_dialogues)-1})")

    dialogue = all_dialogues[args.dialogue_index]
    turns = flatten_locomo_turns(dialogue, max_turns=args.max_turns)
    qas = select_retrievable_qas(dialogue, turns, qa_limit=args.qa_limit)

    print(
        f"Loaded LoCoMo dialogue #{args.dialogue_index}: "
        f"turns={len(turns)}, qa_candidates={len(dialogue.get('qa', []))}, qa_used={len(qas)}"
    )

    tokenizer_texts = [x["fact"] for x in turns] + [q["question"] for q in qas]
    tokenizer = CharTokenizer.build(tokenizer_texts + ACTIONS)

    model = TinyPolicyTransformer(
        vocab_size=len(tokenizer.stoi),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.max_len,
    ).to(device)

    reward_manager = build_reward_manager()
    logs = train_one_dialogue(
        model=model,
        tokenizer=tokenizer,
        turns=turns,
        qas=qas,
        reward_manager=reward_manager,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        max_len=args.max_len,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.dumps(),
            "args": vars(args),
        },
        out_dir / "policy_locomo_one_dialog.pt",
    )
    with open(out_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    with open(out_dir / "qa_subset.json", "w", encoding="utf-8") as f:
        json.dump(qas, f, ensure_ascii=False, indent=2)

    print(f"Saved checkpoint: {out_dir / 'policy_locomo_one_dialog.pt'}")
    print(f"Saved logs: {out_dir / 'train_log.json'}")


if __name__ == "__main__":
    main()

