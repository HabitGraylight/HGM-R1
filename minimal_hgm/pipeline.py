#!/usr/bin/env python3
"""
Minimal end-to-end HyperGraph Memory demo.

What this script does:
1) Train a tiny causal language model to output memory actions:
   WRITE_SHORT / RETRIEVE / CONSOLIDATE
2) Execute the predicted actions on a hypergraph memory manager
   with short-term -> long-term transition.
3) Run a full demo episode and save trace + checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ACTIONS = ["WRITE_SHORT", "RETRIEVE", "CONSOLIDATE"]
NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve"]
FACTS = [
    "likes sushi",
    "lives in Shanghai",
    "works at NASA",
    "plays basketball",
    "studies graph learning",
]

WRITE_TEMPLATES = [
    "Remember this: {name} {fact}.",
    "New memory: {name} {fact}.",
    "Store this fact: {name} {fact}.",
]
RETRIEVE_TEMPLATES = [
    "Question: What do you know about {name}?",
    "Recall memory for {name}.",
    "Please answer: what is true about {name}?",
]
CONSOLIDATE_TEMPLATES = [
    "Important memory: {name} {fact}.",
    "Promote to long-term memory: {name} {fact}.",
    "High-priority fact: {name} {fact}.",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        best_idx = 0
        best_free = -1
        for line in out.splitlines():
            idx_str, free_str = [x.strip() for x in line.split(",")]
            idx, free_mb = int(idx_str), int(free_str)
            if free_mb > best_free:
                best_free = free_mb
                best_idx = idx
        if best_idx < torch.cuda.device_count():
            return torch.device(f"cuda:{best_idx}")
    except Exception:
        pass

    return torch.device("cuda:0")


@dataclass
class LabeledSample:
    text: str
    action: str


def generate_samples(num_samples: int, seed: int) -> List[LabeledSample]:
    rng = random.Random(seed)
    samples: List[LabeledSample] = []
    for _ in range(num_samples):
        name = rng.choice(NAMES)
        fact = rng.choice(FACTS)
        p = rng.random()
        if p < 0.45:
            text = rng.choice(WRITE_TEMPLATES).format(name=name, fact=fact)
            action = "WRITE_SHORT"
        elif p < 0.75:
            text = rng.choice(RETRIEVE_TEMPLATES).format(name=name)
            action = "RETRIEVE"
        else:
            text = rng.choice(CONSOLIDATE_TEMPLATES).format(name=name, fact=fact)
            action = "CONSOLIDATE"
        samples.append(LabeledSample(text=text, action=action))
    return samples


class CharTokenizer:
    def __init__(self, stoi: Dict[str, int]):
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    @classmethod
    def build(cls, texts: List[str]) -> "CharTokenizer":
        chars = sorted(set("".join(texts)))
        vocab = ["<pad>", "<bos>", "<eos>", "<unk>"] + chars
        return cls({ch: i for i, ch in enumerate(vocab)})

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        out = []
        for i in ids:
            ch = self.itos.get(int(i), "")
            if skip_special and ch in {"<pad>", "<bos>", "<eos>", "<unk>"}:
                continue
            out.append(ch)
        return "".join(out)

    def dumps(self) -> Dict[str, int]:
        return self.stoi

    @classmethod
    def loads(cls, payload: Dict[str, int]) -> "CharTokenizer":
        return cls(payload)


def build_prompt(text: str) -> str:
    return f"Input: {text}\nAction:"


class ActionCausalDataset(Dataset):
    def __init__(self, samples: List[LabeledSample], tokenizer: CharTokenizer, max_len: int):
        self.samples = samples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        item = self.samples[idx]
        prompt = build_prompt(item.text)
        prompt_ids = [self.tok.bos_id] + self.tok.encode(prompt)
        target_ids = self.tok.encode(" " + item.action) + [self.tok.eos_id]
        input_ids = (prompt_ids + target_ids)[: self.max_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_len]
        return {"input_ids": input_ids, "labels": labels}


def collate_batch(batch: List[Dict[str, List[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, x in enumerate(batch):
        n = len(x["input_ids"])
        input_ids[i, :n] = torch.tensor(x["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(x["labels"], dtype=torch.long)
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class TinyCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool), diagonal=1)
        key_padding_mask = ~attention_mask.bool()
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return self.lm_head(self.norm(x))


@torch.no_grad()
def predict_action(
    model: TinyCausalLM,
    tokenizer: CharTokenizer,
    device: torch.device,
    text: str,
) -> Tuple[str, str]:
    """
    Constrained action decoding:
    score each candidate action by LM log-probability and pick best.
    """
    prompt = build_prompt(text)
    prompt_ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
    scores: Dict[str, float] = {}

    for action in ACTIONS:
        target_ids = tokenizer.encode(" " + action) + [tokenizer.eos_id]
        seq = (prompt_ids + target_ids)[: model.max_len]
        if len(seq) < 2:
            scores[action] = float("-inf")
            continue

        input_ids = torch.tensor([seq[:-1]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        logits = model(input_ids, attention_mask)
        log_probs = torch.log_softmax(logits, dim=-1)

        next_tokens = torch.tensor([seq[1:]], dtype=torch.long, device=device)
        tok_logp = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)[0]

        start = max(0, len(prompt_ids) - 1)
        target_logp = tok_logp[start:]
        if target_logp.numel() == 0:
            scores[action] = float("-inf")
        else:
            # Length-normalized score to avoid bias toward shorter action strings.
            scores[action] = float(target_logp.mean().item())

    best_action = max(scores.items(), key=lambda kv: kv[1])[0]
    score_text = " | ".join(f"{k}:{v:.2f}" for k, v in scores.items())
    return best_action, score_text


@dataclass
class MemoryRecord:
    mem_id: str
    text: str
    memory_type: str
    call_count: int
    created_step: int
    last_access_step: int


class HypergraphMemory:
    def __init__(self, short_to_long_threshold: int = 2):
        self.graph = nx.Graph()
        self.short_to_long_threshold = short_to_long_threshold
        self.records: Dict[str, MemoryRecord] = {}
        self.mem_seq = 0

    def _extract_entities(self, text: str) -> List[str]:
        text_lower = text.lower()
        return [name for name in NAMES if name.lower() in text_lower]

    def _find_by_text(self, fact: str) -> str | None:
        for mem_id, rec in self.records.items():
            if rec.text == fact:
                return mem_id
        return None

    def _promote_if_needed(self, mem_id: str) -> None:
        rec = self.records[mem_id]
        if rec.memory_type == "short_term" and rec.call_count >= self.short_to_long_threshold:
            rec.memory_type = "long_term"

    def add_short_memory(self, fact: str, step: int) -> str:
        existing = self._find_by_text(fact)
        if existing is not None:
            return existing

        mem_id = f"mem-{self.mem_seq}"
        self.mem_seq += 1
        rec = MemoryRecord(
            mem_id=mem_id,
            text=fact,
            memory_type="short_term",
            call_count=0,
            created_step=step,
            last_access_step=step,
        )
        self.records[mem_id] = rec
        self.graph.add_node(mem_id, kind="memory")
        for ent in self._extract_entities(fact):
            if not self.graph.has_node(ent):
                self.graph.add_node(ent, kind="entity")
            self.graph.add_edge(mem_id, ent)
        return mem_id

    def consolidate(self, fact: str, step: int) -> List[str]:
        mem_id = self.add_short_memory(fact, step=step)
        promoted: List[str] = []
        rec = self.records[mem_id]
        rec.memory_type = "long_term"
        promoted.append(mem_id)

        for ent in self._extract_entities(fact):
            for nbr in self.graph.neighbors(ent):
                if nbr in self.records:
                    self.records[nbr].memory_type = "long_term"
                    if nbr not in promoted:
                        promoted.append(nbr)
        return promoted

    def retrieve(self, query: str, step: int, top_k: int = 2) -> List[MemoryRecord]:
        entities = self._extract_entities(query)
        candidates: List[str] = []
        if entities:
            cands = set()
            for ent in entities:
                if self.graph.has_node(ent):
                    cands.update([n for n in self.graph.neighbors(ent) if n in self.records])
            candidates = list(cands)
        else:
            candidates = list(self.records.keys())

        scored: List[Tuple[float, str]] = []
        for mem_id in candidates:
            rec = self.records[mem_id]
            rec_entities = [n for n in self.graph.neighbors(mem_id) if n in NAMES]
            overlap = len(set(entities) & set(rec_entities))
            score = 2.0 * overlap + 0.3 * rec.call_count + (0.8 if rec.memory_type == "long_term" else 0.0)
            scored.append((score, mem_id))
        scored.sort(reverse=True)

        selected: List[MemoryRecord] = []
        for _, mem_id in scored[:top_k]:
            rec = self.records[mem_id]
            rec.call_count += 1
            rec.last_access_step = step
            self._promote_if_needed(mem_id)
            selected.append(rec)
        return selected

    def stats(self) -> Dict[str, int]:
        short_term = sum(1 for x in self.records.values() if x.memory_type == "short_term")
        long_term = sum(1 for x in self.records.values() if x.memory_type == "long_term")
        return {
            "memory_nodes": len(self.records),
            "entity_nodes": len([n for n, d in self.graph.nodes(data=True) if d.get("kind") == "entity"]),
            "edges": self.graph.number_of_edges(),
            "short_term": short_term,
            "long_term": long_term,
        }


def extract_fact(text: str) -> str:
    if ":" in text:
        fact = text.split(":", 1)[1].strip()
    else:
        fact = text.strip()
    return fact.rstrip()


def evaluate(model: TinyCausalLM, tokenizer: CharTokenizer, device: torch.device, samples: List[LabeledSample]) -> float:
    correct = 0
    for x in samples:
        pred, _ = predict_action(model, tokenizer, device, x.text)
        correct += int(pred == x.action)
    return correct / max(1, len(samples))


def train_policy(
    model: TinyCausalLM,
    tokenizer: CharTokenizer,
    train_samples: List[LabeledSample],
    val_samples: List[LabeledSample],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    max_len: int,
) -> None:
    ds = ActionCausalDataset(train_samples, tokenizer, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, tokenizer.pad_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        train_loss = total_loss / max(1, len(loader))
        model.eval()
        val_acc = evaluate(model, tokenizer, device, val_samples[:200])
        model.train()
        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_action_acc={val_acc:.4f}")


def run_demo_episode(
    model: TinyCausalLM,
    tokenizer: CharTokenizer,
    device: torch.device,
    output_dir: Path,
) -> List[Dict]:
    episode = [
        "Remember this: Alice likes sushi.",
        "Remember this: Bob lives in Shanghai.",
        "Important memory: Alice likes sushi.",
        "Question: What do you know about Alice?",
        "Question: What do you know about Bob?",
    ]
    memory = HypergraphMemory(short_to_long_threshold=2)
    trace: List[Dict] = []

    for step, user_text in enumerate(episode, start=1):
        action, raw_action = predict_action(model, tokenizer, device, user_text)
        fact = extract_fact(user_text)
        result_text = ""

        if action == "WRITE_SHORT":
            mem_id = memory.add_short_memory(fact, step=step)
            result_text = f"stored {mem_id}: {fact}"
        elif action == "CONSOLIDATE":
            promoted = memory.consolidate(fact, step=step)
            result_text = f"promoted to long-term: {promoted}"
        else:
            recalled = memory.retrieve(user_text, step=step, top_k=1)
            if recalled:
                result_text = f"answer: {recalled[0].text}"
            else:
                result_text = "answer: memory not found"

        stats = memory.stats()
        step_obj = {
            "step": step,
            "input": user_text,
            "pred_action": action,
            "raw_generation": raw_action,
            "result": result_text,
            "stats": stats,
        }
        trace.append(step_obj)
        print(
            f"[Step {step}] action={action:12s} | {result_text} | "
            f"short={stats['short_term']} long={stats['long_term']}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "demo_trace.json", "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal HyperGraph Memory training + demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0...")
    parser.add_argument("--train-samples", type=int, default=1800)
    parser.add_argument("--val-samples", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=192)
    parser.add_argument("--output-dir", type=str, default="output/minimal_hgm")
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    train_samples = generate_samples(args.train_samples, seed=args.seed)
    val_samples = generate_samples(args.val_samples, seed=args.seed + 1)

    tokenizer_texts = []
    for s in train_samples + val_samples:
        tokenizer_texts.append(build_prompt(s.text))
        tokenizer_texts.append(" " + s.action)
    tokenizer = CharTokenizer.build(tokenizer_texts + ACTIONS)

    model = TinyCausalLM(
        vocab_size=len(tokenizer.stoi),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.max_len,
    ).to(device)

    train_policy(
        model=model,
        tokenizer=tokenizer,
        train_samples=train_samples,
        val_samples=val_samples,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.dumps(),
        "config": vars(args),
    }
    torch.save(ckpt, out_dir / "tiny_policy.pt")
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved checkpoint to {out_dir / 'tiny_policy.pt'}")

    run_demo_episode(model, tokenizer, device, out_dir)
    print("Done. Trace saved to output/minimal_hgm/demo_trace.json")


if __name__ == "__main__":
    main()
