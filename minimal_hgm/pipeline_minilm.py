#!/usr/bin/env python3
"""
Stable minimal pipeline:
- Tiny Transformer policy model (GPU)
- Hypergraph memory (short/long term)
- End-to-end demo run
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
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
ID_TO_ACTION = {i: a for a, i in ACTION_TO_ID.items()}

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


@dataclass
class LabeledSample:
    text: str
    action: str


def generate_samples(num_samples: int, seed: int) -> List[LabeledSample]:
    rng = random.Random(seed)
    data: List[LabeledSample] = []
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
        data.append(LabeledSample(text=text, action=action))
    return data


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

    def encode(self, text: str, max_len: int) -> Tuple[List[int], List[int]]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text][:max_len]
        attn = [1] * len(ids)
        if len(ids) < max_len:
            pad_n = max_len - len(ids)
            ids += [self.pad_id] * pad_n
            attn += [0] * pad_n
        return ids, attn

    def dumps(self) -> Dict[str, int]:
        return self.stoi


class ActionDataset(Dataset):
    def __init__(self, samples: List[LabeledSample], tokenizer: CharTokenizer, max_len: int):
        self.samples = samples
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        ids, attn = self.tok.encode(item.text, self.max_len)
        label = ACTION_TO_ID[item.action]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }


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

        # Masked mean pooling.
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


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
        lower = text.lower()
        return [n for n in NAMES if n.lower() in lower]

    def _find_by_text(self, fact: str) -> str | None:
        for mem_id, rec in self.records.items():
            if rec.text == fact:
                return mem_id
        return None

    def add_short_memory(self, fact: str, step: int) -> str:
        existing = self._find_by_text(fact)
        if existing is not None:
            return existing
        mem_id = f"mem-{self.mem_seq}"
        self.mem_seq += 1
        rec = MemoryRecord(mem_id, fact, "short_term", 0, step, step)
        self.records[mem_id] = rec
        self.graph.add_node(mem_id, kind="memory")
        for ent in self._extract_entities(fact):
            if not self.graph.has_node(ent):
                self.graph.add_node(ent, kind="entity")
            self.graph.add_edge(mem_id, ent)
        return mem_id

    def consolidate(self, fact: str, step: int) -> List[str]:
        mem_id = self.add_short_memory(fact, step)
        promoted = []
        self.records[mem_id].memory_type = "long_term"
        promoted.append(mem_id)
        for ent in self._extract_entities(fact):
            for nbr in self.graph.neighbors(ent):
                if nbr in self.records:
                    self.records[nbr].memory_type = "long_term"
                    if nbr not in promoted:
                        promoted.append(nbr)
        return promoted

    def retrieve(self, query: str, step: int, top_k: int = 1) -> List[MemoryRecord]:
        entities = self._extract_entities(query)
        cand = set()
        for ent in entities:
            if self.graph.has_node(ent):
                cand.update([x for x in self.graph.neighbors(ent) if x in self.records])
        if not cand:
            cand = set(self.records.keys())

        scored: List[Tuple[float, str]] = []
        for mem_id in cand:
            rec = self.records[mem_id]
            linked_ents = [x for x in self.graph.neighbors(mem_id) if x in NAMES]
            overlap = len(set(entities) & set(linked_ents))
            score = 2.0 * overlap + 0.5 * (rec.memory_type == "long_term") + 0.2 * rec.call_count
            scored.append((score, mem_id))
        scored.sort(reverse=True)

        result = []
        for _, mem_id in scored[:top_k]:
            rec = self.records[mem_id]
            rec.call_count += 1
            rec.last_access_step = step
            if rec.memory_type == "short_term" and rec.call_count >= self.short_to_long_threshold:
                rec.memory_type = "long_term"
            result.append(rec)
        return result

    def stats(self) -> Dict[str, int]:
        short_n = sum(1 for r in self.records.values() if r.memory_type == "short_term")
        long_n = sum(1 for r in self.records.values() if r.memory_type == "long_term")
        return {
            "memory_nodes": len(self.records),
            "entity_nodes": len([n for n, d in self.graph.nodes(data=True) if d.get("kind") == "entity"]),
            "edges": self.graph.number_of_edges(),
            "short_term": short_n,
            "long_term": long_n,
        }


def extract_fact(text: str) -> str:
    return text.split(":", 1)[1].strip() if ":" in text else text.strip()


@torch.no_grad()
def predict_action(model: TinyPolicyTransformer, tokenizer: CharTokenizer, text: str, max_len: int, device: torch.device) -> str:
    ids, attn = tokenizer.encode(text, max_len=max_len)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attn], dtype=torch.long, device=device)
    logits = model(input_ids, attention_mask)
    pred_id = int(torch.argmax(logits, dim=-1).item())
    return ID_TO_ACTION[pred_id]


def evaluate(model: TinyPolicyTransformer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask)
        pred = torch.argmax(logits, dim=-1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    return correct / max(1, total)


def train_policy(
    model: TinyPolicyTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())

        train_loss = total_loss / max(1, len(train_loader))
        val_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {ep}] train_loss={train_loss:.4f} val_action_acc={val_acc:.4f}")


def run_demo(model: TinyPolicyTransformer, tokenizer: CharTokenizer, max_len: int, device: torch.device, output_dir: Path) -> None:
    episode = [
        "Remember this: Alice likes sushi.",
        "Remember this: Bob lives in Shanghai.",
        "Important memory: Alice likes sushi.",
        "Question: What do you know about Alice?",
        "Question: What do you know about Bob?",
    ]
    memory = HypergraphMemory(short_to_long_threshold=2)
    trace = []

    for step, text in enumerate(episode, start=1):
        action = predict_action(model, tokenizer, text, max_len=max_len, device=device)
        fact = extract_fact(text)
        if action == "WRITE_SHORT":
            mem_id = memory.add_short_memory(fact, step)
            result = f"stored {mem_id}: {fact}"
        elif action == "CONSOLIDATE":
            promoted = memory.consolidate(fact, step)
            result = f"promoted to long-term: {promoted}"
        else:
            recalled = memory.retrieve(text, step, top_k=1)
            result = f"answer: {recalled[0].text}" if recalled else "answer: memory not found"

        stats = memory.stats()
        step_obj = {"step": step, "input": text, "action": action, "result": result, "stats": stats}
        trace.append(step_obj)
        print(
            f"[Step {step}] action={action:12s} | {result} | "
            f"short={stats['short_term']} long={stats['long_term']}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "demo_trace.json", "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser("Minimal HyperGraph Memory pipeline (stable classifier version)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--train-samples", type=int, default=1600)
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="output/minimal_hgm")
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    train_samples = generate_samples(args.train_samples, args.seed)
    val_samples = generate_samples(args.val_samples, args.seed + 1)

    tokenizer = CharTokenizer.build([x.text for x in train_samples + val_samples] + ACTIONS)
    train_ds = ActionDataset(train_samples, tokenizer, args.max_len)
    val_ds = ActionDataset(val_samples, tokenizer, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TinyPolicyTransformer(
        vocab_size=len(tokenizer.stoi),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.max_len,
    ).to(device)

    train_policy(model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "tokenizer": tokenizer.dumps(),
        "config": vars(args),
    }
    torch.save(ckpt, out_dir / "tiny_policy_classifier.pt")
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved checkpoint to {out_dir / 'tiny_policy_classifier.pt'}")

    run_demo(model, tokenizer, args.max_len, device, out_dir)
    print("Done. Trace saved to output/minimal_hgm/demo_trace.json")


if __name__ == "__main__":
    main()

