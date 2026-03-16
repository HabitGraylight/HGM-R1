"""
Centralized prompt definitions for HyperGraphMem agent.
"""

from typing import List, Optional, Dict, Any, Union
from hypergraphmem.prompt import (
    PROMPTS,
    JSON_SCHEMAS,
    EXTRACTION_PROMPT,
    NER_PROMPT,
    GENERATION_PROMPT,
)

from hypergraphmem.operation import format_memory_context

# =============================================================================
# JSON SCHEMAS for Qwen structured output，for agent policy model
# =============================================================================

POLICY_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "policy_operations_schema",
        "schema": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": [
                                    "ADD",
                                    "UPDATE",
                                    "DELETE"
                                ],
                                "description": "The action to perform."
                            },
                            # --- Fields primarily for ADD ---
                            "type": {
                                "type": ["string", "null"],
                                "enum": ["long_term", "short_term", None],
                                "description": "Required for ADD. Specifies the memory tier."
                            },
                            "content": {
                                "type": ["string", "null"],
                                "description": "Required for ADD. The content of the new memory."
                            },
                            "happened_at": {
                                "type": ["string", "null"],
                                "description": "Required for ADD. Timestamp (YYYY-MM-DD [HH:MM:SS]) based on Reference Time."
                            },
                            
                            # --- Fields primarily for UPDATE/DELETE ---
                            "memory_id": {
                                "type": ["string", "null"],
                                "description": "Required for UPDATE and DELETE. The unique ID of the target memory."
                            },
                            
                            # --- Fields specific to UPDATE (Optional) ---
                            "new_content": {
                                "type": ["string", "null"],
                                "description": "Optional for UPDATE. Use this for correction or refinement."
                            },
                            "new_happened_at": {
                                "type": ["string", "null"],
                                "description": "Optional for UPDATE. Use this to correct the timestamp."
                            },
                            "new_type": {
                                "type": ["string", "null"],
                                "enum": ["long_term", "short_term", None],
                                "description": "Optional for UPDATE. Use this for promoting (short->long) or demoting."
                            }
                        },
                        # 仅 action 是严格必须存在的，其他字段根据 action 动态出现
                        "required": ["action"], 
                        "additionalProperties": False
                    },
                    "description": "List of memory operations to perform"
                }
            },
            "required": ["operations"],
            "additionalProperties": False
        },
        "strict": True
    }
}


AGENT_JSON_SCHEMAS = {
    "policy": POLICY_JSON_SCHEMA,
}

# =============================================================================
# SYSTEM PROMPTS 
# =============================================================================

MEMORY_SYSTEM_PROMPT = """
You manage a hierarchical memory system.
1. Short-Term Memory: Temporary, session-specific, working memory.
2. Long-Term Memory: Persistent, core facts, frequently accessed (high access_count).

Lifecycle Rules:
- New info defaults to ST.
- Promote ST to LT if accessed frequently.
- Delete outdated or contradicted info.
- Ignore redundant info.
- **Time Sensitivity**: Always preserve the specific 'happened' time of events.
"""

POLICY_PROMPT_TEMPLATE = """{system_prompt}

## Current Context (Temporal Anchors)
- **Reference Time (Narrative Time)**: {reference_time}
  - *Usage*: The timestamp of the current conversation/event. Use this as the anchor for calculating `happened_at` for new memories (e.g., if text says "today", use this date).
- **System Time (Physical Time)**: {current_time}
  - *Usage*: The current real-world time. Compare this with a memory's `last_accessed` to judge its freshness.

## Task
Based on the **Extracted Facts** and **Retrieved Related Memories**, decide what operations to perform to keep the memory system accurate.

## Input Explanation

### 1. Extracted Facts (Current Input)
- **Format**: `[Happened: YYYY-MM-DD (HH:MM:SS)] Content`
- **Metadata**: The timestamp aligns with **Reference Time**. The time part (HH:MM:SS) is optional.

### 2. Retrieved Related Memories (Context)
- **Format**: XML blocks (`<long_term_memory>`, `<short_term_memory>`)
- **Metadata Attributes**:
  - `id`: Unique identifier. **Required** for UPDATE/DELETE.
  - `type`: Current tier (long_term/short_term).
  - `calls`: **Access Frequency**. High calls imply high importance.
  - `happened`: **Semantic Time**. When the event occurred historically.
  - `last_accessed`: **System Time**. When this memory was last retrieved.

## Available Actions & Output Rules
Output purely in JSON.

1. **ADD**
   - **Use when**: Storing new information.
   - **Required**: `content`, `happened_at`, `type` ("long_term" or "short_term").
   - **Rule**: 
     - `happened_at` must be derived from **Reference Time**.
     - Typically, use the fact's timestamp directly, unless logical deduction implies otherwise (e.g. relative dates).

2. **UPDATE**
   - **Use when**: Modifying **ANY** aspect of a memory (Correction, Tier Change, Status Update).
   - **Required**: `memory_id`.
   - **Optional**: `new_content`, `new_happened_at`, `new_type`.

3. **DELETE**
   - **Use when**: Removing invalid or contradicted memory.
   - **Required**: `memory_id`.

## Guidelines for "UPDATE"
- **CHANGE TIER (Promote/Demote)**: Set `new_type`.
  - **Promote** (Short->Long): Trigger if a Short-Term memory has high `calls` OR is re-confirmed by current facts. Refine content to be absolute.
  - **Demote** (Long->Short): Trigger if a Long-Term memory becomes temporarily specific or loses long-term significance.
- **CORRECT / REVISE**: Use `new_content` or `new_happened_at` to:
  - Fix factual errors.
  - **Update Status**: Reflect new developments in the user's life (e.g., "living in NY" -> "moved to SF").

## Examples

### Example 1: Add New Info (Reference Time Anchor)
**Current Context**:
- Reference Time: 2023-10-01
- System Time: 2025-01-01 09:00:00
**Facts**:
- [Happened: 2023-10-01] Amy bought a new Tesla.
**Memories**: (empty)
**Output**:
{{
  "operations": [
    {{
      "action": "ADD",
      "type": "long_term",
      "content": "Amy bought a new Tesla Model 3.",
      "happened_at": "2023-10-01"  // Matches Reference Time
    }}
  ]
}}

### Example 2: Update Status & Time
**Current Context**:
- Reference Time: 2024-01-15
- System Time: 2025-01-01 09:00:00
**Facts**:
- [Happened: 2024-01-15] Tom moved to SF.
**Memories**:
<long_term_memory>
- CONTENT: Tom lives in New York.
  METADATA: type=long_term | id="mem_0" | calls=5 | happened=2022-05-01
</long_term_memory>
**Output**:
{{
  "operations": [
    {{
      "action": "UPDATE",
      "memory_id": "mem_0",
      "new_content": "Tom lives in San Francisco.",
      "new_happened_at": "2024-01-15" // Reflects the new status change
    }}
  ]
}}

### Example 3: Promote (High Calls)
**Current Context**:
- Reference Time: 2023-09-20 10:00:00
- System Time: 2023-09-20 10:05:00
**Facts**:
- [Happened: 2023-09-20] Jack mentions hiking again.
**Memories**:
<short_term_memory>
- CONTENT: Jack enjoys hiking this weekend.
  METADATA: type=short_term | id="mem_1" | calls=12 | happened=2023-09-10 | last_accessed=2023-09-19
</short_term_memory>
**Reasoning**: High calls (12) + Re-confirmation -> Promote to Long Term.
**Output**:
{{
  "operations": [
    {{
      "action": "UPDATE",
      "memory_id": "mem_1",
      "new_type": "long_term",
      "new_content": "Jack enjoys hiking."
    }}
  ]
}}

### Example 4: Delete
**Current Context**:
- Reference Time: 2023-08-05
- System Time: 2025-01-01
**Facts**:
- [Happened: 2023-08-05] Cancelled gym.
**Memories**:
<short_term_memory>
- CONTENT: Gym membership active.
  METADATA: type=short_term | id="mem_2" | calls=1 | happened=2023-08-01
</short_term_memory>
**Output**:
{{
  "operations": [
    {{
      "action": "DELETE",
      "memory_id": "mem_2"
    }}
  ]
}}

Real Input:
**Current Context**:
- Reference Time: {reference_time}
- System Time: {current_time}

**Facts**:
{facts_text}

**Memories**:
{retrieved_context}

**Output**:
"""

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_policy_prompt(
    facts: List[Union[str, Dict]], 
    reference_time: str = "",
    system_time: str = "",
    retrieved_context: str = "",
) -> str:
    """
    Build prompt for policy model. 
    Handles structured facts with timestamps.
    """

    formatted_facts = []
    
    if not facts:
        facts_text = "No facts extracted."
    else:
        for f in facts:
            if isinstance(f, dict):
                # Format: "- [Happened: 2023-XX-XX] User went to..."
                content = f.get("content", "")
                ts = f.get("timestamp", "Unknown")
                formatted_facts.append(f"- [Happened: {ts}] {content}")
            else:
                # Fallback for legacy string lists
                formatted_facts.append(f"- {f}")
        facts_text = "\n".join(formatted_facts)

    retrieved_text = retrieved_context if retrieved_context else "No related memories found."
    
    return POLICY_PROMPT_TEMPLATE.format(
        system_prompt=MEMORY_SYSTEM_PROMPT,
        reference_time=reference_time,
        current_time=system_time,
        facts_text=facts_text,
        retrieved_context=retrieved_text,
    )

__all__ = [
    "PROMPTS", "JSON_SCHEMAS", "EXTRACTION_PROMPT", "NER_PROMPT", "GENERATION_PROMPT", "MEMORY_SYSTEM_PROMPT", "POLICY_PROMPT_TEMPLATE", "POLICY_JSON_SCHEMA", "AGENT_JSON_SCHEMAS",
    "build_policy_prompt",
]