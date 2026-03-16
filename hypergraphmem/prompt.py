
# EXTRACTION_PROMPT = """
# Decompose the "Content" into clear and simple knowledge units, ensuring they are interpretable out of context.
# 1. Split compound sentence into simple sentences. Maintain the original phrasing from the input whenever possible.
# 2. For any named entity that is accompanied by additional descriptive information, separate this information into its own distinct proposition.
# 3. Pronoun Elimination: Replace ALL pronouns (it, they, this, etc.) with full taxonomic names or explicit references. Never use possessives (its, their), always use "[entity]'s [property]" construction.
# 4. Present the results as a list of strings, formatted in JSON.

# Example-1:
# Input: Jes\u00fas Aranguren. His 13-year professional career was solely associated with Athletic Bilbao, with which he played in nearly 400 official games, winning two Copa del Rey trophies.
# Output: {{ "knowledge_units": [ "Jesús Aranguren had a 13-year professional career.", "Jesús Aranguren's professional career was solely associated with Athletic Bilbao.", "Athletic Bilbao is a football club.", "Jesús Aranguren played for Athletic Bilbao in nearly 400 official games.", "Jesús Aranguren won two Copa del Rey trophies with Athletic Bilbao."]}}

# Example-2:
# Input: Ophrys apifera. Ophrys apifera grows to a height of 15 -- 50 centimetres (6 -- 20 in). This hardy orchid develops small rosettes of leaves in autumn. They continue to grow slowly during winter. Basal leaves are ovate or oblong - lanceolate, upper leaves and bracts are ovate - lanceolate and sheathing. The plant blooms from mid-April to July producing a spike composed from one to twelve flowers. The flowers have large sepals, with a central green rib and their colour varies from white to pink, while petals are short, pubescent, yellow to greenish. The labellum is trilobed, with two pronounced humps on the hairy lateral lobes, the median lobe is hairy and similar to the abdomen of a bee. It is quite variable in the pattern of coloration, but usually brownish - red with yellow markings. The gynostegium is at right angles, with an elongated apex.
# Output: {{ "knowledge_units": [ "Ophrys apifera grows to a height of 15-50 centimetres (6-20 in)", "Ophrys apifera is a hardy orchid", "Ophrys apifera develops small rosettes of leaves in autumn", "The leaves of Ophrys apifera continue to grow slowly during winter", "The basal leaves of Ophrys apifera are ovate or oblong-lanceolate", "The upper leaves and bracts of Ophrys apifera are ovate-lanceolate and sheathing", "Ophrys apifera blooms from mid-April to July", "Ophrys apifera produces a spike composed of one to twelve flowers", "The flowers of Ophrys apifera have large sepals with a central green rib", "The flowers of Ophrys apifera vary in colour from white to pink", "The petals of Ophrys apifera are short, pubescent, and yellow to greenish", "The labellum of Ophrys apifera is trilobed with two pronounced humps on the hairy lateral lobes", "The median lobe of Ophrys apifera's labellum is hairy and resembles a bee's abdomen", "The coloration pattern of Ophrys apifera is variable but usually brownish-red with yellow markings", "The gynostegium of Ophrys apifera is at right angles with an elongated apex" ]}}

# JUST OUTPUT THE RESULTS IN JSON FORMAT! DON'T OUTPUT ANYTHING INRELEVENT!
# Input: {passage}
# Output:
# """

EXTRACTION_PROMPT = """
Reference Time: {reference_time}

You are a precise memory extraction system. Your task is to decompose the "Content" into atomic knowledge units and resolve their absolute timestamps based on the "Reference Time".

Output must be a JSON object adhering to the schema:
{{
  "knowledge_units": [
    {{ "content": "...", "timestamp": "..." }}
  ]
}}

### Rules & Instructions:

1. **Temporal Resolution (Precision Rule)**:
   - Use the "Reference Time" as the anchor.
   - **Do NOT hallucinate time**: If the input only implies a date (e.g., "yesterday"), the timestamp output MUST be just the date (YYYY-MM-DD). Do NOT append "00:00:00" unless the input explicitly specifies midnight.
   - **Default Fallback**: If the text contains no relative time expressions (e.g. "I like apples"), use the **Reference Time** verbatim as the timestamp.

2. **Content Rewriting (Context Injection)**:
   - **For Specific Events/Actions**: You MUST embed the resolved absolute date/time into the `content` string to make it standalone.
     * *Example*: "I flew yesterday" (Ref: 2023-10-05) -> Content: "User flew on 2023-10-04".
   - **For General Attributes/Preferences**: Do NOT embed the date into the content. Keep the fact timeless.
     * *Example*: "My eyes are brown" (Ref: 2023-10-05) -> Content: "User's eyes are brown". (Timestamp is still 2023-10-05).

3. **Atomic Facts**: 
   - Split compound sentences.
   - Replace pronouns (he, it, they) with specific entity names.

### Examples:

**Example 1 (Relative Date Event)**
Reference Time: 2023-10-05 Saturday
Input: I had a flight to Tokyo yesterday.
Output: {{
  "knowledge_units": [
    {{
      "content": "User had a flight to Tokyo on 2023-10-04.",
      "timestamp": "2023-10-04"
    }}
  ]
}}

**Example 2 (General Attribute - Timeless Content)**
Reference Time: 2025-12-13
Input: My favorite color is blue.
Output: {{
  "knowledge_units": [
    {{
      "content": "User's favorite color is blue.",
      "timestamp": "2025-12-13"
    }}
  ]
}}

**Example 3 (Specific Timestamp Event)**
Reference Time: 2023-05-25
Input: I am eating lunch right now (13:15).
Output: {{
  "knowledge_units": [
    {{
      "content": "User is eating lunch at 13:15 on 2023-05-25.",
      "timestamp": "2023-05-25 13:15"
    }}
  ]
}}

**Example 4 (No Time Info - Default to Reference)**
Reference Time: 2024-01-01 10:00:00
Input: I am feeling a bit tired.
Output: {{
  "knowledge_units": [
    {{
      "content": "User is feeling a bit tired.",
      "timestamp": "2024-01-01 10:00:00"
    }}
  ]
}}

JUST OUTPUT THE RESULTS IN JSON FORMAT!
Input: {passage}
Output:
"""

NER_PROMPT = """
Your task is to extract named entities from the given paragraph. 
Respond with a JSON list of entities.
Example:
Input: Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.
Output: {{"named_entities":["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]}}

Input: {passage}
Output:
"""

# DONE：GENERATION_PROMPT写得太简单了，人家mem0的prompt是有那些基于长期和短期记忆进行回答例子，可以参考那边的prompt把这个GENERATION_PROMPT完善一下。在operation里打包了相关的context
GENERATION_PROMPT = """You are an intelligent assistant with access to a structured memory system containing both long-term and short-term memories.

{context_data}

INSTRUCTIONS:
- Long-term memories are consolidated, frequently accessed, and verified as reliable information
- Short-term memories are recent observations that may need further verification
- Source documents provide the original context from which memories were extracted
- If memories conflict, prefer long-term memories or explain the discrepancy
- Reference specific memory IDs when citing information
- Use the retrieved memories to answer the question accurately
- If the memories contain relevant information, synthesize them into a coherent answer
- If the memories don't contain enough information, say so honestly
- Be concise and direct in your response

USER QUERY: {question}

Provide a helpful, accurate, and concise response based on the available memory context.
You MUST output your response in JSON format adhering to the schema provided.
Do not output your thinking process, only provide the final JSON object ({{"answer": "Your answer"}})
"""

PROMPTS = {
    "extraction": EXTRACTION_PROMPT,
    "ner": NER_PROMPT,
    "generation": GENERATION_PROMPT,
    "fail_response": "No relevant information found.",
}

"""
"extraction": {
        "type": "json_schema",
        "json_schema": {
            "name": "knowledge_units_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "knowledge_units": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["knowledge_units"],
                "additionalProperties": False,
            },
        },
    },
"""

JSON_SCHEMAS = {
    "extraction": {
        "type": "json_schema",
        "json_schema": {
            "name": "knowledge_units_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "knowledge_units": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "The atomic fact. For specific events, INCLUDE the absolute date/time in the text string. For general attributes, keep the text clean."
                                },
                                "timestamp": {
                                    "type": "string",
                                    "description": "The occurrence time (e.g., 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'). Inherit 'Reference Time' if no specific relative time is found."
                                }
                            },
                            "required": ["content", "timestamp"],
                            "additionalProperties": False
                        },
                    }
                },
                "required": ["knowledge_units"],
                "additionalProperties": False,
            },
        },
    },
    "ner": {
        "type": "json_schema",
        "json_schema": {
            "name": "named_entities_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "named_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["named_entities"],
                "additionalProperties": False,
            },
        },
    },
    "generation": {
        "type": "json_schema",
        "json_schema": {
            "name": "generation_response_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final concise answer to the user query."
                    },
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    },
}

