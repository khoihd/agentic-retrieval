import json
import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel

from MemoryRelationship import MemoryRelationship, MemoryRelationshipList
from utils import (
    DEFAULT_MEMORY_SEARCH_TOP_K,
    MAX_MEMORY_RETRIEVAL,
    MEMORY_RELATIONSHIP_TOP_K,
    MODEL,
    cosine_similarity,
    embed_text,
    get_client,
)

MEMORY_FILE = "memories.json"


class Memory(BaseModel):
    id: str
    text: str
    memory_type: str
    scope: str
    embedding: list[float]
    created_at: str
    status: Literal["active", "superseded", "conflicted"]

class MemoryReference(BaseModel):
    id: str
    text: str
    memory_type: str
    scope: str
    status: Literal["active", "superseded", "conflicted"]

    @classmethod
    def from_memory(cls, memory: Memory) -> "MemoryReference":
        return cls.model_validate(memory.model_dump())

class ExtractedMemory(BaseModel):
    text: str
    memory_type: str
    scope: str


class ExtractedMemoryList(BaseModel):
    memories: list[ExtractedMemory]


def extract_memories(text: str) -> list[ExtractedMemory]:
    prompt = f"""

You are a memory extraction system for an AI assistant.
Your task is to extract durable, independently meaningful facts from
the user's input that may be useful in future conversations.

EXTRACTION RULES:

1. DURABILITY
- Extract facts, preferences, decisions, requirements, and constraints
  that are likely to remain useful in future conversations.
- Ignore temporary activities, one-time questions, greetings, and
  conversational filler.
- Do not extract information merely because it is mentioned.

2. ATOMICITY
- Each memory must contain exactly ONE independent fact.
- If a sentence contains multiple independent facts, split them into
  separate memories, even if they concern the same subject.
- Each memory must be understandable without the original input.
- Never combine independent requirements, technologies, preferences,
  or decisions into one memory.

Example:
Input:
"Project X now requires Python 3.14 and uses Redis for caching."

Correct:
- "Project X requires Python 3.14."
- "Project X uses Redis for caching."

Incorrect:
- "Project X requires Python 3.14 and uses Redis for caching."

3. TEMPORAL REASONING
- When the user describes a change, extract the latest confirmed state.
- Do not extract outdated information as a current fact.
- Preserve important qualifiers such as "currently", "planned",
  or "preferred" when they affect the meaning.
- Do not treat plans as completed actions.

Example:
Input:
"Project X used Python 3.12 last year but now requires Python 3.13."

Correct:
- "Project X requires Python 3.13."

Incorrect:
- "Project X requires Python 3.12."

4. ACCURACY
- Preserve the meaning of the user's statements.
- Do not invent facts, preferences, or requirements.
- Do not infer information that the user has not explicitly stated.
- Do not resolve contradictions by guessing.
- If no durable memories are present, return an empty list.

5. MEMORY TYPE

Assign a concise memory_type describing the fact.
Examples:
- project_requirement
- project_technology
- user_preference
- user_profile
- project_decision

6. SCOPE

Assign the appropriate scope:
- "user": Facts or preferences about the user.
- "project_x": Facts specifically about Project X.
- For other explicitly named projects, use a consistent
  project-specific scope.

7. OUTPUT
- Return memories matching the required structured output schema.
- Each extracted memory must have:
    - text
    - memory_type
    - scope

- Do not include explanations or commentary.
- Do not include IDs, embeddings, timestamps, or lifecycle statuses.

INPUT:
{text}
"""
    response = get_client().chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_format=ExtractedMemoryList,
        temperature=0,
        extra_body={
            "think": False
        },
    )

    extracted_memory_list = response.choices[0].message.parsed

    return extracted_memory_list.memories


def rerank_memories(original_query: str, memories: list[Memory]) -> list[Memory]:
    query_embedding = embed_text(original_query)
    reranked_memories = []

    for memory in memories:
        memory_embedding = memory.embedding
        rerank_score = cosine_similarity(
            query_embedding,
            memory_embedding,
        )

        reranked_memory = {"memory": memory, "rerank_score": rerank_score}
        reranked_memories.append(reranked_memory)

    reranked_memories.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return [entry["memory"] for entry in reranked_memories]


def search_memory(query: str, scope: str = "", top_k: int = DEFAULT_MEMORY_SEARCH_TOP_K) -> list[Memory]:
    """
    Search persistent memory created from previous interactions.
    Use this to retrieve relevant user preferences, previous decisions,
    project-specific facts learned in earlier conversations, and other
    remembered information. This function searches memory, not shared
    company documents, policies, or other external knowledge sources.

    Args:
        query: A semantic search query describing the remembered
            information needed.

    Returns:
        A formatted string containing relevant retrieved memories, or a
        message indicating that no relevant memories were found.
    """
    top_k = min(top_k, MAX_MEMORY_RETRIEVAL)
    memories = load_memories()
    related_memories = find_related_memories(query, memories, scope, top_k)

    return related_memories


def format_memory_retrieval_context(retrieved_memories: list[Memory]) -> str:
    formatted = []
    for memory in retrieved_memories:
        formatted.append(
            f"""MEMORY: {memory.id}
MEMORY TEXT: {memory.text}"""
        )

    return "\n\n".join(formatted)


# Assuming proposed_memory takes some form with three fields: text, memory_type, and scope
def process_new_memory(user_query: str) -> None:
    # Load current memories
    existing_memories = load_memories()
    
    # Extract key memories from user_query
    proposed_memories = extract_memories(user_query)
    
    for proposed_memory in proposed_memories:
        print(f"proposed_memory: {proposed_memory}")
        new_memory = create_memory(proposed_memory)

        related_memories = find_related_memories(new_memory.text, existing_memories, new_memory.scope)
        
        if not related_memories:
            existing_memories.append(new_memory)
        else:
            relationships = classify_memory_relationship(new_memory, related_memories)
            memory_update(new_memory, existing_memories, relationships)

    # Write the updated memories to the file
    persist_memories(existing_memories)


def memory_update(new_memory: Memory, memories: list[Memory], relationships: list[MemoryRelationship]) -> None:
    # ["duplicate", "superseding", "conflicting", "related"]
    conflict = False
    superseding_ids = set()
    for r in relationships:
        if r.relationship == "duplicate":
            return
        elif r.relationship == "conflicting":
            conflict = True
        elif r.relationship == "superseding":
            superseding_ids.add(r.id)

    for m in memories:
        if m.id in superseding_ids:
            m.status = "superseded"

    if conflict:
        new_memory.status = "conflicted"
    else:
        new_memory.status = "active"

    memories.append(new_memory)


def persist_memories(memories: list[Memory]) -> None:
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump([memory.model_dump() for memory in memories], 
                  f, indent=2, ensure_ascii=False)


def load_memories() -> list[Memory]:
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return [Memory.model_validate(item) for item in data]
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def create_memory(memory: ExtractedMemory) -> Memory:
    return Memory(
        id=str(uuid.uuid4()),
        text=memory.text,
        memory_type=memory.memory_type,
        scope=memory.scope,
        embedding=embed_text(memory.text),
        created_at=datetime.now(timezone.utc).isoformat(),
        status="active",
    )


def write_memories(queries: list[str]) -> None:
    persist_memories([])

    new_memories = []
    extracted_memories = extract_memories("\n".join(queries))
    for extracted_memory in extracted_memories:
        new_memories.append(create_memory(extracted_memory))

    persist_memories(new_memories)


def find_related_memories(query: str, memories: list[Memory], scope: str = "", top_k: int = MEMORY_RELATIONSHIP_TOP_K) -> list[Memory]:
    """Return a list of memory and score"""
    query_embedding = embed_text(query)
    results = []

    for memory in memories:
        if memory.status != "active":
            continue

        if not scope or memory.scope == scope:
            score = cosine_similarity(query_embedding, memory.embedding)
            results.append({"memory": memory, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    return [entry["memory"] for entry in results[:top_k]]


def classify_memory_relationship(new_memory: Memory, existing_memories: list[Memory]):
    new_memory = MemoryReference.from_memory(new_memory)
    existing_memories = [MemoryReference.from_memory(memory) for memory in existing_memories]
    
    prompt = f"""
    Find the semantic relationship between the new memory, namely NEW_MEMORY, and every existing memory, called EXISTING_MEMORY, in the collection EXISTING_MEMORIES
    Rules:
    - Return a list of MemoryRelationship, each representing the relationship between NEW_MEMORY and EXISTING_MEMORY
    - The attribute 'relationship' is defined as
        - 'duplicate': The new memory expresses essentially the same fact, preference, decision,
            or information as the existing memory. Minor differences in wording or
            additional non-material details do not make it a different memory.
        - 'superseding': The new memory describes a newer state, update, replacement, or change
            that makes the existing memory historical or outdated.
        - 'conflicting': The new and existing memories make incompatible claims, but there is not
            enough information to determine that the new memory represents a newer
            state that replaces the existing memory.
        - 'related': The new and existing memories concern the same or closely related subject,
            entity, project, or context, but they provide distinct information.
            Neither memory duplicates, supersedes, nor conflicts with the other.
    - The attribute 'id' is the id of the EXISTING_MEMORY
    - If none of the above relationship matches, then do not include the relationship MemoryRelationship between NEW_MEMORY and EXISTING_MEMORY in the result
    
    NEW_MEMORY:
    {new_memory}
    
    EXISTING_MEMORIES:
    {existing_memories}
    """

    response = get_client().chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=MemoryRelationshipList,
        temperature=0,
        extra_body={
            "think": False
        },
    )

    print(response.choices[0].message.parsed.relationships)

    return response.choices[0].message.parsed.relationships


def get_search_memory_declaration() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search persistent memory from previous interactions for "
                "user preferences, previous decisions, project-specific facts, "
                "and other remembered information. Do not use this to search "
                "shared company documents or policies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Search query describing the remembered information needed."
                        ),
                    },
                    "scope": {
                        "type": "string",
                        "description": (
                            "Optional scope used to restrict the memory search, "
                            "such as 'project_x'. Omit when no specific scope is known."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "Maximum number of relevant memories to retrieve. "
                            "Use a small value for specific questions and a larger "
                            "value for broad questions that require more complete coverage."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    }
